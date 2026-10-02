"""Validate delivered values independently of whether a binding comes from a dataset or a node."""

from contextlib import nullcontext
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

import polars as pl

from common import qualifiers
from common.polars import DataFrameMeta, to_ppdf
from nodes.constants import VALUE_COLUMN, YEAR_COLUMN
from nodes.exceptions import NodeError
from nodes.units import unit_registry

if TYPE_CHECKING:
    from common.polars import PathsDataFrame
    from nodes.instance import Instance
    from nodes.node import Node


type ValueEnforcement = Literal['block_publish', 'block_submission']
"""
What a violation blocks.

``block_publish`` means the computed result would be wrong: a value that could not be
computed, one out of range, two rows where one is allowed, or an activity without the
factor it needs. ``block_submission`` means the result is not certifiable: a required
combination or quality grade is missing, or no inventory years have been declared. A
draft with such problems still publishes; a submission refuses it.
"""


class QualifierRequirement(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    min: float | None = None
    max: float | None = None
    present: bool = True

    @model_validator(mode='after')
    def validate_bounds(self) -> QualifierRequirement:
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError('Minimum must not exceed maximum')
        return self


class RequiredValueCombination(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    categories: dict[str, str]
    qualifiers: dict[str, QualifierRequirement] = Field(default_factory=dict)


class ValueContract(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    combinations: list[RequiredValueCombination] = Field(default_factory=list)
    years: Literal['inventory', 'active'] = 'inventory'
    required_if_positive: UUID | None = None
    combinations_from_positive: UUID | None = None
    max_rows: int | None = Field(default=None, ge=1)
    min: float | None = None
    max: float | None = None

    @model_validator(mode='after')
    def validate_bounds(self) -> ValueContract:
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError('Minimum must not exceed maximum')
        return self

    @property
    def enforcement(self) -> ValueEnforcement:
        """
        The tier of this contract's completeness and quality requirements.

        A contract conditioned on another input's positive values requires a factor for
        reported activity, and activity without its factor is a wrong result, not an
        incomplete one.
        """
        if self.required_if_positive is not None or self.combinations_from_positive is not None:
            return 'block_publish'
        return 'block_submission'


class ValueValidationViolation(BaseModel):
    node_uuid: UUID
    port_uuid: UUID
    binding_uuid: UUID | None = None
    code: str
    message: str
    years: list[int]
    categories: dict[str, str] = Field(default_factory=dict)
    enforcement: ValueEnforcement = 'block_publish'


def publication_blockers(violations: list[ValueValidationViolation]) -> list[ValueValidationViolation]:
    return [violation for violation in violations if violation.enforcement == 'block_publish']


def _qualifier_satisfies(rows: pl.DataFrame, path: str, requirement: QualifierRequirement) -> bool:
    column = qualifiers.qualifier_column(VALUE_COLUMN)
    if column not in rows.columns:
        return not requirement.present and requirement.min is None and requirement.max is None
    expression = pl.col(column)
    for part in path.split('.'):
        expression = expression.struct.field(part)
    try:
        values = rows.select(expression.alias('_Assessment'))['_Assessment']
    except pl.exceptions.StructFieldNotFoundError, pl.exceptions.InvalidOperationError:
        return False
    valid = values.is_not_null() if requirement.present else pl.Series([True] * len(values))
    if requirement.min is not None:
        valid &= (values >= requirement.min).fill_null(value=False)
    if requirement.max is not None:
        valid &= (values <= requirement.max).fill_null(value=False)
    return valid.all()


def _matching_combinations(df: pl.DataFrame, combination: RequiredValueCombination) -> pl.DataFrame:
    for dimension, category in combination.categories.items():
        if dimension not in df.columns:
            return df.clear()
        df = df.filter(pl.col(dimension).cast(pl.String) == category)
    return df


def _out_of_range_years(df: pl.DataFrame, contract: ValueContract, years: list[int]) -> list[int]:
    condition = (
        (~pl.col(VALUE_COLUMN).is_finite() & pl.col(VALUE_COLUMN).is_not_null())
        if contract.min is not None or contract.max is not None
        else pl.lit(value=False)
    )
    if contract.min is not None:
        condition |= pl.col(VALUE_COLUMN) < contract.min
    if contract.max is not None:
        condition |= pl.col(VALUE_COLUMN) > contract.max
    return sorted(df.filter(pl.col(YEAR_COLUMN).is_in(years) & condition)[YEAR_COLUMN].unique().to_list())


def _contract_requirements(
    df: PathsDataFrame, contract: ValueContract, years: list[int], required_values: PathsDataFrame | None
) -> list[tuple[RequiredValueCombination, list[int]]]:
    requirements = [(combination, years) for combination in contract.combinations]
    if required_values is None:
        return requirements
    dimensions = [dimension for dimension in df.dim_ids if dimension in required_values.dim_ids]
    metric = required_values.metric_cols[0]
    coordinates = (
        required_values
        .filter(pl.col(metric) > 0)
        .select([pl.col(YEAR_COLUMN), *[pl.col(dimension).cast(pl.String) for dimension in dimensions]])
        .unique()
        .sort([YEAR_COLUMN, *dimensions])
    )
    for row in coordinates.iter_rows(named=True):
        year = row[YEAR_COLUMN]
        if year in years:
            requirements.append((
                RequiredValueCombination(categories={dimension: row[dimension] for dimension in dimensions}),
                [year],
            ))
    return requirements


def validate_value_contract(
    df: PathsDataFrame,
    contract: ValueContract,
    years: list[int],
    *,
    node_uuid: UUID,
    port_uuid: UUID,
    binding_uuid: UUID | None = None,
    required_values: PathsDataFrame | None = None,
) -> list[ValueValidationViolation]:
    """Missing rows and nulls both fail a declared requirement; zero is an ordinary value."""
    present = df.filter(pl.col(VALUE_COLUMN).is_not_null() & pl.col(VALUE_COLUMN).is_finite())
    if contract.years == 'active':
        reporting_column = df.qualifier_cols.get(VALUE_COLUMN)
        if reporting_column is not None:
            active = df.filter(pl.col(reporting_column).struct.field(qualifiers.REPORTED).struct.field(qualifiers.ANY))
        else:
            active = present
        years = sorted(set(years) & set(active[YEAR_COLUMN].to_list()))
    problems: list[ValueValidationViolation] = []
    for combination, combination_years in _contract_requirements(df, contract, years, required_values):
        candidates = _matching_combinations(present, combination)
        for year in combination_years:
            rows = candidates.filter(pl.col(YEAR_COLUMN) == year)
            failures: list[tuple[str, str]] = []
            if rows.is_empty():
                failures.append(('missing_required_value', 'No value for the required combination'))
            else:
                for path, requirement in combination.qualifiers.items():
                    satisfied = _qualifier_satisfies(rows, path, requirement)
                    if not satisfied:
                        failures.append(('required_qualifier', f'Qualifier {path} does not meet the requirement'))
            for code, message in failures:
                problems.append(
                    ValueValidationViolation(
                        node_uuid=node_uuid,
                        port_uuid=port_uuid,
                        binding_uuid=binding_uuid,
                        code=code,
                        message=f'{message} in {year}: {combination.categories}',
                        years=[year],
                        categories=combination.categories,
                        enforcement=contract.enforcement,
                    )
                )
    problems.extend(_bounds_violations(df, contract, years, node_uuid=node_uuid, port_uuid=port_uuid, binding_uuid=binding_uuid))
    return problems


def _bounds_violations(
    df: PathsDataFrame, contract: ValueContract, years: list[int], *, node_uuid: UUID, port_uuid: UUID, binding_uuid: UUID | None
) -> list[ValueValidationViolation]:
    problems: list[ValueValidationViolation] = []
    bad_years = _out_of_range_years(df, contract, years)
    if bad_years:
        problems.append(
            ValueValidationViolation(
                node_uuid=node_uuid,
                port_uuid=port_uuid,
                binding_uuid=binding_uuid,
                code='value_range',
                message=f'Delivered values are outside the required range in {bad_years}',
                years=bad_years,
            )
        )
    if contract.max_rows is not None:
        excess = df.filter(pl.col(YEAR_COLUMN).is_in(years)).group_by(YEAR_COLUMN).len().filter(pl.col('len') > contract.max_rows)
        excess_years = sorted(excess[YEAR_COLUMN].to_list())
        if excess_years:
            problems.append(
                ValueValidationViolation(
                    node_uuid=node_uuid,
                    port_uuid=port_uuid,
                    binding_uuid=binding_uuid,
                    code='row_count',
                    message=f'At most {contract.max_rows} delivered rows allowed per year in {excess_years}',
                    years=excess_years,
                )
            )
    return problems


def _input_value(target: Node, port_id: UUID) -> PathsDataFrame | None:
    values = [binding.get_value() for binding in target.runtime_input_bindings if binding.target_port_id == port_id]
    if not values:
        return None
    value = values[0].rename({values[0].metric_cols[0]: VALUE_COLUMN})
    for additional in values[1:]:
        value = value.paths.add_with_dims(additional.rename({additional.metric_cols[0]: VALUE_COLUMN}), how='outer')
    return value


def _positive_input_years(target: Node, port_id: UUID, years: list[int]) -> list[int]:
    value = _input_value(target, port_id)
    if value is None:
        return []
    active = set(value.filter(pl.col(VALUE_COLUMN) > 0)[YEAR_COLUMN].to_list())
    return sorted(active.intersection(years))


def collect_instance_value_violations(  # noqa: C901, PLR0912
    instance: Instance, *, node_uuid: UUID | None = None, undeclared: Literal['report', 'evaluate'] = 'report'
) -> list[ValueValidationViolation]:
    """
    Read consumer-owned contracts through the graph's dataset-or-edge binding abstraction.

    Without a declared inventory calendar, an inventory contract's own requirements cannot
    be checked year by year: ``report`` replaces them with one ``inventory_years_undeclared``
    problem per port, while ``evaluate`` checks them over the whole historical span. The
    model computes with ``evaluate`` (`valid_inputs`), so declaring a calendar changes what
    is reported, not what was computed before it. Publication-tier checks run either way.
    """
    ctx = instance.context
    graph = ctx.instance_graph
    if graph is None:
        return []
    years_spec = graph.spec.years
    historical = years_spec.historical
    calendar_undeclared = historical is None and undeclared == 'report'
    years = (
        historical
        if historical is not None
        else list(
            range(
                instance.minimum_historical_year,
                (instance.maximum_historical_year or instance.reference_year) + 1,
            )
        )
    )
    problems: list[ValueValidationViolation] = []
    with nullcontext() if ctx.perf_run is not None else ctx.run():
        for meta in graph.nodes:
            if meta.identifier is None or (node_uuid is not None and meta.id != node_uuid):
                continue
            target = ctx.get_node(meta.identifier)
            for port in meta.spec.input_ports:
                if port.validation is None:
                    continue
                values: list[PathsDataFrame] = []
                for binding in target.runtime_input_bindings:
                    if binding.target_port_id != port.id:
                        continue
                    try:
                        values.append(binding.get_value())
                    except NodeError as exc:
                        problems.append(
                            ValueValidationViolation(
                                node_uuid=meta.id,
                                port_uuid=port.id,
                                binding_uuid=binding.id,
                                code='value_computation_error',
                                message=f'{exc}: {exc.__cause__}' if exc.__cause__ is not None else str(exc),
                                years=years,
                            )
                        )
                if not values and any(p.node_uuid == meta.id and p.port_uuid == port.id for p in problems):
                    continue
                if not values:
                    # A disconnected required input is as incomplete as a bound empty output.
                    empty = to_ppdf(
                        pl.DataFrame({YEAR_COLUMN: pl.Series([], dtype=pl.Int64), VALUE_COLUMN: pl.Series([], dtype=pl.Float64)}),
                        meta=DataFrameMeta(
                            units={VALUE_COLUMN: port.unit or unit_registry.dimensionless}, primary_keys=[YEAR_COLUMN]
                        ),
                    )
                    values.append(empty)
                values = [
                    value.rename({value.metric_cols[0]: VALUE_COLUMN})
                    if len(value.metric_cols) == 1 and value.metric_cols[0] != VALUE_COLUMN
                    else value
                    for value in values
                ]
                if port.unit is not None:
                    values = [value.ensure_unit(VALUE_COLUMN, port.unit) for value in values]
                value = values[0]
                for additional in values[1:]:
                    value = value.paths.add_with_dims(additional, how='outer')
                if (
                    calendar_undeclared
                    and port.validation.years == 'inventory'
                    and port.validation.enforcement == 'block_submission'
                ):
                    problems.append(
                        ValueValidationViolation(
                            node_uuid=meta.id,
                            port_uuid=port.id,
                            code='inventory_years_undeclared',
                            message='No inventory years have been declared, so the required values cannot be checked',
                            years=years,
                            enforcement='block_submission',
                        )
                    )
                    problems.extend(
                        _bounds_violations(value, port.validation, years, node_uuid=meta.id, port_uuid=port.id, binding_uuid=None)
                    )
                    continue
                required_years = years
                required_values = None
                try:
                    if port.validation.required_if_positive is not None:
                        required_years = _positive_input_years(target, port.validation.required_if_positive, years)
                    if port.validation.combinations_from_positive is not None:
                        required_values = _input_value(target, port.validation.combinations_from_positive)
                except NodeError as exc:
                    problems.append(
                        ValueValidationViolation(
                            node_uuid=meta.id,
                            port_uuid=port.id,
                            code='value_computation_error',
                            message=f'{exc}: {exc.__cause__}' if exc.__cause__ is not None else str(exc),
                            years=years,
                        )
                    )
                    continue
                problems.extend(
                    validate_value_contract(
                        value,
                        port.validation,
                        required_years,
                        node_uuid=meta.id,
                        port_uuid=port.id,
                        required_values=required_values,
                    )
                )
    return problems


class InstanceValueValidationError(Exception):
    def __init__(self, violations: list[ValueValidationViolation]) -> None:
        self.violations = violations
        super().__init__('; '.join(violation.message for violation in violations))
