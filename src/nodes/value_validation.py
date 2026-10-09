"""Validate delivered values independently of whether a binding comes from a dataset or a node."""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

import polars as pl

from paths.refs import PortRef, ShapeRef

from common import qualifiers
from common.polars import DataFrameMeta, to_ppdf
from common.validation import blocks_operation
from nodes.constants import VALUE_COLUMN, YEAR_COLUMN
from nodes.exceptions import NodeError
from nodes.units import unit_registry

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from common.polars import PathsDataFrame
    from nodes.instance import Instance
    from nodes.node import Node
    from nodes.shapes import EffectiveShape


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


class ValueContract(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    shape: ShapeRef | None = None
    """
    A shape the delivered values must conform to: each of its required groups needs a value,
    and when it is closed, no value may fall outside its combinations.
    """
    required: bool = False
    """A value is required in each checked year, whatever its categories."""
    qualifiers: dict[str, QualifierRequirement] = Field(default_factory=dict)
    """What the year's values must meet besides being present; only with `required`."""
    years: Literal['inventory', 'active'] = 'inventory'
    required_if_positive: PortRef | None = None
    combinations_from_positive: PortRef | None = None
    max_rows: int | None = Field(default=None, ge=1)
    min: float | None = None
    max: float | None = None
    enforcement: ValueEnforcement | None = None
    """The tier of the completeness requirements, when the derived one does not fit; see `effective_enforcement`."""

    @model_validator(mode='after')
    def validate_bounds(self) -> ValueContract:
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError('Minimum must not exceed maximum')
        if self.qualifiers and not self.required:
            raise ValueError('Qualifier requirements apply to a required value; set required')
        return self

    @property
    def effective_enforcement(self) -> ValueEnforcement:
        """
        The tier of this contract's completeness and quality requirements.

        Unless declared, a contract conditioned on another input's positive values blocks
        publication: it usually requires a factor for reported activity, and activity
        without its factor is a wrong result, not an incomplete one. A condition whose
        absence only sends the model down a fallback route declares ``block_submission``.
        """
        if self.enforcement is not None:
            return self.enforcement
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
    return [violation for violation in violations if blocks_operation(violation.enforcement, 'publish')]


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


@dataclass(frozen=True, slots=True)
class ValueRequirement:
    """A value in at least one of the alternatives, meeting the qualifiers; one alternative is the usual case."""

    alternatives: tuple[Mapping[str, str], ...]
    qualifiers: Mapping[str, QualifierRequirement]
    identifier: str | None = None
    """The required group's identifier, when the requirement comes from a shape."""

    @property
    def categories(self) -> dict[str, str]:
        """The single alternative's categories, or none when the group offers a choice."""
        return dict(self.alternatives[0]) if len(self.alternatives) == 1 else {}


def contract_requirements(contract: ValueContract, shape: EffectiveShape | None) -> list[ValueRequirement]:
    """Return the contract's unconditional requirements: its year-level one and its shape's required groups."""
    requirements: list[ValueRequirement] = []
    if contract.required:
        requirements.append(ValueRequirement(alternatives=({},), qualifiers=contract.qualifiers))
    if shape is not None:
        categories = {combination.uuid: combination.categories for combination in shape.combinations}
        requirements.extend(
            ValueRequirement(
                alternatives=tuple(categories[member] for member in group.combinations),
                qualifiers=group.qualifiers,
                identifier=group.identifier,
            )
            for group in shape.required
        )
    return requirements


def requirement_failures(
    rows_by_alternative: Sequence[pl.DataFrame], qualifiers: Mapping[str, QualifierRequirement]
) -> list[tuple[str, str]]:
    """
    Check one requirement in one year, given each alternative's present rows.

    Satisfied when some alternative has values and all of the group's values meet the
    qualifiers: a quality grade required of grid-bound electricity holds for every sector
    that reports it, not for one of them.
    """
    present = [rows for rows in rows_by_alternative if not rows.is_empty()]
    if not present:
        return [('missing_required_value', 'No value')]
    return required_value_failures(pl.concat(present, how='vertical_relaxed'), qualifiers)


def _matching_combinations(df: pl.DataFrame, categories: Mapping[str, str]) -> pl.DataFrame:
    for dimension, category in categories.items():
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
    df: PathsDataFrame,
    contract: ValueContract,
    shape: EffectiveShape | None,
    years: list[int],
    required_values: PathsDataFrame | None,
) -> list[tuple[ValueRequirement, list[int]]]:
    requirements = [(requirement, years) for requirement in contract_requirements(contract, shape)]
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
                ValueRequirement(alternatives=({dimension: row[dimension] for dimension in dimensions},), qualifiers={}),
                [year],
            ))
    return requirements


def _describe_categories(categories: Mapping[str, str]) -> str:
    return ', '.join(f'{dimension}={category}' for dimension, category in categories.items())


def _describe_requirement(requirement: ValueRequirement) -> str:
    """Name a requirement for a message; a year-level requirement names none."""
    if len(requirement.alternatives) == 1:
        categories = requirement.alternatives[0]
        return f' for {_describe_categories(categories)}' if categories else ''
    choices = ' or '.join(f'({_describe_categories(categories)})' for categories in requirement.alternatives)
    return f' for {requirement.identifier}: any of {choices}'


def required_value_failures(
    rows: pl.DataFrame,
    requirements: Mapping[str, QualifierRequirement],
) -> list[tuple[str, str]]:
    """Check required values and assessments for both delivered values and source-entry views."""
    if rows.is_empty():
        return [('missing_required_value', 'No value')]
    return [
        ('required_qualifier', f'Qualifier {path} does not meet the requirement')
        for path, requirement in requirements.items()
        if not _qualifier_satisfies(rows, path, requirement)
    ]


def validate_value_contract(
    df: PathsDataFrame,
    contract: ValueContract,
    years: list[int],
    *,
    node_uuid: UUID,
    port_uuid: UUID,
    binding_uuid: UUID | None = None,
    required_values: PathsDataFrame | None = None,
    shape: EffectiveShape | None = None,
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
    for requirement, requirement_years in _contract_requirements(df, contract, shape, years, required_values):
        candidates = [_matching_combinations(present, categories) for categories in requirement.alternatives]
        for year in requirement_years:
            rows_by_alternative = [rows.filter(pl.col(YEAR_COLUMN) == year) for rows in candidates]
            failures = requirement_failures(rows_by_alternative, requirement.qualifiers)
            for code, message in failures:
                problems.append(
                    ValueValidationViolation(
                        node_uuid=node_uuid,
                        port_uuid=port_uuid,
                        binding_uuid=binding_uuid,
                        code=code,
                        message=f'{message} in {year}{_describe_requirement(requirement)}',
                        years=[year],
                        categories=requirement.categories,
                        enforcement=contract.effective_enforcement,
                    )
                )
    if shape is not None and shape.closed:
        problems.extend(
            _outside_shape_violations(
                present, shape, years, contract, node_uuid=node_uuid, port_uuid=port_uuid, binding_uuid=binding_uuid
            )
        )
    problems.extend(_bounds_violations(df, contract, years, node_uuid=node_uuid, port_uuid=port_uuid, binding_uuid=binding_uuid))
    return problems


def _outside_shape_violations(
    present: pl.DataFrame,
    shape: EffectiveShape,
    years: list[int],
    contract: ValueContract,
    *,
    node_uuid: UUID,
    port_uuid: UUID,
    binding_uuid: UUID | None,
) -> list[ValueValidationViolation]:
    """Values a closed shape does not allow, one violation per category tuple."""

    def violation(code: str, message: str, violation_years: list[int], categories: dict[str, str]) -> ValueValidationViolation:
        return ValueValidationViolation(
            node_uuid=node_uuid,
            port_uuid=port_uuid,
            binding_uuid=binding_uuid,
            code=code,
            message=message,
            years=violation_years,
            categories=categories,
            enforcement=contract.effective_enforcement,
        )

    present = present.filter(pl.col(YEAR_COLUMN).is_in(years))
    if present.is_empty():
        return []
    missing = [dimension for dimension in shape.dimensions if dimension not in present.columns]
    if missing:
        message = f'Delivered values have no {", ".join(missing)} dimension, which shape {shape.spec.label} constrains'
        return [violation('shape_dimension_missing', message, sorted(set(present[YEAR_COLUMN].to_list())), {})]
    allowed = shape.combination_keys()
    dimensions = list(shape.dimensions)
    tuples = (
        present
        .select([pl.col(YEAR_COLUMN), *[pl.col(dimension).cast(pl.String) for dimension in dimensions]])
        .group_by(dimensions)
        .agg(pl.col(YEAR_COLUMN).unique().sort())
        .sort(dimensions)
    )
    problems: list[ValueValidationViolation] = []
    for row in tuples.iter_rows(named=True):
        categories = {dimension: row[dimension] for dimension in dimensions}
        if tuple(sorted(categories.items())) in allowed:
            continue
        message = f'Value outside shape {shape.spec.label} for {_describe_categories(categories)}'
        problems.append(violation('outside_shape', message, list(row[YEAR_COLUMN]), categories))
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


def _unknown_shape(node_uuid: UUID, port_uuid: UUID, shape_id: UUID, years: list[int]) -> ValueValidationViolation:
    return ValueValidationViolation(
        node_uuid=node_uuid,
        port_uuid=port_uuid,
        code='unknown_shape',
        message=f'The input contract refers to shape {shape_id}, which is not declared',
        years=years,
    )


def collect_instance_value_violations(  # noqa: C901, PLR0912, PLR0915
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
                    and port.validation.effective_enforcement == 'block_submission'
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
                shape = graph.shapes.get(port.validation.shape) if port.validation.shape is not None else None
                if port.validation.shape is not None and shape is None:
                    problems.append(_unknown_shape(meta.id, port.id, port.validation.shape, years))
                    continue
                problems.extend(
                    validate_value_contract(
                        value,
                        port.validation,
                        required_years,
                        node_uuid=meta.id,
                        port_uuid=port.id,
                        required_values=required_values,
                        shape=shape,
                    )
                )
    return problems


class InstanceValueValidationError(Exception):
    def __init__(self, violations: list[ValueValidationViolation]) -> None:
        self.violations = violations
        super().__init__('; '.join(violation.message for violation in violations))
