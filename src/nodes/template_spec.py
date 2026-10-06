"""Compose declarations from a pinned method and sparse municipal additions and values."""

import json
from typing import TYPE_CHECKING

from django.core.exceptions import ValidationError

from nodes.defs.data_entry import ComposedDataEntrySpec, DataEntrySpec
from nodes.scenario import Scenario, ScenarioKind
from params.base import ParameterOwner
from params.param import ValidationError as ParameterValidationError

if TYPE_CHECKING:
    from pydantic import BaseModel

    from nodes.defs.instance_defs import InstanceModelSpec
    from nodes.instance_serialization import NodeSnapshot
    from params import Parameter


DECLARATION_KEYS = {
    'result_excels': 'name',
    'pages': 'id',
    'impact_overviews': 'id',
    'normalizations': 'normalizer_node_id',
    'action_groups': 'uuid',
}
DECLARATION_LISTS = tuple(DECLARATION_KEYS)


def declaration_identity(field: str, declaration: BaseModel) -> str:
    data = declaration.model_dump(mode='json')
    identity = data[DECLARATION_KEYS[field]]
    if identity is None and field == 'impact_overviews':
        identity = data['effect_node_id']
    return json.dumps(identity, sort_keys=True)


def parameters_by_id(spec: InstanceModelSpec, nodes: list[NodeSnapshot]) -> dict[str, Parameter]:
    # instance_graph imports snapshots, whose legacy adapters import this module.
    from nodes.instance_graph import node_class_for_spec

    parameters = {parameter.local_id: parameter for parameter in spec.params}
    for node in nodes:
        if node.spec is not None:
            node.spec.params = node_class_for_spec(node.spec).parameters_for_spec(node.spec)
            parameters.update({f'{node.identifier}.{parameter.local_id}': parameter for parameter in node.spec.params})
    return parameters


def compose_instance_spec(  # noqa: C901, PLR0912, PLR0915
    base: InstanceModelSpec, local: InstanceModelSpec, nodes: list[NodeSnapshot], *, errors: list[str]
) -> InstanceModelSpec:
    """Build an independent effective spec, suitable for freezing in a revision."""
    if local.dataset_repo is not None:
        raise ValueError('Template-based instances cannot override dataset_repo')
    result = base.model_copy(deep=True)
    if not result.scenarios:
        result.scenarios = [Scenario(id='default', name='Default', kind=ScenarioKind.DEFAULT)]
    result.years = local.years.model_copy(deep=True)
    if base.data_entry is not None or local.data_entry is not None:
        if isinstance(base.data_entry, ComposedDataEntrySpec) or isinstance(local.data_entry, ComposedDataEntrySpec):
            raise ValueError('Data-entry composition requires authored layouts')
        result.data_entry = ComposedDataEntrySpec(
            template=(base.data_entry or DataEntrySpec()).model_copy(deep=True),
            local=(local.data_entry or DataEntrySpec()).model_copy(deep=True),
        )
    shared_ids = {parameter.local_id for parameter in base.params}
    local_ids = [parameter.local_id for parameter in local.params]
    if shared_ids.intersection(local_ids) or len(local_ids) != len(set(local_ids)):
        raise ValueError('Local parameter declarations cannot shadow template parameters')
    result.params.extend(parameter.model_copy(deep=True) for parameter in local.params)
    for field in DECLARATION_LISTS:
        inherited = getattr(result, field)
        additions = getattr(local, field)
        inherited_ids = {declaration_identity(field, item) for item in inherited}
        if any(declaration_identity(field, item) in inherited_ids for item in additions):
            raise ValueError(f'Local {field} declarations cannot shadow template declarations')
        setattr(result, field, [item.model_copy(deep=True) for item in [*inherited, *additions]])
    shared_dimensions = {dimension['id'] for dimension in base.dimensions}
    if any(dimension['id'] in shared_dimensions for dimension in local.dimensions):
        raise ValueError('Local dimensions cannot shadow template dimensions')
    result.dimensions.extend(dict(dimension) for dimension in local.dimensions)
    for field in ('theme_identifier', 'sample_size'):
        if field in local.model_fields_set:
            setattr(result, field, getattr(local, field))
    for field in ('features', 'terms'):
        inherited = getattr(base, field)
        overrides = getattr(local, field).model_dump(mode='json', exclude_unset=True)
        setattr(result, field, type(inherited).model_validate({**inherited.model_dump(mode='json'), **overrides}))
    shared_scenarios = {scenario.id for scenario in result.scenarios}
    result.scenarios.extend(scenario.model_copy(deep=True) for scenario in local.scenarios if scenario.id not in shared_scenarios)
    if any(not str(scenario.name) for scenario in result.scenarios):
        raise ValueError('New local scenarios must have a name')
    result._is_composed = True
    parameters = parameters_by_id(result, nodes)
    for scenario in result.scenarios:
        if scenario.id in shared_scenarios:
            continue
        for identifier in list(scenario.param_values):
            parameter = parameters.get(identifier)
            if parameter is not None and parameter.owner == ParameterOwner.FRAMEWORK:
                errors.append(f'Scenario {scenario.id} overrides framework-owned parameter {identifier}')
                scenario.param_values.pop(identifier)
    for scenario in result.scenarios:
        override = next((item for item in local.scenarios if item.id == scenario.id), None)
        if override is None:
            continue
        for identifier, value in override.param_values.items():
            parameter = parameters.get(identifier)
            # An obsolete override no longer has meaning. Upgrade also prunes these entries.
            if parameter is None or parameter.owner == ParameterOwner.FRAMEWORK:
                continue
            if override.parameter_types.get(identifier, parameter.type) != parameter.type:
                continue
            try:
                cleaned = parameter.clean(value)
            except ParameterValidationError as exc:
                errors.append(f'Scenario {scenario.id}: {exc}')
                continue
            scenario.param_values[identifier] = cleaned
            if scenario.kind == ScenarioKind.DEFAULT:
                parameter.set(cleaned, notify=False)
    return result


def validate_spec_references(spec: InstanceModelSpec, nodes: list[NodeSnapshot]) -> None:
    """Validate authored reports/scenarios at publication, before runtime initialization."""
    identifiers = {node.identifier for node in nodes}
    parameters = parameters_by_id(spec, nodes)
    for report in spec.result_excels:
        missing = set((report.node_ids or []) + (report.action_ids or [])) - identifiers
        if missing:
            raise ValidationError(f'Report {report.name} references missing nodes: {", ".join(sorted(missing))}')
    for scenario in spec.scenarios:
        for identifier, value in scenario.param_values.items():
            parameter = parameters.get(identifier)
            if parameter is None:
                raise ValidationError(f'Scenario {scenario.id} references missing parameter {identifier}')
            try:
                parameter.clean(value)
            except ParameterValidationError as exc:
                raise ValidationError(str(exc)) from exc
