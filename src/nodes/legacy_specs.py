"""Explicit adapters for pre-inheritance specs and v12 formula parameters."""

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from nodes.defs.instance_defs import InstanceModelSpec
from nodes.scenario import Scenario, ScenarioKind
from nodes.template_spec import DECLARATION_LISTS, declaration_identity, parameters_by_id
from params.base import ParameterOwner
from params.param import ReferenceParameter

if TYPE_CHECKING:
    from nodes.instance_serialization import InstanceSnapshot
    from nodes.template_settings import InheritedNodeSettings


def upgrade_formula_specs_v13(  # noqa: C901, PLR0912
    data: dict[str, Any], *, discard_captured_defaults: bool = False
) -> None:
    """Upgrade Wagtail's serialized JSON in place, never editing historical revision content."""
    formulas: dict[str, str] = {}
    defaults: dict[str, Any] = {}
    for parameter in data.get('spec', {}).get('params', []):
        defaults[parameter['local_id']] = parameter.get('value')
    for node in data.get('nodes', []):
        spec = node.get('spec')
        if not spec:
            continue
        config = spec.get('type_config', {})
        params = spec.get('params', [])
        legacy_formula = next((p for p in params if p['local_id'] == 'formula'), None)
        if config.get('node_class') in ('formula.FormulaNode', 'nodes.formula.FormulaNode'):
            if legacy_formula is None:
                raise ValueError(f'Legacy formula node {node.get("identifier")} has no formula')
            spec['type_config'] = {'kind': 'formula', 'formula': legacy_formula['value']}
            config = spec['type_config']
        if (
            config.get('node_class') in ('formula.FormulaAction', 'nodes.actions.formula.FormulaAction')
            and legacy_formula is not None
        ):
            config['formula'] = legacy_formula['value']
        if config.get('kind') == 'formula' or config.get('formula') is not None:
            formula = config['formula']
            if legacy_formula is not None and legacy_formula.get('value') != formula:
                raise ValueError(f'Conflicting formula declarations for {node.get("identifier")}')
            formulas[f'{node["identifier"]}.formula'] = formula
            spec['params'] = [p for p in params if p['local_id'] != 'formula']
        for parameter in spec.get('params', []):
            if 'value' in parameter:
                defaults[f'{node["identifier"]}.{parameter["local_id"]}'] = parameter['value']
    for scenario in data.get('spec', {}).get('scenarios', []):
        values = scenario.get('param_values', {})
        if discard_captured_defaults and scenario.get('kind') == 'default':
            for identifier in list(values):
                if identifier.endswith('.formula'):
                    del values[identifier]
        for identifier, formula in formulas.items():
            if identifier in values:
                if values[identifier] != formula:
                    raise ValueError(f'Scenario {scenario["id"]} changes formula {identifier}; migrate it explicitly')
                del values[identifier]
        if scenario.get('kind') == 'default':
            for identifier in list(values):
                if identifier in defaults and values[identifier] == defaults[identifier]:
                    del values[identifier]


def local_spec_from_template(local: InstanceModelSpec, base: InstanceSnapshot) -> InstanceModelSpec:  # noqa: C901, PLR0912
    """Turn an old copied spec into additions and explicit differences against its pinned edition."""
    result = InstanceModelSpec(years=local.years.model_copy(deep=True))
    template_params = parameters_by_id(base.spec, base.nodes)
    template_global = {p.local_id: p for p in base.spec.params}
    default = next((scenario for scenario in base.spec.scenarios if scenario.kind == ScenarioKind.DEFAULT), None)
    default_values = default.param_values if default else {}
    local_values = Scenario(id=default.id if default is not None else 'default', name='')
    for parameter in local.params:
        original = template_global.get(parameter.local_id)
        if original is None:
            result.params.append(parameter.model_copy(deep=True))
        elif original.owner != ParameterOwner.FRAMEWORK and original.type == parameter.type:
            value = parameter.model_dump(mode='json').get('value')
            inherited = default_values.get(parameter.local_id, original.model_dump(mode='json').get('value'))
            if value != inherited:
                local_values.param_values[parameter.local_id] = value
                local_values.parameter_types[parameter.local_id] = parameter.type
    if local_values.param_values and default is not None:
        target = result.local_scenario(default.id)
        target.param_values.update(local_values.param_values)
        target.parameter_types.update(local_values.parameter_types)
    shared_scenarios = {scenario.id: scenario for scenario in base.spec.scenarios}
    for scenario in local.scenarios:
        original = shared_scenarios.get(scenario.id)
        if original is None:
            result.scenarios.append(scenario.model_copy(deep=True))
            continue
        values = result.local_scenario(scenario.id)
        for identifier, value in scenario.param_values.items():
            parameter = template_params.get(identifier)
            if identifier.endswith('.formula'):
                # Old default snapshots captured formula parameters automatically.
                if scenario.kind != ScenarioKind.DEFAULT:
                    raise ValueError(f'Scenario {scenario.id} changes formula {identifier}; migrate it explicitly')
                continue
            if parameter is None or parameter.owner == ParameterOwner.FRAMEWORK:
                continue
            inherited = original.param_values.get(identifier, parameter.model_dump(mode='json').get('value'))
            if value != inherited:
                values.param_values[identifier] = deepcopy(value)
                values.parameter_types[identifier] = parameter.type
    for field in DECLARATION_LISTS:
        inherited_ids = {declaration_identity(field, item) for item in getattr(base.spec, field)}
        # Existing entries with template identities are copied declarations, even
        # if their contents have gone stale. Keep only independently named additions.
        setattr(
            result,
            field,
            [
                item.model_copy(deep=True)
                for item in getattr(local, field)
                if declaration_identity(field, item) not in inherited_ids
            ],
        )
    for field in ('terms', 'theme_identifier', 'sample_size'):
        value = getattr(local, field)
        inherited = getattr(base.spec, field)
        if hasattr(value, 'model_dump'):
            equal = value.model_dump(mode='json') == inherited.model_dump(mode='json') if inherited is not None else False
        else:
            equal = value == inherited
        if not equal:
            setattr(result, field, deepcopy(value))
    result.features = type(local.features).model_validate({
        field: value for field, value in local.features.model_dump().items() if value != base.spec.features.model_dump()[field]
    })
    result.scenarios = [scenario for scenario in result.scenarios if str(scenario.name) or scenario.param_values]
    return result


def migrate_inherited_node_settings(
    local: InstanceModelSpec,
    settings: list[InheritedNodeSettings],
    base: InstanceSnapshot,
) -> list[InheritedNodeSettings]:
    """Move legacy node values into municipal defaults and retire redundant or obsolete settings."""
    nodes = {node.uuid: node for node in base.nodes}
    default = next((scenario for scenario in base.spec.scenarios if scenario.default), None)
    default_id = default.id if default is not None else 'default'
    default_values = default.param_values if default is not None else {}
    result = []
    for original in settings:
        node = nodes.get(original.node_uuid)
        if node is None or node.spec is None:
            continue
        setting = original.model_copy(deep=True)
        parameters = {parameter.local_id: parameter for parameter in node.spec.params}
        for identifier, value in setting.parameter_values.items():
            parameter = parameters.get(identifier)
            if parameter is None or parameter.owner == ParameterOwner.FRAMEWORK:
                continue
            global_id = f'{node.identifier}.{identifier}'
            inherited = default_values.get(global_id, parameter.model_dump(mode='json').get('value'))
            if value != inherited:
                override = local.local_scenario(default_id)
                override.param_values.setdefault(global_id, value)
                override.parameter_types[global_id] = parameter.type
        setting.parameter_values = {}
        setting.parameter_sources = {
            identifier: source
            for identifier, source in setting.parameter_sources.items()
            if identifier in parameters and parameters[identifier].owner != ParameterOwner.FRAMEWORK
        }
        if setting.goals is not None and setting.goals.model_dump(mode='json') == node.spec.goals.model_dump(mode='json'):
            setting.goals = None
        if setting.layout is not None and node.layout is not None and setting.layout.model_dump() == node.layout.model_dump():
            setting.layout = None
        if setting.goals is not None or setting.layout is not None or setting.parameter_sources:
            result.append(setting)
    return result


def authoring_snapshot_from_legacy(snapshot: InstanceSnapshot) -> InstanceSnapshot:  # noqa: C901
    """Recover local authoring inputs from old flattened publications at the restore boundary."""
    from wagtail.models import Revision

    from nodes.instance_serialization import InputBindingOverrideSnapshot, InstanceSnapshot
    from nodes.template_graph import snapshot_content_hash
    from nodes.template_settings import InheritedNodeSettings

    revision = Revision.objects.get(pk=snapshot.template_revision_id)
    base = InstanceSnapshot.from_serialized_data(revision.content['model_snapshot']['structured'], compose=False)
    shared = {node.uuid: node for node in base.nodes}
    overrides = []
    ports = {(binding.node_id, binding.port_id) for binding in [*base.bindings, *snapshot.bindings] if binding.node_id in shared}
    for node_id, port_id in ports:
        inherited = [item for item in base.bindings if (item.node_id, item.port_id) == (node_id, port_id)]
        effective = [item for item in snapshot.bindings if (item.node_id, item.port_id) == (node_id, port_id)]
        if inherited != effective:
            overrides.append(InputBindingOverrideSnapshot(node_uuid=node_id, port_uuid=port_id, bindings=effective))
    settings = []
    for node in snapshot.nodes:
        original = shared.get(node.uuid)
        if original is None or original.spec is None or node.spec is None:
            continue
        selection = InheritedNodeSettings(node_uuid=node.uuid)
        if node.spec.goals != original.spec.goals:
            selection.goals = node.spec.goals
        if node.layout != original.layout:
            selection.layout = node.layout
        original_params = {parameter.local_id: parameter for parameter in original.spec.params}
        for parameter in node.spec.params:
            inherited = original_params.get(parameter.local_id)
            if inherited is None or inherited.owner == ParameterOwner.FRAMEWORK:
                continue
            if isinstance(parameter, ReferenceParameter):
                if not isinstance(inherited, ReferenceParameter) or parameter.target_id != inherited.target_id:
                    selection.parameter_sources[parameter.local_id] = parameter.target_id
            elif parameter.type == inherited.type and parameter.value != inherited.value:
                selection.parameter_values[parameter.local_id] = parameter.value
        if (
            selection.goals is not None
            or selection.layout is not None
            or selection.parameter_values
            or selection.parameter_sources
        ):
            settings.append(selection)
    inherited_datasets = {dataset.id for dataset in base.all_datasets()}
    result = snapshot.model_copy(
        update={
            'snapshot_kind': 'authored',
            'template_content_hash': snapshot_content_hash(base),
            'spec': local_spec_from_template(snapshot.spec, base.model_copy(deep=True)),
            'nodes': [node for node in snapshot.nodes if node.uuid not in shared],
            'bindings': [item for item in snapshot.bindings if item.node_id not in shared],
            'binding_overrides': overrides,
            'node_settings': settings,
            'dimensions': [
                item for item in snapshot.dimensions if item.id not in {dimension.id for dimension in base.dimensions}
            ],
            'datasets': [item for item in snapshot.datasets if item.id not in inherited_datasets],
            'dataset_revisions': [item for item in snapshot.dataset_revisions if item.dataset_uuid not in inherited_datasets],
        }
    )
    result.node_settings = migrate_inherited_node_settings(result.spec, settings, base.model_copy(deep=True))
    result._template = base
    return result
