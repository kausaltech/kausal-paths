"""
Serialize DB-stored specs to the dict format InstanceLoader expects.

Reads InstanceConfig.spec and NodeConfig.spec (Pydantic models stored via
SchemaField) and converts them to the same dict structure that YAML config
parsing produces, so the existing InstanceLoader machinery can consume it.

Edges and DatasetPorts are still read from their Django models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nodes.defs.instance_defs import ActionGroupDef, InstanceSpec, ScenarioDef
    from nodes.defs.node_defs import NodeSpec
    from nodes.models import InstanceConfig, NodeConfig, NodeEdge
    from params.base import Parameter


def serialize_instance_to_dict(ic: InstanceConfig) -> dict[str, Any]:
    """Convert a DB-sourced InstanceConfig and its related models into a YAML-equivalent dict."""
    spec = ic.spec
    config = _serialize_instance_metadata(ic, spec)
    config['action_groups'] = [_serialize_action_group(ag) for ag in spec.action_groups]
    config['scenarios'] = [_serialize_scenario(s) for s in spec.scenarios]

    _add_nodes_and_edges(ic, config)
    config['dimensions'] = spec.dimensions
    return config


def _serialize_instance_metadata(ic: InstanceConfig, spec: InstanceSpec) -> dict[str, Any]:
    years = spec.years
    repo = spec.dataset_repo

    config: dict[str, Any] = {
        'id': ic.identifier,
        'default_language': ic.primary_language,
        'name': ic.name,
        'owner': ic.organization.name if ic.organization_id else '',
        'site_url': ic.site_url,
        'supported_languages': list(ic.other_languages or []),
        'target_year': years.target,
        'reference_year': years.reference,
        'minimum_historical_year': years.min_historical,
        'maximum_historical_year': years.max_historical,
        'model_end_year': years.model_end or years.target,
        'features': spec.features or {},
        'params': [_param_to_dict(p) for p in cast('Sequence[Parameter]', spec.params)],
        'dataset_repo': {
            'url': repo.url,
            'commit': repo.commit,
            'dvc_remote': repo.dvc_remote,
        },
    }
    return config


def _add_nodes_and_edges(ic: InstanceConfig, config: dict[str, Any]) -> None:
    from nodes.models import NodeConfig, NodeEdge

    node_configs = list(NodeConfig.objects.filter(instance=ic))
    edges = list(NodeEdge.objects.filter(instance=ic).select_related('from_node', 'to_node'))

    output_edges, _input_edges = _build_edge_maps(edges)

    nodes_list: list[dict[str, Any]] = []
    actions_list: list[dict[str, Any]] = []
    for nc in node_configs:
        node_dict = _serialize_node_config(
            nc,
            output_nodes=output_edges.get(nc.identifier, []),
        )
        spec = nc.spec
        if spec.type_config.kind == 'action':
            actions_list.append(node_dict)
        else:
            nodes_list.append(node_dict)

    config['nodes'] = nodes_list
    config['actions'] = actions_list


def _serialize_node_config(
    nc: NodeConfig,
    output_nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    spec: NodeSpec = nc.spec
    node: dict[str, Any] = {'id': nc.identifier}

    if nc.name:
        node['name'] = nc.name
    if nc.i18n:
        node.update(nc.i18n)

    # Python class path
    if spec.node_class:
        node['type'] = spec.node_class

    # Display fields (still on Django model)
    if nc.color:
        node['color'] = nc.color
    if nc.order is not None:
        node['order'] = nc.order
    if not nc.is_visible:
        node['is_visible'] = False

    # Spec-derived fields
    if spec.is_outcome:
        node['is_outcome'] = True
    if spec.minimum_year is not None:
        node['minimum_year'] = spec.minimum_year

    # Output metrics → unit/quantity (for single-metric nodes) or output_metrics list
    if spec.output_metrics:
        if len(spec.output_metrics) == 1:
            m = spec.output_metrics[0]
            if m.unit:
                node['unit'] = m.unit
            if m.quantity:
                node['quantity'] = m.quantity
        else:
            node['output_metrics'] = [{'id': m.id, 'unit': m.unit, 'quantity': m.quantity} for m in spec.output_metrics]

    # Parameters
    if spec.params:
        params = cast('Sequence[Parameter]', spec.params)
        node['params'] = [_param_to_dict(p) for p in params]

    # Computation
    if spec.pipeline is not None:
        node['pipeline'] = spec.pipeline
    if spec.input_ports:
        node['input_ports'] = [p.model_dump() for p in spec.input_ports]
    if spec.output_ports:
        node['output_ports'] = [p.model_dump() for p in spec.output_ports]

    # Type-config specifics
    tc = spec.type_config
    if tc.kind == 'formula':
        node['formula'] = tc.formula
    elif tc.kind == 'action' and tc.decision_level:
        node['decision_level'] = tc.decision_level

    # Datasets and dimensions from spec
    if spec.input_datasets:
        node['input_datasets'] = spec.input_datasets
    if spec.input_dimensions:
        node['input_dimensions'] = spec.input_dimensions
    if spec.output_dimensions:
        node['output_dimensions'] = spec.output_dimensions

    # Legacy extra fields
    extra = spec.extra
    if extra.historical_values:
        node['historical_values'] = extra.historical_values
    if extra.forecast_values:
        node['forecast_values'] = extra.forecast_values
    if extra.input_dataset_processors:
        node['input_dataset_processors'] = extra.input_dataset_processors
    if extra.tags:
        node['tags'] = extra.tags
    if extra.other:
        for key, val in extra.other.items():
            node.setdefault(key, val)

    # Edges (from Django models)
    if output_nodes:
        node['output_nodes'] = output_nodes

    return node


def _param_to_dict(p: Parameter) -> dict[str, Any]:
    """Serialize a Parameter to the dict format InstanceLoader expects."""
    from params.param import ReferenceParameter

    if isinstance(p, ReferenceParameter):
        return {'id': p.local_id, 'ref': p.target_id}
    d = p.model_dump(exclude_none=True)
    # InstanceLoader expects 'id', Pydantic model uses 'local_id'
    d['id'] = d.pop('local_id')
    return d


def _serialize_action_group(ag: ActionGroupDef) -> dict[str, Any]:
    result: dict[str, Any] = {'id': ag.id}
    if ag.name:
        result['name'] = str(ag.name)
    if ag.color:
        result['color'] = ag.color
    return result


def _serialize_scenario(scenario: ScenarioDef) -> dict[str, Any]:
    result: dict[str, Any] = {'id': scenario.id}
    if scenario.name:
        result['name'] = str(scenario.name)
    if scenario.description:
        result['description'] = str(scenario.description)
    if scenario.kind is not None and scenario.kind.value == 'default':
        result['default'] = True
    if scenario.all_actions_enabled:
        result['all_actions_enabled'] = True

    if scenario.params:
        params = []
        for override in scenario.params:
            param: dict[str, Any] = {'id': override.parameter_id, 'value': override.value}
            if override.node_id:
                param['node'] = override.node_id
            params.append(param)
        result['params'] = params

    return result


# ---------------------------------------------------------------------------
# Edge and dataset helpers (unchanged — these read from Django models)
# ---------------------------------------------------------------------------


def _build_edge_maps(
    edges: list[NodeEdge],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    output_edges: dict[str, list[dict[str, Any]]] = {}
    input_edges: dict[str, list[dict[str, Any]]] = {}

    for edge in edges:
        from_id = edge.from_node.identifier
        to_id = edge.to_node.identifier

        output_entry: dict[str, Any] = {'id': to_id}
        input_entry: dict[str, Any] = {'id': from_id}

        if edge.tags:
            for entry in (output_entry, input_entry):
                entry['tags'] = edge.tags
        # transformations is stored as {'from_dimensions': [...], 'to_dimensions': [...]}
        if edge.transformations:
            transforms = edge.transformations
            if isinstance(transforms, dict):
                if transforms.get('from_dimensions'):
                    for entry in (output_entry, input_entry):
                        entry['from_dimensions'] = transforms['from_dimensions']
                if transforms.get('to_dimensions'):
                    for entry in (output_entry, input_entry):
                        entry['to_dimensions'] = transforms['to_dimensions']
                if transforms.get('metrics'):
                    for entry in (output_entry, input_entry):
                        entry['metrics'] = transforms['metrics']

        output_edges.setdefault(from_id, []).append(output_entry)
        input_edges.setdefault(to_id, []).append(input_entry)

    return output_edges, input_edges
