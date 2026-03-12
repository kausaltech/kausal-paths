"""
Serialize Trailhead DB models to the dict format InstanceLoader expects.

This module converts the relational models (InstanceConfig, NodeConfig,
NodeEdge, DatasetPort, ActionGroup, Scenario) into the same dict structure
that YAML config parsing produces, so the existing InstanceLoader machinery
can consume it without changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nodes.models import InstanceConfig, NodeConfig, NodeEdge


def serialize_instance_to_dict(ic: InstanceConfig) -> dict[str, Any]:
    """Convert a DB-sourced InstanceConfig and its related models into a YAML-equivalent dict."""
    config = _serialize_instance_metadata(ic)
    _add_related_data(ic, config)
    return config


def _serialize_instance_metadata(ic: InstanceConfig) -> dict[str, Any]:
    config: dict[str, Any] = {
        'id': ic.identifier,
        'default_language': ic.primary_language,
        'name': ic.name,
        'owner': ic.organization.name if ic.organization_id else '',
        'site_url': ic.site_url,
        'supported_languages': list(ic.other_languages or []),
        'target_year': ic.target_year,
        'reference_year': ic.reference_year,
        'minimum_historical_year': ic.minimum_historical_year,
        'maximum_historical_year': ic.maximum_historical_year,
        'model_end_year': ic.model_end_year or ic.target_year,
        'emission_unit': ic.emission_unit or '',
        'features': ic.features or {},
        'parameters': ic.parameters or [],
    }

    if ic.dataset_repo_url:
        config['dataset_repo'] = {
            'url': ic.dataset_repo_url,
            'commit': ic.dataset_repo_commit,
            'dvc_remote': ic.dataset_repo_dvc_remote,
        }
    else:
        config['dataset_repo'] = {'url': '', 'commit': None, 'dvc_remote': None}

    # Merge catch-all extra fields
    if ic.extra:
        for key, val in ic.extra.items():
            config.setdefault(key, val)

    return config


def _add_related_data(ic: InstanceConfig, config: dict[str, Any]) -> None:
    from nodes.models import ActionGroup as ActionGroupModel, DatasetPort, NodeConfig, NodeEdge, Scenario as ScenarioModel

    # Action groups
    action_groups = ActionGroupModel.objects.filter(instance=ic).order_by('order')
    config['action_groups'] = [_serialize_action_group(ag) for ag in action_groups]

    # Pre-fetch related data
    node_configs = list(NodeConfig.objects.filter(instance=ic).select_related('action_group'))
    edges = list(NodeEdge.objects.filter(instance=ic).select_related('from_node', 'to_node'))
    dataset_ports = list(DatasetPort.objects.filter(instance=ic).select_related('node', 'dataset', 'metric'))

    # Build lookup maps
    output_edges, input_edges = _build_edge_maps(edges)
    datasets_by_node = _build_dataset_map(dataset_ports)

    # Nodes and actions
    nodes_list: list[dict[str, Any]] = []
    actions_list: list[dict[str, Any]] = []
    for nc in node_configs:
        node_dict = _serialize_node_config(
            nc,
            output_nodes=output_edges.get(nc.identifier, []),
            input_nodes=input_edges.get(nc.identifier, []),
            input_datasets=datasets_by_node.get(nc.identifier, []),
        )
        if nc.node_type == 'action':
            actions_list.append(node_dict)
        else:
            nodes_list.append(node_dict)
    config['nodes'] = nodes_list
    config['actions'] = actions_list

    # Scenarios
    scenarios = ScenarioModel.objects.filter(instance=ic)
    config['scenarios'] = [_serialize_scenario(s) for s in scenarios]

    # Dimensions placeholder
    config.setdefault('dimensions', [])


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

        if edge.transformations:
            for entry in (output_entry, input_entry):
                entry['tags'] = edge.tags or []
            _add_dimension_transforms(output_entry, edge.transformations)
            _add_dimension_transforms(input_entry, edge.transformations)

        output_edges.setdefault(from_id, []).append(output_entry)
        input_edges.setdefault(to_id, []).append(input_entry)

    return output_edges, input_edges


def _build_dataset_map(dataset_ports: list[Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for dp in dataset_ports:
        ds_entry: dict[str, Any] = {'id': dp.dataset.identifier}
        if dp.metric:
            ds_entry['metric'] = dp.metric.name
        result.setdefault(dp.node.identifier, []).append(ds_entry)
    return result


def _serialize_action_group(ag: Any) -> dict[str, Any]:
    result: dict[str, Any] = {'id': ag.identifier, 'name': ag.name}
    if ag.color:
        result['color'] = ag.color
    if ag.i18n:
        result.update(ag.i18n)
    return result


def _serialize_node_config(  # noqa: C901, PLR0912
    nc: NodeConfig,
    output_nodes: list[dict[str, Any]],
    input_nodes: list[dict[str, Any]],
    input_datasets: list[dict[str, Any]],
) -> dict[str, Any]:
    node: dict[str, Any] = {'id': nc.identifier}

    if nc.name:
        node['name'] = nc.name
    if nc.i18n:
        node.update(nc.i18n)

    # Python class path stored in extra during import
    if nc.extra and 'type' in nc.extra:
        node['type'] = nc.extra['type']

    for attr in ('quantity', 'unit', 'color', 'node_group'):
        val = getattr(nc, attr)
        if val:
            node[attr] = val

    if nc.order is not None:
        node['order'] = nc.order
    if nc.is_outcome:
        node['is_outcome'] = True
    if not nc.is_visible:
        node['is_visible'] = False
    if nc.params:
        node['params'] = nc.params
    if nc.input_ports:
        node['input_ports'] = nc.input_ports
    if nc.output_ports:
        node['output_ports'] = nc.output_ports
    if nc.pipeline is not None:
        node['pipeline'] = nc.pipeline
    if nc.formula:
        node['formula'] = nc.formula

    # Action-specific fields
    if nc.node_type == 'action':
        if nc.action_group:
            node['group'] = nc.action_group.identifier
        if nc.decision_level:
            node['decision_level'] = nc.decision_level

    if output_nodes:
        node['output_nodes'] = output_nodes
    if input_nodes:
        node['input_nodes'] = input_nodes
    if input_datasets:
        node['input_datasets'] = input_datasets

    # Merge extra fields (excluding 'type' which is handled above)
    if nc.extra:
        for key, val in nc.extra.items():
            if key != 'type':
                node.setdefault(key, val)

    return node


def _serialize_scenario(scenario: Any) -> dict[str, Any]:
    result: dict[str, Any] = {'id': scenario.identifier, 'name': scenario.name}
    if scenario.i18n:
        result.update(scenario.i18n)
    if scenario.description:
        result['description'] = scenario.description
    if scenario.kind == 'default':
        result['default'] = True
    if scenario.kind:
        result['kind'] = scenario.kind
    if scenario.all_actions_enabled:
        result['all_actions_enabled'] = True

    if scenario.parameter_overrides:
        params = []
        for override in scenario.parameter_overrides:
            param: dict[str, Any] = {
                'id': override.get('parameter_id', override.get('id', '')),
                'value': override.get('value'),
            }
            if override.get('node_id'):
                param['node'] = override['node_id']
            params.append(param)
        result['params'] = params

    return result


def _add_dimension_transforms(
    edge_entry: dict[str, Any],
    transformations: list[dict[str, Any]],
) -> None:
    """Convert edge transformations to from_dimensions/to_dimensions format."""
    from_dims: list[dict[str, Any]] = []
    to_dims: list[dict[str, Any]] = []

    for t in transformations:
        kind = t.get('kind', '')
        if kind == 'flatten':
            from_dims.append({'id': t['dimension'], 'flatten': True})
        elif kind == 'select_categories':
            entry: dict[str, Any] = {'id': t['dimension']}
            if t.get('categories'):
                entry['categories'] = t['categories']
            if t.get('flatten'):
                entry['flatten'] = True
            if t.get('exclude'):
                entry['exclude'] = True
            to_dims.append(entry)
        elif kind == 'assign_category':
            to_dims.append({'id': t['dimension'], 'categories': [t['category']]})

    if from_dims:
        edge_entry['from_dimensions'] = from_dims
    if to_dims:
        edge_entry['to_dimensions'] = to_dims
