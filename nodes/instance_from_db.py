"""
Serialize DB-stored specs to the dict format InstanceLoader expects.

Reads InstanceConfig.spec and NodeConfig.spec (Pydantic models stored via
SchemaField) and converts them to the same dict structure that YAML config
parsing produces, so the existing InstanceLoader machinery can consume it.

Edges and DatasetPorts are still read from their Django models.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

from django.db.models import F
from django.utils.functional import Promise

from loguru import logger

from kausal_common.i18n.pydantic import TranslatedString

from nodes.constants import VALUE_COLUMN
from nodes.defs import ActionConfig, FormulaConfig, SimpleConfig
from nodes.defs.edge_def import AssignCategoryTransformation, SelectCategoriesTransformation
from nodes.defs.node_defs import NodeKind

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kausal_common.i18n.pydantic import I18nString

    from nodes.defs.edge_def import EdgeTransformation
    from nodes.defs.instance_defs import ActionGroup, InstanceSpec
    from nodes.defs.node_defs import NodeSpec
    from nodes.models import DatasetPort, InstanceConfig, NodeConfig, NodeEdge
    from nodes.scenario import Scenario
    from params.base import Parameter


def _ts_to_yaml(field: str, val: I18nString | None) -> dict[str, str]:
    if val is None:
        return {}
    result: dict[str, str] = {}
    if isinstance(val, str):
        result[f'{field}'] = val
        return result
    if isinstance(val, TranslatedString):
        for lang, text in val.i18n.items():
            result[f'{field}_{lang}'] = text
    assert not isinstance(val, Promise)
    return result


def serialize_instance_to_dict(ic: InstanceConfig) -> dict[str, Any]:
    """Convert a DB-sourced InstanceConfig and its related models into a YAML-equivalent dict."""
    spec = ic.spec
    assert spec is not None, f'InstanceConfig {ic.identifier!r} has no spec'
    config = _serialize_instance_metadata(ic, spec)
    config['action_groups'] = [_serialize_action_group(ag) for ag in spec.action_groups]
    config['scenarios'] = [_serialize_scenario(s) for s in spec.scenarios]
    config['terms'] = spec.terms.model_dump(exclude_none=True)
    config['result_excels'] = [result.model_dump(exclude_none=True) for result in spec.result_excels]
    config['pages'] = [page.model_dump(exclude_none=True) for page in spec.pages]
    config['impact_overviews'] = [overview.model_dump(exclude_none=True) for overview in spec.impact_overviews]
    config['normalizations'] = [normalization.model_dump(exclude_none=True) for normalization in spec.normalizations]

    _add_nodes_and_edges(ic, config)
    config['dimensions'] = _resolve_dimensions(ic, spec)
    return config


def _resolve_dimensions(ic: InstanceConfig, spec: InstanceSpec) -> list[dict[str, Any]]:
    """
    Build the dimensions config and check the ORM covers spec.dimensions.

    Transitional: during the migration from `InstanceSpec.dimensions` to the
    ORM Dimension/DimensionCategory tables (plus their `spec` JSONFields),
    we keep both sources and verify the ORM is not missing anything the
    runtime needs. The computation model fails when a dim or cat is missing;
    extras or cosmetic diffs (labels, colors, aliases) only cause log noise.
    """
    orm_cats_by_dim = _orm_category_ids_by_dim(ic)
    missing: list[str] = []
    for dim_dict in spec.dimensions:
        dim_id = dim_dict['id']
        spec_cat_ids = {cat['id'] for cat in dim_dict.get('categories', [])}
        orm_cat_ids = orm_cats_by_dim.get(dim_id)
        if orm_cat_ids is None:
            missing.append(f'dim {dim_id!r} not present in ORM')
            continue
        missing_cats = spec_cat_ids - orm_cat_ids
        if missing_cats:
            missing.append(f'dim {dim_id!r}: missing cats {sorted(missing_cats)}')

    if missing:
        for line in missing:
            logger.error('Dimension ORM gap for {id}: {line}', id=ic.identifier, line=line)
        raise AssertionError(f'Dimension ORM missing entries for instance {ic.identifier!r}: {missing}')

    return spec.dimensions


def _orm_category_ids_by_dim(ic: InstanceConfig) -> dict[str, set[str]]:
    from kausal_common.datasets.models import DimensionScope

    scopes = DimensionScope.objects.for_instance_config(ic).select_related('dimension').prefetch_related('dimension__categories')
    result: dict[str, set[str]] = {}
    for scope in scopes:
        assert scope.identifier is not None
        result[scope.identifier] = {cat.identifier for cat in scope.dimension.categories.all() if cat.identifier is not None}
    return result


def _serialize_instance_metadata(ic: InstanceConfig, spec: InstanceSpec) -> dict[str, Any]:
    years = spec.years
    repo = spec.dataset_repo

    config: dict[str, Any] = {
        'id': spec.identifier or ic.identifier,
        'uuid': spec.uuid,
        'default_language': spec.primary_language,
        'site_url': ic.site_url,
        'supported_languages': spec.other_languages,
        'target_year': years.target,
        'reference_year': years.reference,
        'minimum_historical_year': years.min_historical,
        'maximum_historical_year': years.max_historical,
        'model_end_year': years.model_end or years.target,
        'features': spec.features.model_dump(),
        'params': [_param_to_dict(p) for p in cast('Sequence[Parameter]', spec.params)],
        'theme_identifier': spec.theme_identifier,
        **_ts_to_yaml('owner', spec.owner),
        **_ts_to_yaml('name', spec.name),
        **(
            {'dataset_repo': {'url': repo.url, 'commit': repo.commit, 'dvc_remote': repo.dvc_remote}} if repo and repo.url else {}
        ),
    }
    return config


def _add_nodes_and_edges(ic: InstanceConfig, config: dict[str, Any]) -> list[NodeConfig]:
    node_configs = ic.nodes_for_serialization
    edges = list(ic.edges.annotate(from_node_identifier=F('from_node__identifier'), to_node_identifier=F('to_node__identifier')))
    dataset_ports = list(
        ic.dataset_ports.select_related('node', 'dataset', 'metric').order_by(
            'node__identifier',
            'dataset_index',
            'metric__order',
            'port_id',
        )
    )

    nodes_by_identifier: dict[str, NodeConfig] = {nc.identifier: nc for nc in node_configs}

    _output_edges, input_edges = _build_edge_maps(cast('list[EdgeWithNodeIdentifiers]', edges), nodes_by_identifier)
    dataset_ports_by_node: defaultdict[str, list[DatasetPort]] = defaultdict(list)
    for port in dataset_ports:
        dataset_ports_by_node[port.node.identifier].append(port)

    nodes_list: list[dict[str, Any]] = []
    actions_list: list[dict[str, Any]] = []
    for nc in node_configs:
        node_dict = _serialize_node_config(
            nc,
            input_nodes=input_edges.get(nc.identifier, []),
            dataset_ports=dataset_ports_by_node.get(nc.identifier, []),
        )
        spec = nc.spec
        assert spec is not None
        if spec.type_config.kind == NodeKind.ACTION:
            actions_list.append(node_dict)
        else:
            nodes_list.append(node_dict)

    config['nodes'] = nodes_list
    config['actions'] = actions_list

    return node_configs


def _serialize_node_config(  # noqa: C901, PLR0912, PLR0915
    nc: NodeConfig,
    input_nodes: list[dict[str, Any]],
    dataset_ports: list[DatasetPort],
) -> dict[str, Any]:
    assert nc.spec is not None
    spec: NodeSpec = nc.spec
    node: dict[str, Any] = {'id': spec.identifier or nc.identifier}

    if spec.name:
        node.update(_ts_to_yaml('name', spec.name))
    elif nc.name:
        node['name'] = nc.name
    if nc.i18n:
        node.update(nc.i18n)
    if spec.short_name:
        node.update(_ts_to_yaml('short_name', spec.short_name))

    # Python class path
    kind_config = spec.type_config
    if isinstance(kind_config, (ActionConfig, SimpleConfig)):
        node['type'] = kind_config.node_class
    elif isinstance(kind_config, FormulaConfig):
        node['type'] = 'formula.FormulaNode'
    else:
        raise TypeError(f'Unknown node type config: {type(kind_config)}')
    # Display fields
    if spec.color:
        node['color'] = spec.color
    elif nc.color:
        node['color'] = nc.color
    if spec.order is not None:
        node['order'] = spec.order
    elif nc.order is not None:
        node['order'] = nc.order
    if spec.is_visible is False:
        node['is_visible'] = False
    elif not nc.is_visible:
        node['is_visible'] = False
    if spec.description:
        node['description'] = spec.description
    elif nc.short_description:
        node['description'] = nc.description

    # Spec-derived fields
    if spec.is_outcome:
        node['is_outcome'] = True
    if spec.minimum_year is not None:
        node['minimum_year'] = spec.minimum_year
    if spec.allow_nulls:
        node['allow_nulls'] = True
    if spec.node_group:
        node['node_group'] = spec.node_group

    # Output ports → unit/quantity (for single-port nodes) or output_metrics list
    if spec.output_ports:
        if len(spec.output_ports) == 1:
            d = spec.output_ports[0].model_dump(mode='json', exclude_defaults=True)
            node['unit'] = d['unit']
            if 'quantity' in d:
                node['quantity'] = d['quantity']
        else:
            node['output_metrics'] = [
                {
                    'id': p.column_id,
                    'column_id': p.column_id,
                    **p.model_dump(mode='json', include={'unit', 'quantity'}, exclude_defaults=True),
                }
                for p in spec.output_ports
            ]

    # Parameters
    if spec.params:
        params = cast('Sequence[Parameter]', spec.params)
        node['params'] = [_param_to_dict(p) for p in params]
    if spec.goals.root:
        node['goals'] = spec.goals.model_dump(exclude_none=True)
    if spec.visualizations.root:
        node['visualizations'] = spec.visualizations.model_dump(exclude_none=True)

    # Computation
    if spec.pipeline is not None:
        node['pipeline'] = spec.pipeline
    if spec.input_ports:
        node['input_ports'] = [p.model_dump(mode='json') for p in spec.input_ports]
    if spec.output_ports:
        node['output_ports'] = [p.model_dump(mode='json') for p in spec.output_ports]

    # Type-config specifics
    tc = spec.type_config
    if isinstance(tc, FormulaConfig):
        node['formula'] = tc.formula
    elif isinstance(tc, ActionConfig):
        if tc.decision_level:
            node['decision_level'] = tc.decision_level.as_str()
        if tc.group:
            node['group'] = tc.group
        if tc.parent:
            node['parent'] = tc.parent
        if tc.no_effect_value is not None:
            node['no_effect_value'] = tc.no_effect_value

    # Datasets from explicit port bindings; dimensions from spec.
    input_datasets = _serialize_dataset_ports(dataset_ports)
    if input_datasets:
        node['input_datasets'] = input_datasets
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
    # Use incoming edges here so the target node's input port order survives
    # the DB round-trip for order-sensitive nodes like MultiplicativeNode.
    if input_nodes:
        node['input_nodes'] = input_nodes

    return node


def _dataset_port_group_key(port: DatasetPort) -> tuple[str, str]:
    dataset_id = port.dataset.identifier or str(port.dataset.uuid)
    spec_json = port.spec.model_dump_json(exclude_defaults=True, exclude_none=True)
    return (dataset_id, spec_json)


def _serialize_dataset_ports(dataset_ports: list[DatasetPort]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[DatasetPort]] = {}
    for port in dataset_ports:
        grouped.setdefault(_dataset_port_group_key(port), []).append(port)

    input_datasets: list[dict[str, Any]] = []
    for ports in grouped.values():
        first = ports[0]
        dataset_id = first.dataset.identifier or str(first.dataset.uuid)
        ds_def = first.spec.to_input_dataset(id=dataset_id)
        input_datasets.append(ds_def.model_dump(mode='json', exclude_defaults=True, exclude_none=True))
    return input_datasets


def _param_to_dict(p: Parameter) -> dict[str, Any]:
    """Serialize a Parameter to the dict format InstanceLoader expects."""
    from params.param import ReferenceParameter

    if isinstance(p, ReferenceParameter):
        return {'id': p.local_id, 'ref': p.target_id}
    d = p.model_dump(exclude_none=True)
    # InstanceLoader expects 'id', Pydantic model uses 'local_id'
    d['id'] = d.pop('local_id')
    return d


def _serialize_action_group(ag: ActionGroup) -> dict[str, Any]:
    result: dict[str, Any] = {'id': ag.id}
    if ag.name:
        result.update(_ts_to_yaml('name', ag.name))
    result['color'] = ag.color
    return result


def _serialize_scenario(scenario: Scenario) -> dict[str, Any]:
    result: dict[str, Any] = {'id': scenario.id}
    result.update(_ts_to_yaml('name', scenario.name))
    result.update(_ts_to_yaml('description', scenario.description))
    if scenario.kind is not None and scenario.kind.value == 'default':
        result['default'] = True
    if scenario.all_actions_enabled:
        result['all_actions_enabled'] = True
    result['is_selectable'] = scenario.is_selectable
    if scenario.param_values:
        result['params'] = [{'id': k, 'value': v} for k, v in scenario.param_values.items()]
    return result


# ---------------------------------------------------------------------------
# Edge and dataset helpers (unchanged — these read from Django models)
# ---------------------------------------------------------------------------


if TYPE_CHECKING:

    class EdgeWithNodeIdentifiers(NodeEdge):
        from_node_identifier: str
        to_node_identifier: str


def _transforms_to_config(transforms: list[EdgeTransformation]) -> dict[str, list[dict[str, Any]]]:
    """Convert structured EdgeTransformation list to the dict format Edge.from_config expects."""
    from_dims: list[dict[str, Any]] = []
    to_dims: list[dict[str, Any]] = []
    from nodes.defs.edge_def import FlattenTransformation

    for t in transforms:
        if isinstance(t, SelectCategoriesTransformation):
            d: dict[str, Any] = {'id': t.dimension}
            if t.categories:
                d['categories'] = list(t.categories)
            if t.flatten:
                d['flatten'] = True
            if t.exclude:
                d['exclude'] = True
            from_dims.append(d)
        elif isinstance(t, AssignCategoryTransformation):
            to_dims.append({'id': t.dimension, 'categories': [t.category]})
        elif isinstance(t, FlattenTransformation):
            to_dims.append({'id': t.dimension, 'exclude': True, 'flatten': True})
    result: dict[str, list[dict[str, Any]]] = {}
    if from_dims:
        result['from_dimensions'] = from_dims
    if to_dims:
        result['to_dimensions'] = to_dims
    return result


def _build_edge_maps(  # noqa: C901, PLR0912
    edges: Sequence[EdgeWithNodeIdentifiers],
    nodes_by_identifier: dict[str, NodeConfig],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    output_edges: dict[str, list[dict[str, Any]]] = {}
    input_edges_with_order: defaultdict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)

    edge_metrics: defaultdict[str, defaultdict[str, list[tuple[str, EdgeWithNodeIdentifiers]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for edge in edges:
        from_spec = nodes_by_identifier[edge.from_node_identifier].spec
        assert from_spec is not None
        from_port = from_spec.output_port_by_id[edge.from_port]
        column_id = from_port.column_id or VALUE_COLUMN
        edge_metrics[edge.from_node_identifier][edge.to_node_identifier].append((column_id, edge))

    for from_node_id, to_nodes in edge_metrics.items():
        from_spec = nodes_by_identifier[from_node_id].spec
        assert from_spec is not None
        from_is_multi_metric = len(from_spec.output_ports) > 1
        for to_node_id, metric_tuples in to_nodes.items():
            from_entry: dict[str, Any] = {'id': from_node_id}
            to_entry: dict[str, Any] = {'id': to_node_id}

            metrics_entry: list[str] = []
            _, first_edge = metric_tuples[0]
            transforms = first_edge.transformations
            tags = first_edge.tags
            for metric_column_id, edge in metric_tuples:
                if transforms:
                    assert edge.transformations == transforms
                if tags:
                    assert tuple(edge.tags) == tuple(tags)
                metrics_entry.append(metric_column_id)

            # Only emit `metrics` when the source node has multiple output ports.
            # For single-output nodes YAML leaves `metrics` implicit, and the
            # runtime treats `metrics=[]` as pass-through. Emitting `['Value']`
            # activates a different code path that drops null-metric rows,
            # which breaks compute_impact when the action is disabled and the
            # input df has nulls.
            if from_is_multi_metric:
                for entry in (from_entry, to_entry):
                    entry['metrics'] = metrics_entry

            if tags:
                for entry in (from_entry, to_entry):
                    entry['tags'] = tags
            if transforms:
                config = _transforms_to_config(transforms)
                if 'from_dimensions' in config:
                    for entry in (from_entry, to_entry):
                        entry['from_dimensions'] = config['from_dimensions']
                if 'to_dimensions' in config:
                    for entry in (from_entry, to_entry):
                        entry['to_dimensions'] = config['to_dimensions']

            output_edges.setdefault(from_node_id, []).append(to_entry)
            to_spec = nodes_by_identifier[to_node_id].spec
            assert to_spec is not None
            input_port_order = {port.id: idx for idx, port in enumerate(to_spec.input_ports)}
            input_edges_with_order[to_node_id].append((
                input_port_order.get(first_edge.to_port, len(input_port_order)),
                from_entry,
            ))

    input_edges = {
        node_id: [entry for _, entry in sorted(entries, key=lambda item: item[0])]
        for node_id, entries in input_edges_with_order.items()
    }
    return output_edges, input_edges
