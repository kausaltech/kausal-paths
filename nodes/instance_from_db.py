"""
The loader's build-side shim: InstanceSnapshot to loader-consumable dicts.

This is transitional (stage 1 of the loader inversion): the loader's build
half still consumes YAML-shaped dicts internally, so spec bundles are
converted back into dicts here. Each subsystem of the loader will migrate to
consuming the specs directly, at which point its portion of this module is
deleted.

DB-sourced instances get their snapshot from ``build_instance_snapshot``;
this module has no ORM access of its own.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

from django.utils.functional import Promise

from loguru import logger

from kausal_common.i18n.pydantic import TranslatedString

from nodes.constants import VALUE_COLUMN
from nodes.defs import ActionConfig, FormulaConfig, SimpleConfig
from nodes.defs.node_defs import NodeKind

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from kausal_common.i18n.pydantic import I18nString

    from nodes.defs.instance_defs import ActionGroup
    from nodes.defs.node_defs import NodeSpec
    from nodes.defs.transform_def import FilterDimensionOp, PortTransformOp
    from nodes.instance_serialization import (
        DatasetPortSnapshot,
        EdgeSnapshot,
        InstanceSnapshot,
        NodeSnapshot,
    )
    from nodes.models import InstanceConfig
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
    """Convert a DB-sourced InstanceConfig into a YAML-equivalent dict, via its snapshot."""
    from nodes.instance_serialization import build_instance_snapshot

    _check_dimension_orm_coverage(ic)
    return snapshot_to_config_dict(build_instance_snapshot(ic))


def snapshot_to_config_dict(snapshot: InstanceSnapshot) -> dict[str, Any]:
    """Convert an InstanceSnapshot into the YAML-equivalent dict the loader consumes."""
    spec = snapshot.spec
    config = _serialize_instance_metadata(snapshot)
    config['action_groups'] = [_serialize_action_group(ag) for ag in spec.action_groups]
    config['scenarios'] = [_serialize_scenario(s) for s in spec.scenarios]
    config['terms'] = spec.terms.model_dump(exclude_none=True)
    config['result_excels'] = [result.model_dump(exclude_none=True) for result in spec.result_excels]
    config['pages'] = [page.model_dump(exclude_none=True) for page in spec.pages]
    config['impact_overviews'] = [overview.model_dump(exclude_none=True) for overview in spec.impact_overviews]
    config['normalizations'] = [normalization.model_dump(exclude_none=True) for normalization in spec.normalizations]

    _add_nodes_and_edges(snapshot, config)
    config['dimensions'] = spec.dimensions
    return config


def _check_dimension_orm_coverage(ic: InstanceConfig) -> None:
    """
    Check the ORM covers spec.dimensions.

    Transitional: during the migration from `InstanceSpec.dimensions` to the
    ORM Dimension/DimensionCategory tables (plus their `spec` JSONFields),
    we keep both sources and verify the ORM is not missing anything the
    runtime needs. The computation model fails when a dim or cat is missing;
    extras or cosmetic diffs (labels, colors, aliases) only cause log noise.
    """
    spec = ic.spec
    assert spec is not None, f'InstanceConfig {ic.identifier!r} has no spec'
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


def _orm_category_ids_by_dim(ic: InstanceConfig) -> dict[str, set[str]]:
    from kausal_common.datasets.models import DimensionScope

    scopes = DimensionScope.objects.for_instance_config(ic).select_related('dimension').prefetch_related('dimension__categories')
    result: dict[str, set[str]] = {}
    for scope in scopes:
        assert scope.identifier is not None
        result[scope.identifier] = {cat.identifier for cat in scope.dimension.categories.all() if cat.identifier is not None}
    return result


def _serialize_instance_metadata(snapshot: InstanceSnapshot) -> dict[str, Any]:
    meta = snapshot.metadata
    spec = snapshot.spec
    years = spec.years
    repo = spec.dataset_repo

    config: dict[str, Any] = {
        'id': meta.identifier,
        'uuid': meta.uuid,
        'default_language': meta.primary_language,
        'supported_languages': meta.other_languages,
        'target_year': years.target,
        'reference_year': years.reference,
        'minimum_historical_year': years.min_historical,
        'maximum_historical_year': years.max_historical,
        'model_end_year': years.model_end or years.target,
        'features': spec.features.model_dump(),
        'params': [_param_to_dict(p) for p in cast('Sequence[Parameter]', spec.params)],
        'theme_identifier': spec.theme_identifier,
        'sample_size': spec.sample_size,
        **_ts_to_yaml('owner', meta.owner),
        **_ts_to_yaml('name', meta.name),
        **_ts_to_yaml('lead_title', meta.lead_title),
        **_ts_to_yaml('lead_paragraph', meta.lead_paragraph),
        **(
            {'dataset_repo': {'url': repo.url, 'commit': repo.commit, 'dvc_remote': repo.dvc_remote}} if repo and repo.url else {}
        ),
    }
    return config


def _add_nodes_and_edges(snapshot: InstanceSnapshot, config: dict[str, Any]) -> None:
    node_snapshots = snapshot.nodes
    specs_by_uuid: dict[UUID, NodeSpec] = {}
    identifiers_by_uuid: dict[UUID, str] = {}
    for n in node_snapshots:
        assert n.spec is not None, f'Node {n.uuid} has no spec'
        if n.identifier is None:
            raise ValueError(f'Node {n.uuid} has no identifier; the legacy runtime still requires one')
        specs_by_uuid[n.uuid] = n.spec
        identifiers_by_uuid[n.uuid] = n.identifier

    _output_edges, input_edges = _build_edge_maps(snapshot.edge_bindings, specs_by_uuid, identifiers_by_uuid)
    dataset_ports_by_node: defaultdict[UUID, list[DatasetPortSnapshot]] = defaultdict(list)
    for port in sorted(snapshot.dataset_bindings, key=lambda p: (p.node, p.dataset_index, str(p.port_id))):
        dataset_ports_by_node[port.node].append(port)

    nodes_list: list[dict[str, Any]] = []
    actions_list: list[dict[str, Any]] = []
    for n in node_snapshots:
        node_dict = _serialize_node_config(
            n,
            input_nodes=input_edges.get(n.uuid, []),
            dataset_ports=dataset_ports_by_node.get(n.uuid, []),
        )
        spec = specs_by_uuid[n.uuid]
        if spec.type_config.kind == NodeKind.ACTION:
            actions_list.append(node_dict)
        else:
            nodes_list.append(node_dict)

    config['nodes'] = nodes_list
    config['actions'] = actions_list


def _serialize_node_config(  # noqa: C901, PLR0912, PLR0915
    n: NodeSnapshot,
    input_nodes: list[dict[str, Any]],
    dataset_ports: list[DatasetPortSnapshot],
) -> dict[str, Any]:
    assert n.spec is not None
    if n.identifier is None:
        raise ValueError(f'Node {n.uuid} has no identifier; the legacy runtime still requires one')
    spec: NodeSpec = n.spec
    node: dict[str, Any] = {'id': n.identifier}

    if n.name:
        node.update(_ts_to_yaml('name', n.name))
    if n.short_name:
        node.update(_ts_to_yaml('short_name', n.short_name))

    # Python class path
    kind_config = spec.type_config
    if isinstance(kind_config, (ActionConfig, SimpleConfig)):
        node['type'] = kind_config.node_class
    elif isinstance(kind_config, FormulaConfig):
        node['type'] = 'formula.FormulaNode'
    else:
        raise TypeError(f'Unknown node type config: {type(kind_config)}')
    # Display fields
    if n.color:
        node['color'] = n.color
    if n.order is not None:
        node['order'] = n.order
    if not n.is_visible:
        node['is_visible'] = False
    if n.short_description:
        node['description'] = n.short_description

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

    # Edges (from the snapshot)
    # Use incoming edges here so the target node's input port order survives
    # the DB round-trip for order-sensitive nodes like MultiplicativeNode.
    if input_nodes:
        node['input_nodes'] = input_nodes

    return node


def _dataset_port_group_key(port: DatasetPortSnapshot) -> tuple[int, str]:
    """
    Identify the binding a port belongs to.

    ``dataset_index`` *is* binding identity — it is the position of the binding
    in the owning node's ``input_dataset_instances``, and several ports share it
    when one column-less binding expands to a port per metric. The dataset id is
    part of the key only as a safety net for rows written before
    ``dataset_index`` existed (migration 0043 has no backfill, so those all have
    index 0); a re-sync makes it redundant.
    """
    return (port.dataset_index, port.dataset)


def _serialize_dataset_ports(dataset_ports: list[DatasetPortSnapshot]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[DatasetPortSnapshot]] = {}
    for port in dataset_ports:
        grouped.setdefault(_dataset_port_group_key(port), []).append(port)

    input_datasets: list[dict[str, Any]] = []
    for (dataset_index, dataset_id), ports in grouped.items():
        first = ports[0]
        specs = {port.spec.model_dump_json(exclude_defaults=True, exclude_none=True) for port in ports}
        if len(specs) > 1:
            # The spec belongs to the binding, not to the individual port. A
            # divergence means the DB mirror is stale or was edited per-port;
            # the first port wins, and a re-sync fixes it.
            logger.warning(
                'Node {}: dataset ports for binding {} ({}) have {} differing specs; using the first',
                first.node,
                dataset_index,
                dataset_id,
                len(specs),
            )
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
# Edge helpers
# ---------------------------------------------------------------------------


def _filter_dimension_to_config(t: FilterDimensionOp) -> dict[str, Any]:
    d: dict[str, Any] = {'id': t.dimension}
    if t.categories:
        d['categories'] = list(t.categories)
    if t.groups:
        d['groups'] = list(t.groups)
    if t.flatten:
        d['flatten'] = True
    if t.exclude:
        d['exclude'] = True
    return d


def _transforms_to_config(
    transforms: Sequence[PortTransformOp],
    *,
    required_dimensions: Sequence[str] = (),
) -> dict[str, list[dict[str, Any]]]:
    """
    Convert a binding's transformations to the dict format Edge.from_config expects.

    This is the seam between the stored vocabulary and the legacy runtime: until
    ``_get_output_for_target()`` consumes the transform pipeline directly, only
    what these dicts can express is executable on an edge — which is why the
    edge mutations accept only the dimension-reshaping transformations.
    """
    from nodes.defs.transform_def import (
        AssignDimensionOp,
        FilterDimensionOp,
        FlattenTransformation,
        modernized_transformations,
    )

    from_dims: list[dict[str, Any]] = []
    to_dims: list[dict[str, Any]] = []

    # Pre-step-2 snapshots carry bare ``to_dimensions`` declarations as
    # FlattenTransformation entries. Preserve that immutable-history format at
    # this compatibility boundary while all newly built snapshots source the
    # declaration from InputPortDef.required_dimensions.
    legacy_declared_dimensions = [t.dimension for t in transforms if isinstance(t, FlattenTransformation)]
    declared_dimensions = list(dict.fromkeys([*required_dimensions, *legacy_declared_dimensions]))
    to_dims.extend({'id': dimension, 'exclude': True, 'flatten': True} for dimension in declared_dimensions)

    for t in modernized_transformations(transforms):
        match t:
            case FilterDimensionOp():
                from_dims.append(_filter_dimension_to_config(t))
            case AssignDimensionOp():
                to_dims.append({'id': t.dimension, 'categories': [t.category]})
            case _:
                raise ValueError(f'Edge transformation "{t.kind}" is not executable by the legacy edge runtime')
    result: dict[str, list[dict[str, Any]]] = {}
    if from_dims:
        result['from_dimensions'] = from_dims
    if to_dims:
        result['to_dimensions'] = to_dims
    return result


def _build_edge_maps(  # noqa: C901, PLR0912
    edges: Sequence[EdgeSnapshot],
    specs_by_uuid: dict[UUID, NodeSpec],
    identifiers_by_uuid: dict[UUID, str],
) -> tuple[dict[UUID, list[dict[str, Any]]], dict[UUID, list[dict[str, Any]]]]:
    output_edges: dict[UUID, list[dict[str, Any]]] = {}
    input_edges_with_order: defaultdict[UUID, list[tuple[int, dict[str, Any]]]] = defaultdict(list)

    edge_metrics: defaultdict[UUID, defaultdict[UUID, list[tuple[str, EdgeSnapshot]]]] = defaultdict(lambda: defaultdict(list))

    for edge in edges:
        from_spec = specs_by_uuid[edge.from_node]
        from_port = from_spec.output_port_by_id[edge.from_port]
        column_id = from_port.column_id or VALUE_COLUMN
        edge_metrics[edge.from_node][edge.to_node].append((column_id, edge))

    for from_node_id, to_nodes in edge_metrics.items():
        from_spec = specs_by_uuid[from_node_id]
        from_is_multi_metric = len(from_spec.output_ports) > 1
        for to_node_id, metric_tuples in to_nodes.items():
            from_entry: dict[str, Any] = {'id': identifiers_by_uuid[from_node_id]}
            to_entry: dict[str, Any] = {'id': identifiers_by_uuid[to_node_id]}

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
            to_spec = specs_by_uuid[to_node_id]
            to_port = to_spec.input_port_by_id[first_edge.to_port]
            if transforms or to_port.required_dimensions:
                config = _transforms_to_config(
                    transforms,
                    required_dimensions=to_port.required_dimensions,
                )
                if 'from_dimensions' in config:
                    for entry in (from_entry, to_entry):
                        entry['from_dimensions'] = config['from_dimensions']
                if 'to_dimensions' in config:
                    for entry in (from_entry, to_entry):
                        entry['to_dimensions'] = config['to_dimensions']
            output_edges.setdefault(from_node_id, []).append(to_entry)
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
