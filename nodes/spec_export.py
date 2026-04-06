"""
Export a runtime Instance to InstanceSpec / NodeSpec for DB storage.

Given a fully loaded Instance (from YAML InstanceLoader), introspect
the live object graph and produce the Pydantic spec models that can be
stored on InstanceConfig.spec and NodeConfig.spec.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload
from uuid import uuid3, uuid4

from loguru import logger

from kausal_common.i18n.pydantic import TranslatedString, set_i18n_context

from nodes.actions.action import ActionNode
from nodes.datasets import DVCDataset
from nodes.defs import (
    ActionConfig,
    InputDatasetDef,
    InstanceSpec,
    NodeSpec,
    SimpleConfig,
    YearsSpec,
)
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.goals import NodeGoals
from nodes.models import NodeEdge
from nodes.visualizations import NodeVisualizations

if TYPE_CHECKING:
    from collections.abc import Iterable
    from uuid import UUID

    from kausal_common.i18n.pydantic import I18nString

    from nodes.context import Context
    from nodes.defs import (
        ActionGroup,
        FormulaConfig,
    )
    from nodes.defs.node_defs import NodeSpecExtra
    from nodes.edges import Edge, EdgeDimension
    from nodes.instance import Instance
    from nodes.models import InstanceConfig, NodeConfig
    from nodes.node import Node
    from nodes.scenario import Scenario
    from params import Parameter


@overload
def _to_ts(val: I18nString) -> TranslatedString: ...


@overload
def _to_ts(val: None) -> None: ...


def _to_ts(val: I18nString | None) -> TranslatedString | None:
    """Coerce an I18nString (str | TranslatedString | lazy) to TranslatedString | None."""
    if val is None:
        return None
    if isinstance(val, TranslatedString):
        return val
    return TranslatedString(str(val))


def export_instance_spec(instance: Instance) -> InstanceSpec:
    """Build an InstanceSpec from a live Instance."""
    ctx = instance.context

    years = YearsSpec(
        reference=instance.reference_year,
        min_historical=instance.minimum_historical_year,
        max_historical=instance.maximum_historical_year,
        target=ctx.target_year,
        model_end=ctx.model_end_year,
    )

    params = _export_global_params(ctx)
    action_groups = _export_action_groups(instance)
    scenarios = _export_scenarios(ctx)

    return InstanceSpec(
        uuid=instance.config.uuid,
        identifier=instance.id,
        name=_to_ts(instance.name),
        owner=_to_ts(instance.owner),
        primary_language=instance.default_language,
        other_languages=[lang for lang in instance.supported_languages if lang != instance.default_language],
        years=years,
        dataset_repo=ctx.dataset_repo_spec,
        dimensions=_export_dimensions(ctx),
        features=instance.features,
        terms=instance.terms,
        result_excels=[result.to_spec() for result in instance.result_excels],
        pages=[page.model_copy() for page in instance.pages],
        impact_overviews=[overview.spec.model_copy() for overview in ctx.impact_overviews],
        normalizations=[norm.spec.model_copy() for norm in ctx.normalizations.values()],
        params=params,
        action_groups=action_groups,
        scenarios=scenarios,
        theme_identifier=instance.theme_identifier,
    )


def export_node_spec(node: Node, nc: NodeConfig) -> NodeSpec:
    """Build a NodeSpec from a live Node."""
    type_config = _export_type_config(node)
    input_ports = _export_input_ports(node)
    output_ports = _export_output_ports(node)
    params = _export_node_params(node)

    # Capture dataset configs and dimension IDs.
    # Skip internal dimensions — they're created dynamically by node classes at runtime.
    input_datasets = _export_input_datasets(node)
    extra = _export_node_extra(node)
    input_dim_ids = [d for d, dim in node.input_dimensions.items() if not dim.is_internal] if node.input_dimensions else []
    output_dim_ids = [d for d, dim in node.output_dimensions.items() if not dim.is_internal] if node.output_dimensions else []

    uuid = nc.uuid
    if not nc.pk or not uuid:
        uuid = uuid4()
        nc.uuid = uuid

    return NodeSpec(
        uuid=uuid,
        kind=type_config.kind,
        identifier=node.id,
        name=_to_ts(node.name),
        description=_to_ts(node.description),
        color=node.db_obj.color if node.db_obj is not None else None,
        order=node.db_obj.order if node.db_obj is not None else None,
        is_visible=node.db_obj.is_visible if node.db_obj is not None else True,
        type_config=type_config,
        input_ports=input_ports,
        output_ports=output_ports,
        input_datasets=input_datasets,
        input_dimensions=input_dim_ids,
        output_dimensions=output_dim_ids,
        params=params,
        goals=node.goals.model_copy() if node.goals is not None else NodeGoals(),
        visualizations=node.visualizations.model_copy() if node.visualizations is not None else NodeVisualizations(),
        allow_nulls=node.allow_nulls,
        node_group=node.node_group,
        is_outcome=node.is_outcome,
        minimum_year=node.minimum_year,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Instance-level helpers
# ---------------------------------------------------------------------------


def _export_dimensions(ctx: Context) -> list[dict[str, Any]]:
    return [dim.model_dump(exclude_none=True) for dim in ctx.dimensions.values()]


def _export_global_params(ctx: Context) -> list[Parameter]:
    return [param.model_copy() for param in ctx.global_parameters.values()]


def _export_action_groups(instance: Instance) -> list[ActionGroup]:
    return [ag.model_copy(update={'order': idx}) for idx, ag in enumerate(instance.action_groups)]


def _export_scenarios(ctx: Context) -> list[Scenario]:
    from nodes.scenario import CustomScenario

    return [s for s in ctx.scenarios.values() if not isinstance(s, CustomScenario)]


# ---------------------------------------------------------------------------
# Node-level helpers
# ---------------------------------------------------------------------------


def _export_type_config(node: Node) -> FormulaConfig | ActionConfig | SimpleConfig:
    kls = type(node)
    node_class = f'{kls.__module__}.{kls.__qualname__}'

    if isinstance(node, ActionNode):
        return ActionConfig(
            decision_level=node.decision_level,
            group=node.group.id if node.group is not None else None,
            parent=node.parent_action.id if node.parent_action is not None else None,
            no_effect_value=node.no_effect_value,
            node_class=node_class,
        )

    assert not hasattr(node, 'formula')

    return SimpleConfig(node_class=node_class)


def uuid_from_identifiers(instance: Instance, identifiers: Iterable[str]) -> UUID:
    return uuid3(instance.config.uuid, ':'.join(identifiers))


def _export_input_ports(node: Node) -> list[InputPortDef]:
    """Build InputPortDefs from a node's incoming edges and input datasets."""
    ports: list[InputPortDef] = []

    for idx, dataset in enumerate(node.input_dataset_instances):
        port_id = uuid_from_identifiers(node.context.instance, [node.id, 'dataset', str(idx)])
        port = InputPortDef(
            id=port_id,
            unit=getattr(dataset, 'unit', None),
        )
        ports.append(port)

    node_counts: dict[str, int] = defaultdict(int)
    for edge in node.edges:
        if edge.output_node.id != node.id:
            continue
        from_node = edge.input_node
        if not edge.metrics:
            edge_metric_ids = [metric.column_id for metric in from_node.output_metrics.values()]
        else:
            edge_metric_ids = edge.metrics
        # if edge.tags:
        #     raise ValueError(f'Edge {from_node.id}:{node.id} has tags: {edge.tags}')
        seen_metric_ids = set[str]()
        for edge_metric_id in edge_metric_ids:
            # # First we need to hunt for the right metric; match by column_id
            # for from_metric_idx, from_metric in from_node.output_metrics.values():
            #     if metric.column_id == metric_id:
            #         break
            # else:
            #     raise ValueError(f'Metric {metric_id} not found in {from_node.id}')
            metrics_by_column_id = {metric.column_id: metric for metric in from_node.output_metrics.values()}
            from_metric = from_node.output_metrics.get(edge_metric_id)
            if from_metric is None:
                from_metric = metrics_by_column_id.get(edge_metric_id)
            if from_metric is None:
                raise ValueError(f'Metric {edge_metric_id} not found in {from_node.id}')

            #     if len(from_node.output_metrics) != 1:
            #         raise ValueError(f'Node {from_node.id} has multiple metrics: {from_node.output_metrics.keys()}')

            #     from_metric_id, from_metric = next(iter(from_node.output_metrics.items()))
            #     if from_metric.column_id != edge_metric_id:
            #         raise ValueError(f'Metric {edge_metric_id} not found in {from_node.id}')
            # else:
            #     from_metric_id = edge_metric_id
            assert from_metric.id not in seen_metric_ids
            seen_metric_ids.add(from_metric.id)
            port_id = uuid_from_identifiers(node.context.instance, [from_node.id, node.id, 'edge', from_metric.id])
            assert str(port_id) not in edge._to_port_ids
            edge._to_port_ids.append(str(port_id))
            assert from_metric.id not in edge._from_output_metric_ids
            edge._from_output_metric_ids.append(from_metric.id)
            port = InputPortDef(
                id=port_id,
                quantity=from_metric.quantity,
                unit=from_metric.unit,
                # TODO: multi & dimensions? tags? transformations?
                # supported_dimensions=src.supported_dimensions,
                # required_dimensions=src.required_dimensions,
            )
            port._from_node = edge.input_node.id
            port._edge_metric_id = from_metric.id
            ports.append(port)
        node_counts[edge.input_node.id] += 1

    return ports


def _export_output_ports(node: Node) -> list[OutputPortDef]:
    """Build OutputPortDefs from a node's runtime output metrics."""
    # Check whether the node class defines output_metrics at the class level.
    # If so, ports derived from those are non-editable.
    class_metric_ids: set[str] = set()
    class_metrics = getattr(type(node), 'output_metrics', None)
    if isinstance(class_metrics, dict):
        class_metric_ids = set(class_metrics.keys())

    ports: list[OutputPortDef] = []
    for metric_id, metric in node.output_metrics.items():
        assert metric.unit is not None
        port = OutputPortDef(
            id=uuid_from_identifiers(node.context.instance, [node.id, metric_id]),
            label=_to_ts(metric.label),
            unit=metric.unit,
            quantity=metric.quantity or None,
            column_id=metric.column_id,
            is_editable=metric_id not in class_metric_ids,
        )
        port._metric_id = metric_id
        ports.append(port)
    return ports


def _export_input_datasets(node: Node) -> list[InputDatasetDef]:
    """
    Export node input datasets as InputDatasetDef models.

    FixedDatasets are skipped — they are created from historical_values/forecast_values
    at the node config level, not from input_datasets.
    """
    from nodes.datasets import DatasetWithFilters

    result: list[InputDatasetDef] = []
    for ds in node.input_dataset_instances:
        if not isinstance(ds, DatasetWithFilters):
            continue
        result.append(
            InputDatasetDef(
                id=ds.id,
                tags=ds.tags or [],
                input_dataset=ds.input_dataset if isinstance(ds, DVCDataset) else None,
                column=ds.column,
                forecast_from=ds.forecast_from,
                filters=ds.filters or [],
                dropna=ds.dropna,
                min_year=ds.min_year,
                max_year=ds.max_year,
            )
        )
    return result


def _export_node_extra(node: Node) -> NodeSpecExtra:
    """Export legacy/attic fields from a runtime node."""
    from nodes.datasets import FixedDataset
    from nodes.defs.node_defs import NodeSpecExtra

    historical_values: list[tuple[int, float]] | None = None
    forecast_values: list[tuple[int, float]] | None = None
    for ds in node.input_dataset_instances:
        if isinstance(ds, FixedDataset):
            if ds.historical:
                historical_values = ds.historical
            if ds.forecast:
                forecast_values = ds.forecast

    # input_dataset_processors: check if node uses interpolation
    processors: list[str] = []
    for ds in node.input_dataset_instances:
        if getattr(ds, 'interpolate', False):
            processors = ['LinearInterpolation']
            break

    tags = list(node.tags) if node.tags else []

    return NodeSpecExtra(
        historical_values=historical_values,
        forecast_values=forecast_values,
        input_dataset_processors=processors,
        tags=tags,
    )


def _export_node_params(node: Node) -> list[Parameter]:
    """Export node-local parameters (including reference params)."""
    return [param.model_copy() for param in node.parameters.values()]


# ---------------------------------------------------------------------------
# Edge serialization
# ---------------------------------------------------------------------------


def serialize_edge_dimension(dim_id: str, ed: EdgeDimension) -> dict[str, Any]:
    """Serialize an EdgeDimension to the YAML-compatible dict format."""
    d: dict[str, Any] = {'id': dim_id}
    cat_ids = [c.id for c in ed.categories]
    if cat_ids:
        d['categories'] = cat_ids
    if ed.flatten:
        d['flatten'] = True
    if ed.exclude:
        d['exclude'] = True
    return d


def _resolve_from_port(edge: Edge, from_node: NodeSpec, metric_id: str) -> OutputPortDef:
    """Determine the output port ID for an edge's source side."""
    assert len(from_node.output_ports) >= 1
    ports = from_node.output_ports
    for port in ports:
        if port._metric_id == metric_id:
            return port

    raise ValueError(
        f'No port found for node {from_node.identifier} edge {edge.input_node.id}:{edge.output_node.id} metric {metric_id}'
    )


def _resolve_to_port(to_node: NodeSpec, from_node: NodeSpec, metric_id: str) -> InputPortDef:
    """Determine the input port ID for an edge's target side."""
    assert len(to_node.input_ports) >= 1
    for to_port in to_node.input_ports:
        if to_port._from_node != from_node.identifier:
            continue
        if to_port._edge_metric_id == metric_id:
            return to_port
    raise ValueError(
        f'No input port found for node {to_node.identifier} for edge from {from_node.identifier}, metric {metric_id}'
    )


def edge_to_transforms(edge: Edge) -> dict[str, Any]:
    """
    Convert runtime Edge dimension mappings to a dict with from/to separated.

    Also includes `metrics` (list of metric IDs to pass through), since
    NodeEdge doesn't have a dedicated field for it yet.
    """
    result: dict[str, Any] = {}
    if edge.from_dimensions:
        result['from_dimensions'] = [serialize_edge_dimension(dim_id, ed) for dim_id, ed in edge.from_dimensions.items()]
    if edge.to_dimensions:
        result['to_dimensions'] = [serialize_edge_dimension(dim_id, ed) for dim_id, ed in edge.to_dimensions.items()]
    # if edge.metrics:
    #    result['metrics'] = list(edge.metrics)
    return result


# ---------------------------------------------------------------------------
# Full sync: runtime → DB
# ---------------------------------------------------------------------------


def _update_edges(ic: InstanceConfig, ctx: Context, node_configs: dict[str, NodeConfig]) -> int:
    # Recreate edges
    NodeEdge.objects.filter(instance=ic).delete()
    edge_count = 0
    edge_objs = []
    for node in ctx.nodes.values():
        for edge in node.edges:
            if edge.input_node.id != node.id:
                continue  # only process outgoing edges from this node
            from_nc = node_configs.get(edge.input_node.id)
            to_nc = node_configs.get(edge.output_node.id)
            if not from_nc:
                raise ValueError(f'Source node {edge.input_node.id} not found in node configs')
            if not to_nc:
                raise ValueError(f'Target node {edge.output_node.id} not found in node configs')
            assert len(edge._to_port_ids)
            for from_metric_id, to_port_id in zip(edge._from_output_metric_ids, edge._to_port_ids, strict=True):
                from_port = _resolve_from_port(edge, from_nc.spec, from_metric_id)
                for to_port in to_nc.spec.input_ports:
                    if str(to_port.id) == to_port_id:
                        break
                else:
                    raise ValueError(
                        f'No input port found for node {to_nc.spec.identifier} for edge from '
                        + f'{from_nc.spec.identifier}, metric {from_metric_id}'
                    )
                edge_obj = NodeEdge(
                    instance=ic,
                    from_node=from_nc,
                    from_port=from_port.id,
                    to_node=to_nc,
                    to_port=to_port.id,
                    transformations=edge_to_transforms(edge),
                    tags=list(edge.tags) if edge.tags else [],
                )
                edge_objs.append(edge_obj)
            edge_count += 1
    NodeEdge.objects.bulk_create(edge_objs)
    return edge_count


def sync_instance_to_db(instance_id: str, yaml_path: str | Path | None = None) -> None:
    """
    Load an instance from YAML and sync its spec to the DB.

    If yaml_path is not given, tries configs/{instance_id}.yaml.
    """
    from django.db import transaction

    from nodes.dataset_placeholders import sync_instance_dataset_placeholders
    from nodes.instance_loader import InstanceLoader
    from nodes.models import InstanceConfig, NodeConfig

    if yaml_path is None:
        yaml_path = Path(f'configs/{instance_id}.yaml').resolve()
    else:
        yaml_path = Path(yaml_path).resolve()

    if not yaml_path.exists():
        raise FileNotFoundError(f'YAML file not found: {yaml_path}')

    loader = InstanceLoader.from_yaml(yaml_path)
    instance = loader.instance
    ctx = loader.context
    with transaction.atomic(), set_i18n_context(instance.default_language, instance.supported_languages):
        instance_spec = export_instance_spec(instance)
        ic, _created = InstanceConfig.objects.get_or_create(identifier=instance.id)
        ic.primary_language = instance.default_language
        ic.other_languages = [lang for lang in instance.supported_languages if lang != instance.default_language]
        ic.spec = instance_spec
        ic.config_source = 'database'
        ic.save()

        # Update or create node configs
        node_qs = ic.nodes.all().defer('spec')
        existing_ncs = {nc.identifier: nc for nc in node_qs}
        node_configs: dict[str, NodeConfig] = {}
        for node_id, node in ctx.nodes.items():
            nc = existing_ncs.get(node_id)
            if nc is None:
                nc = NodeConfig(instance=ic, identifier=node_id)
            # Use as_node_config_attributes to properly populate name + i18n
            conf = node.as_node_config_attributes()
            nc.name = conf.get('name', node_id)
            nc.i18n = conf.get('i18n', {})
            nc.spec = export_node_spec(node, nc)
            nc.is_stale = False
            nc.save()
            node_configs[node_id] = nc

        # Remove stale node configs
        stale_ids = set(existing_ncs.keys()) - set(node_configs.keys())
        if stale_ids:
            stale_nodes = ic.nodes.filter(identifier__in=stale_ids).defer('spec')
            logger.warning(f'Detected {len(stale_nodes)} stale nodes: {stale_nodes.values_list("identifier", flat=True)}')
            stale_nodes.update(is_stale=True)
            # NodeConfig.objects.filter(instance=ic, identifier__in=stale_ids, pages__isnull=True).defer('spec').delete()

        edge_count = _update_edges(ic, ctx, node_configs)

        created_placeholder_ids = sync_instance_dataset_placeholders(ic, ctx)

    logger.info(
        'Synced {id}: {nodes} nodes, {edges} edges, {placeholders} dataset placeholders created',
        id=instance.id,
        nodes=len(node_configs),
        edges=edge_count,
        placeholders=len(created_placeholder_ids),
    )
