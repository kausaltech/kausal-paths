"""
Export a runtime Instance to InstanceSpec / NodeSpec for DB storage.

Given a fully loaded Instance (from YAML InstanceLoader), introspect
the live object graph and produce the Pydantic spec models that can be
stored on InstanceConfig.spec and NodeConfig.spec.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, overload
from uuid import uuid3, uuid4

from loguru import logger

from kausal_common.datasets.models import Dataset as DatasetModel, DatasetMetric
from kausal_common.i18n.pydantic import TranslatedString, set_i18n_context

from nodes.actions.action import ActionNode
from nodes.datasets import DatasetWithFilters, DVCDataset
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
    from nodes.defs.edge_def import EdgeTransformation
    from nodes.defs.node_defs import NodeSpecExtra
    from nodes.edges import Edge, EdgeDimension
    from nodes.instance import Instance
    from nodes.models import DatasetPort, InstanceConfig, NodeConfig
    from nodes.node import Node, NodeMetric
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

    goals = node.goals.model_copy() if node.goals is not None else NodeGoals()
    return NodeSpec(
        uuid=uuid,
        kind=type_config.kind,
        identifier=node.id,
        name=_to_ts(node.name),
        short_name=_to_ts(node.short_name),
        description=_to_ts(node.description),
        color=(node.db_obj.color if node.db_obj is not None and node.db_obj.color else None) or node.color,
        order=node.db_obj.order if node.db_obj is not None else None,
        is_visible=node.db_obj.is_visible if node.db_obj is not None else True,
        type_config=type_config,
        input_ports=input_ports,
        output_ports=output_ports,
        input_datasets=input_datasets,
        input_dimensions=input_dim_ids,
        output_dimensions=output_dim_ids,
        params=params,
        goals=goals,
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


@dataclass
class _InputPortMultiCandidate:
    port: InputPortDef
    old_port_id: UUID
    edge: Edge
    metric: NodeMetric
    group: str


def _effective_input_dimension_ids(node: Node, edge: Edge) -> tuple[str, ...]:
    if edge.to_dimensions is not None:
        return tuple(edge.to_dimensions.keys())
    return tuple(node.input_dimensions.keys())


def _is_multi_candidate_group_compatible(node: Node, candidates: list[_InputPortMultiCandidate]) -> bool:
    if not candidates:
        return False

    first = candidates[0]
    expected_dims = _effective_input_dimension_ids(node, first.edge)
    expected_unit = first.metric.unit
    expected_quantity = first.metric.quantity

    if node.unit is not None and not node.is_compatible_unit(expected_unit, node.unit):
        logger.warning(
            'Not marking %s input group %s as multi: metric %s unit %s is incompatible with target unit %s',
            node.id,
            first.group,
            first.metric.id,
            expected_unit,
            node.unit,
        )
        return False

    for candidate in candidates[1:]:
        dims = _effective_input_dimension_ids(node, candidate.edge)
        if set(dims) != set(expected_dims):
            logger.warning(
                'Not marking %s input group %s as multi: edge dimensions differ (%s vs %s)',
                node.id,
                candidate.group,
                sorted(dims),
                sorted(expected_dims),
            )
            return False
        if candidate.metric.quantity != expected_quantity:
            logger.warning(
                'Not marking %s input group %s as multi: metric quantities differ (%s vs %s)',
                node.id,
                candidate.group,
                candidate.metric.quantity,
                expected_quantity,
            )
            return False
        if not node.is_compatible_unit(candidate.metric.unit, expected_unit):
            logger.warning(
                'Not marking %s input group %s as multi: metric units differ dimensionally (%s vs %s)',
                node.id,
                candidate.group,
                candidate.metric.unit,
                expected_unit,
            )
            return False
        if node.unit is not None and not node.is_compatible_unit(candidate.metric.unit, node.unit):
            logger.warning(
                'Not marking %s input group %s as multi: metric %s unit %s is incompatible with target unit %s',
                node.id,
                candidate.group,
                candidate.metric.id,
                candidate.metric.unit,
                node.unit,
            )
            return False

    return True


def _input_port_group_id(node: Node, group: str) -> UUID:
    return uuid_from_identifiers(node.context.instance, [node.id, 'input-group', group])


def _replace_edge_to_port_id(edge: Edge, old_port_id: UUID, new_port_id: UUID) -> None:
    old_port_id_str = str(old_port_id)
    for idx, to_port_id in enumerate(edge._to_port_ids):
        if to_port_id == old_port_id_str:
            edge._to_port_ids[idx] = str(new_port_id)
            return
    raise ValueError(f'Port {old_port_id} not found in exported edge {edge.input_node.id}:{edge.output_node.id}')


def _apply_input_port_multi_hints(node: Node, ports: list[InputPortDef], candidates: list[_InputPortMultiCandidate]) -> None:
    by_group: defaultdict[str, list[_InputPortMultiCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_group[candidate.group].append(candidate)

    for group_candidates in by_group.values():
        if not _is_multi_candidate_group_compatible(node, group_candidates):
            continue
        first = group_candidates[0]
        group_port_id = _input_port_group_id(node, first.group)
        group_dimensions = list(_effective_input_dimension_ids(node, first.edge))

        first.port.id = group_port_id
        first.port.multi = True
        first.port.quantity = first.metric.quantity
        first.port.unit = node.unit or first.metric.unit
        first.port.required_dimensions = group_dimensions
        first.port.supported_dimensions = group_dimensions

        ports_to_remove = {candidate.old_port_id for candidate in group_candidates[1:]}
        for candidate in group_candidates:
            _replace_edge_to_port_id(candidate.edge, candidate.old_port_id, group_port_id)
        ports[:] = [port for port in ports if port.id not in ports_to_remove]


def _export_input_ports(node: Node) -> list[InputPortDef]:
    """Build InputPortDefs from a node's incoming edges and input datasets."""
    ports: list[InputPortDef] = []
    multi_candidates: list[_InputPortMultiCandidate] = []

    for idx, dataset in enumerate(node.input_dataset_instances):
        port_id = uuid_from_identifiers(node.context.instance, [node.id, 'dataset', str(idx)])
        port = InputPortDef(
            id=port_id,
            unit=getattr(dataset, 'unit', None),
        )
        ports.append(port)

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
            hint = node.input_port_multiplicity_hint(edge=edge, metric=from_metric)
            if hint.multi:
                group = hint.group or str(port_id)
                multi_candidates.append(
                    _InputPortMultiCandidate(
                        port=port,
                        old_port_id=port_id,
                        edge=edge,
                        metric=cast('NodeMetric', from_metric),
                        group=group,
                    )
                )
            port._from_node = edge.input_node.id
            port._edge_metric_id = from_metric.id
            ports.append(port)
    _apply_input_port_multi_hints(node, ports, multi_candidates)
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
    from nodes.datasets import DatasetWithFilters, FixedDataset

    result: list[InputDatasetDef] = []
    for ds in node.input_dataset_instances:
        if isinstance(ds, FixedDataset):
            continue
        assert isinstance(ds, DatasetWithFilters)
        ds_def = InputDatasetDef(
            id=ds.id,
            tags=ds.tags or [],
            input_dataset=ds.input_dataset if isinstance(ds, DVCDataset) else None,
            column=ds.column,
            forecast_from=ds.forecast_from,
            filters=ds.filters or [],
            dropna=ds.dropna,
            min_year=ds.min_year,
            max_year=ds.max_year,
            unit=ds.unit,
        )
        result.append(ds_def)
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


def edge_to_transforms(edge: Edge) -> list[EdgeTransformation]:
    """Convert runtime Edge dimension mappings to a structured transformation pipeline."""
    from nodes.defs.edge_def import AssignCategoryTransformation, FlattenTransformation, SelectCategoriesTransformation

    transforms: list[EdgeTransformation] = []

    for dim_id, ed in edge.from_dimensions.items():
        cat_refs = [cat.id for cat in ed.categories]
        transforms.append(
            SelectCategoriesTransformation(
                dimension=dim_id,
                categories=cat_refs,
                flatten=ed.flatten,
                exclude=ed.exclude,
            )
        )

    if edge.to_dimensions:
        for dim_id, ed in edge.to_dimensions.items():
            if not ed.categories:
                if ed.flatten:
                    # Flatten a dimension that the downstream node doesn't want.
                    transforms.append(FlattenTransformation(dimension=dim_id))
                # Entries with no categories and no flatten are pure shape
                # declarations — skip for now.
                continue
            if len(ed.categories) != 1:
                raise ValueError(f'to_dimensions can have only one category for now (got {len(ed.categories)} for {dim_id})')
            transforms.append(
                AssignCategoryTransformation(
                    dimension=dim_id,
                    category=ed.categories[0].id,
                )
            )

    return transforms


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
                from_spec = from_nc.spec
                assert from_spec is not None
                to_spec = to_nc.spec
                assert to_spec is not None
                from_port = _resolve_from_port(edge, from_spec, from_metric_id)
                for to_port in to_spec.input_ports:
                    if str(to_port.id) == to_port_id:
                        break
                else:
                    raise ValueError(
                        f'No input port found for node {to_spec.identifier} for edge from '
                        + f'{from_spec.identifier}, metric {from_metric_id}'
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


def _resolve_dataset_port(
    ic: InstanceConfig,
    nc: NodeConfig,
    node: Node,
    idx: int,
    ds_instance: DatasetWithFilters,
    placeholder_datasets: dict[str, DatasetModel],
    metrics_by_schema_and_name: dict[tuple[int, str], DatasetMetric],
) -> DatasetPort:
    from nodes.datasets import DBDataset
    from nodes.models import DatasetPort

    # The port_id must match the one generated in _export_input_ports
    port_id = uuid_from_identifiers(node.context.instance, [node.id, 'dataset', str(idx)])

    # Resolve the Dataset model object depending on the dataset type.
    if isinstance(ds_instance, DBDataset):
        dataset_obj = ds_instance.db_dataset_obj
        assert dataset_obj is not None
    elif isinstance(ds_instance, DVCDataset):
        dataset_obj = placeholder_datasets.get(ds_instance.id)
    else:
        raise TypeError(f'Unknown dataset type: {type(ds_instance)}')

    if dataset_obj is None:
        raise ValueError(f'No dataset object for {ds_instance.id} on node {node.id}')

    column = ds_instance.column
    if column is None or dataset_obj.schema is None:
        raise ValueError(
            f'Cannot create dataset port: column={column}, schema={dataset_obj.schema} for {ds_instance.id} on node {node.id}'
        )

    metric = metrics_by_schema_and_name.get((dataset_obj.schema.pk, column))
    if metric is None:
        raise ValueError(f'No metric {column} in dataset {ds_instance.id} for node {node.id}')

    return DatasetPort(
        instance=ic,
        node=nc,
        port_id=port_id,
        dataset=dataset_obj,
        metric=metric,
        forecast_from=ds_instance.forecast_from,
    )


def _get_placeholder_datasets(ic: InstanceConfig) -> dict[str, DatasetModel]:
    """
    Build a lookup of dataset identifier -> placeholder Dataset for an instance.

    Placeholder datasets are scoped through their schema's DatasetSchemaScope,
    not via the Dataset's own scope fields, so we query through the schema relation.
    """
    from django.contrib.contenttypes.models import ContentType

    from kausal_common.datasets.models import DatasetSchemaScope

    schema_scope_ids = DatasetSchemaScope.objects.filter(
        scope_content_type=ContentType.objects.get_for_model(ic),
        scope_id=ic.pk,
    ).values_list('schema_id', flat=True)
    return {
        ds.identifier: ds
        for ds in DatasetModel.objects.filter(
            schema_id__in=schema_scope_ids,
            is_external_placeholder=True,
        ).select_related('schema')
        if ds.identifier
    }


def _collect_dataset_schema_pks(ctx: Context, placeholder_datasets: dict[str, DatasetModel]) -> set[int]:
    """Collect schema PKs from both placeholder and DB-backed datasets."""
    from nodes.datasets import DBDataset

    pks: set[int] = set()
    for ds in placeholder_datasets.values():
        if ds.schema is not None:
            pks.add(ds.schema.pk)
    for node in ctx.nodes.values():
        for ds_instance in node.input_dataset_instances:
            if isinstance(ds_instance, DBDataset) and ds_instance.db_dataset_obj is not None:
                db_ds = ds_instance.db_dataset_obj
                if db_ds.schema is not None:
                    pks.add(db_ds.schema.pk)
    return pks


def _dataset_metric_binding_key(metric: DatasetMetric) -> str:
    """
    Return the metric identifier used in dataset-port bindings.

    DB-backed datasets deserialize their metric columns using the same fallback order:
    ``name``, then ``label``, then ``uuid``. Keep dataset-port lookup aligned with that
    runtime behavior so bindings resolve to the same effective metric column.
    """
    if metric.name:
        return metric.name
    if metric.label:
        return metric.label
    return str(metric.uuid)


def _update_dataset_ports(ic: InstanceConfig, ctx: Context, node_configs: dict[str, NodeConfig]) -> int:
    """Create DatasetPort objects linking placeholder datasets to node input ports."""
    from nodes.datasets import FixedDataset
    from nodes.models import DatasetPort

    DatasetPort.objects.filter(instance=ic).delete()

    placeholder_datasets = _get_placeholder_datasets(ic)
    all_schema_pks = _collect_dataset_schema_pks(ctx, placeholder_datasets)
    if not all_schema_pks:
        return 0

    # Build lookup: (schema_pk, metric_name) -> DatasetMetric
    metrics_by_schema_and_name: dict[tuple[int, str], DatasetMetric] = {}
    for metric in DatasetMetric.objects.filter(schema__pk__in=all_schema_pks):
        metrics_by_schema_and_name[(metric.schema.pk, _dataset_metric_binding_key(metric))] = metric

    port_objs: list[DatasetPort] = []
    for node in ctx.nodes.values():
        nc = node_configs.get(node.id)
        if nc is None:
            continue

        for idx, ds_instance in enumerate(node.input_dataset_instances):
            if isinstance(ds_instance, FixedDataset):
                continue
            if not isinstance(ds_instance, DatasetWithFilters):
                continue
            if ds_instance.column is None:
                continue

            port = _resolve_dataset_port(ic, nc, node, idx, ds_instance, placeholder_datasets, metrics_by_schema_and_name)
            port_objs.append(port)

    DatasetPort.objects.bulk_create(port_objs)
    return len(port_objs)


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
            nc.update_from_node(node, update_relations=False, skip_descriptions=True)
            spec = export_node_spec(node, nc)
            nc.is_stale = False
            nc.save()
            # Write spec via queryset.update() to bypass ClusterableModel.save()
            # which silently reverts SchemaField values.
            NodeConfig.objects.filter(pk=nc.pk).update(spec=spec)
            nc.spec = spec
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

        dataset_port_count = _update_dataset_ports(ic, ctx, node_configs)

    logger.info(
        'Synced {id}: {nodes} nodes, {edges} edges, {placeholders} dataset placeholders created, {ports} dataset ports',
        id=instance.id,
        nodes=len(node_configs),
        edges=edge_count,
        placeholders=len(created_placeholder_ids),
        ports=dataset_port_count,
    )
