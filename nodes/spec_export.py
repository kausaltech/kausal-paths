"""
Export a runtime Instance to InstanceSpec / NodeSpec for DB storage.

Given a fully loaded Instance (from YAML InstanceLoader), introspect
the live object graph and produce the Pydantic spec models that can be
stored on InstanceConfig.spec and NodeConfig.spec.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload
from uuid import uuid3

from loguru import logger

from kausal_common.datasets.models import Dataset as DatasetModel, DatasetMetric
from kausal_common.i18n.pydantic import TranslatedString, set_i18n_context

from paths.identifiers import identifier_or_none

from nodes.actions.action import ActionNode
from nodes.constants import VALUE_COLUMN
from nodes.datasets import DatasetWithFilters, DVCDataset
from nodes.defs import (
    ActionConfig,
    DatasetPortSpec,
    InputDatasetDef,
    InstanceModelSpec,
    NodeSpec,
    SimpleConfig,
    YearsSpec,
)
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.goals import NodeGoals
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


def _apply_instance_metadata_columns(ic: InstanceConfig, instance: Instance) -> None:
    """
    Seed identity metadata from a runtime Instance onto the InstanceConfig columns.

    Identity (name, owner, languages) lives on the columns now, not in the spec.
    Existing DB-authored name and owner values take precedence.
    """
    ic.update_identity_metadata(
        name=instance.name,
        owner=instance.owner,
        primary_language=instance.default_language,
        other_languages=instance.supported_languages,
    )


def export_instance_spec(instance: Instance) -> InstanceModelSpec:
    """
    Build the computation-only ``InstanceModelSpec`` from a live Instance.

    Identity metadata (identifier, name, owner, languages, uuid) lives on the
    ``InstanceConfig`` columns, not in the spec.
    """
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

    return InstanceModelSpec(
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
        sample_size=ctx.sample_size,
    )


def export_node_spec(node: Node) -> NodeSpec:
    """Build the computation-only NodeSpec from a live Node."""
    type_config = _export_type_config(node)
    input_ports = _export_input_ports(node)
    output_ports = _export_output_ports(node)
    params = _export_node_params(node)

    # Capture dimension IDs.
    # Skip internal dimensions — they're created dynamically by node classes at runtime.
    extra = _export_node_extra(node)
    input_dim_ids = [d for d, dim in node.input_dimensions.items() if not dim.is_internal] if node.input_dimensions else []
    output_dim_ids = [d for d, dim in node.output_dimensions.items() if not dim.is_internal] if node.output_dimensions else []

    goals = node.goals.model_copy() if node.goals is not None else NodeGoals()
    return NodeSpec(
        type_config=type_config,
        input_ports=input_ports,
        output_ports=output_ports,
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
    ic = instance.config
    assert ic is not None
    return uuid3(ic.uuid, ':'.join(identifiers))


@dataclass
class _InputPortMultiCandidate:
    port: InputPortDef
    old_port_id: UUID
    edge: Edge
    metric: NodeMetric
    group: str
    role: str | None = None


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
            'Not marking %s input group %s as multi: metric %s unit %s is incompatible with target unit %s'
            % (
                node.id,
                first.group,
                first.metric.id,
                expected_unit,
                node.unit,
            )
        )
        return False

    for candidate in candidates[1:]:
        dims = _effective_input_dimension_ids(node, candidate.edge)
        if set(dims) != set(expected_dims):
            logger.warning(
                'Not marking %s input group %s as multi: edge dimensions differ (%s vs %s)'
                % (
                    node.id,
                    candidate.group,
                    sorted(dims),
                    sorted(expected_dims),
                )
            )
            return False
        if candidate.metric.quantity != expected_quantity:
            logger.warning(
                'Not marking %s input group %s as multi: metric quantities differ (%s vs %s)'
                % (
                    node.id,
                    candidate.group,
                    candidate.metric.quantity,
                    expected_quantity,
                )
            )
            return False
        if not node.is_compatible_unit(candidate.metric.unit, expected_unit):
            logger.warning(
                'Not marking %s input group %s as multi: metric units differ dimensionally (%s vs %s)'
                % (
                    node.id,
                    candidate.group,
                    candidate.metric.unit,
                    expected_unit,
                )
            )
            return False
        if node.unit is not None and not node.is_compatible_unit(candidate.metric.unit, node.unit):
            logger.warning(
                'Not marking %s input group %s as multi: metric %s unit %s is incompatible with target unit %s'
                % (
                    node.id,
                    candidate.group,
                    candidate.metric.id,
                    candidate.metric.unit,
                    node.unit,
                )
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
        first.port.identifier = identifier_or_none(first.group)
        first.port.role = identifier_or_none(first.role) if first.role is not None else None
        first.port.multi = True
        first.port.quantity = first.metric.quantity
        first.port.unit = node.unit or first.metric.unit
        first.port.required_dimensions = group_dimensions
        first.port.supported_dimensions = group_dimensions

        ports_to_remove = {candidate.old_port_id for candidate in group_candidates[1:]}
        for candidate in group_candidates:
            _replace_edge_to_port_id(candidate.edge, candidate.old_port_id, group_port_id)
        ports[:] = [port for port in ports if port is first.port or port.id not in ports_to_remove]


def _dataset_binding_columns_for_node(node: Node, ds_instance: DatasetWithFilters) -> list[str]:
    """
    Return dataset metric columns to expose as editor bindings.

    Single-column dataset inputs carry ``column`` directly. Multi-metric
    action datasets usually leave it unset and consume columns matching the
    node's output metrics.
    """
    if ds_instance.column is not None:
        return [ds_instance.column]
    columns: list[str] = []
    seen: set[str] = set()
    for metric in node.output_metrics.values():
        if metric.column_id is None:
            continue
        column = str(metric.column_id)
        if column not in seen:
            columns.append(column)
            seen.add(column)
    return columns


def _dataset_port_id(node: Node, dataset_index: int, column: str) -> UUID:
    return uuid_from_identifiers(node.context.instance, [node.id, 'dataset', str(dataset_index), column])


def pair_metrics_to_columns(columns: list[str], metric_keys: list[str], *, log_ctx: str) -> list[tuple[str, str]]:
    """
    Pair a column-less binding's dataset schema metrics with node columns.

    Returns ``(port_column, metric_key)`` pairs: which input port delivers
    which source metric. Name matches pair first (case-insensitively, since
    schema metrics are lowercase identifiers while node columns are often
    TitleCase, e.g. ``fuel`` feeding ``Fuel``); a lone leftover on both sides
    pairs too, since a single remaining metric can only feed the single
    remaining column. Anything still unmatched gets no binding — inventing a
    mapping would be worse than omitting it, and a dangling binding worse
    than a missing one.
    """
    pairs: list[tuple[str, str]] = []
    remaining_metrics = list(metric_keys)
    remaining_columns = list(columns)
    for column in columns:
        match = next((metric for metric in remaining_metrics if metric.lower() == column.lower()), None)
        if match is not None:
            pairs.append((column, match))
            remaining_metrics.remove(match)
            remaining_columns.remove(column)
    if len(remaining_metrics) == 1 and len(remaining_columns) == 1:
        pairs.append((remaining_columns[0], remaining_metrics[0]))
        remaining_metrics.clear()
    if remaining_metrics:
        logger.warning('%s: no input port column for schema metrics %s; they get no binding' % (log_ctx, remaining_metrics))
    return pairs


def _pair_schema_metrics_to_columns(
    node: Node,
    ds_instance: DatasetWithFilters,
    metric_keys: list[str],
) -> list[tuple[str, str]]:
    columns = _dataset_binding_columns_for_node(node, ds_instance)
    return pair_metrics_to_columns(columns, metric_keys, log_ctx=f'Dataset {ds_instance.id} on node {node.id}')


def _metric_for_column(node: Node, column: str) -> NodeMetric | None:
    for metric in node.output_metrics.values():
        if str(metric.column_id) == column:
            return metric
    return None


def _port_identifier_for_column(column: str) -> str | None:
    """
    Derive a port identifier from the bound dataset column, if it makes a usable name.

    A port identifier is meant to be a name a person uses — a formula variable,
    later on. The generic ``Value`` column says nothing about what the port
    carries, and legacy wide-DVC columns are not identifier-shaped at all. Those
    ports stay unnamed rather than filling the namespace with noise.
    """
    if column == VALUE_COLUMN:
        return None
    return identifier_or_none(column)


def _binding_pairs_for_dataset(
    node: Node,
    ds_instance: DatasetWithFilters,
    dataset_obj: DatasetModel,
    metrics_by_schema_and_name: dict[tuple[int, str], DatasetMetric],
) -> list[tuple[str, str]]:
    """
    Return the ``(port_column, metric_key)`` pairs a dataset binds through.

    For column-less bindings the node consumes the full frame, and the dataset
    schema may name its metrics differently from the node's columns (e.g.
    metric ``share`` feeding the generic ``Value`` column). The pairing keeps
    the two concepts distinct: the port UUID is always derived from the
    node-side column — the same key ``_export_dataset_input_ports`` uses — so
    a binding always points at a port that exists in the node spec, while the
    metric names the source.
    """
    if ds_instance.column is not None:
        return [(ds_instance.column, ds_instance.column)]
    assert dataset_obj.schema is not None
    schema_metrics = [name for (schema_pk, name) in metrics_by_schema_and_name if schema_pk == dataset_obj.schema.pk]
    if not schema_metrics:
        return [(column, column) for column in _dataset_binding_columns_for_node(node, ds_instance)]
    pairs = _pair_schema_metrics_to_columns(node, ds_instance, schema_metrics)
    if not pairs:
        # No defensible pairing at all — but these rows are what
        # `_serialize_dataset_ports` rebuilds `input_datasets` from, so
        # dropping every row would remove the dataset from the model on
        # DB-sourced instances. Keep rows keyed by the schema metric names:
        # their port ids dangle (and are warned about), which is a lesser
        # evil than losing the input.
        logger.warning(
            'Dataset %s on node %s: keeping bindings with unresolved port ids so the input dataset survives'
            % (ds_instance.id, node.id),
        )
        pairs = [(name, name) for name in schema_metrics]
    return pairs


def _dataset_input_port_for_column(
    node: Node,
    dataset_index: int,
    dataset: DatasetWithFilters,
    column: str,
) -> InputPortDef:
    metric = _metric_for_column(node, column)
    return InputPortDef(
        id=_dataset_port_id(node, dataset_index, column),
        identifier=_port_identifier_for_column(column),
        unit=metric.unit if metric is not None else getattr(dataset, 'unit', None),
        quantity=metric.quantity if metric is not None else None,
    )


def _export_dataset_input_ports(node: Node) -> list[InputPortDef]:
    ports: list[InputPortDef] = []
    for idx, dataset in enumerate(node.input_dataset_instances):
        if not isinstance(dataset, DatasetWithFilters):
            continue
        ports.extend(
            _dataset_input_port_for_column(node, idx, dataset, column)
            for column in _dataset_binding_columns_for_node(node, dataset)
        )
    return ports


def _export_input_ports(node: Node) -> list[InputPortDef]:
    """Build InputPortDefs from a node's incoming edges and input datasets."""
    ports = _export_dataset_input_ports(node)
    multi_candidates: list[_InputPortMultiCandidate] = []

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
            if len(from_node.output_metrics) > 1:
                port_identifier = identifier_or_none(f'{from_node.id}_{from_metric.id}')
            else:
                port_identifier = identifier_or_none(from_node.id)
            port = InputPortDef(
                id=port_id,
                identifier=port_identifier,
                quantity=from_metric.quantity,
                unit=from_metric.unit,
                required_dimensions=[
                    dim_id for dim_id, dimension in (edge.to_dimensions or {}).items() if not getattr(dimension, 'categories', ())
                ],
                # TODO: multi & dimensions? tags? transformations?
                # supported_dimensions=src.supported_dimensions,
            )
            hint = node.input_port_multiplicity_hint(edge=edge, metric=from_metric)
            if hint.multi:
                group = hint.group or str(port_id)
                multi_candidates.append(
                    _InputPortMultiCandidate(
                        port=port,
                        old_port_id=port_id,
                        edge=edge,
                        metric=from_metric,
                        group=group,
                        role=hint.role,
                    )
                )
            port._from_node = edge.input_node.id
            port._edge_metric_id = from_metric.id
            ports.append(port)
    _apply_input_port_multi_hints(node, ports, multi_candidates)
    _drop_ambiguous_port_identifiers(node.id, ports)
    return ports


def _drop_ambiguous_port_identifiers(node_id: str, ports: list[InputPortDef]) -> None:
    """
    Clear identifiers that would collide within the node.

    Derived names are not guaranteed unique: two datasets can expose the same
    column, and two edges can come from the same source node. An unnamed port
    is better than a mangled or wrongly-shared name, and a name can always be
    assigned in the editor afterwards.
    """
    counts = Counter(port.identifier for port in ports if port.identifier is not None)
    duplicates = {identifier for identifier, count in counts.items() if count > 1}
    if not duplicates:
        return
    logger.debug('Node {}: dropping ambiguous input port identifiers {}', node_id, sorted(duplicates))
    for port in ports:
        if port.identifier in duplicates:
            port.identifier = None


def _export_output_ports(node: Node) -> list[OutputPortDef]:
    """Build OutputPortDefs from a node's runtime output metrics."""
    # Check whether the node class defines output_metrics at the class level.
    # If so, ports derived from those are non-editable.
    class_metric_ids: set[str] = set()
    class_metrics = getattr(type(node), 'output_metrics', None)
    if isinstance(class_metrics, dict):
        class_metric_ids = set(class_metrics.keys())

    role_by_metric_id = {declaration.identifier: declaration.role for declaration in type(node).output_port_declarations}
    ports: list[OutputPortDef] = []
    for metric_id, metric in node.output_metrics.items():
        assert metric.unit is not None
        port = OutputPortDef(
            id=uuid_from_identifiers(node.context.instance, [node.id, metric_id]),
            identifier=identifier_or_none(metric_id),
            role=role_by_metric_id.get(metric_id),
            label=_to_ts(metric.label),
            unit=metric.unit,
            quantity=metric.quantity or None,
            column_id=metric.column_id,
            is_editable=metric_id not in class_metric_ids,
        )
        port._metric_id = metric_id
        ports.append(port)
    return ports


def _input_dataset_def_from_instance(ds: DatasetWithFilters) -> InputDatasetDef:
    """
    Describe a loaded dataset binding as a definition, for storing in the DB.

    The runtime already holds the pipeline — converted from the YAML flat fields
    when the instance loaded — so it is passed straight through rather than
    reconstructed field by field.
    """
    return InputDatasetDef(
        id=ds.id,
        tags=ds.tags or [],
        input_dataset=ds.input_dataset if isinstance(ds, DVCDataset) else None,
        column=ds.column,
        transformations=list(ds.transformations),
    )


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

    # input_dataset_processors: check if node uses interpolation. A class that interpolates
    # by default does not report a processor, because none was authored — the parse path
    # derives the same flag from the class, and the two representations have to agree.
    processors: list[str] = []
    if not type(node).interpolates_input_datasets_by_default:
        for ds in node.input_dataset_instances:
            if any(op.kind == 'interpolate' for op in ds.transformations):
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
    """Export authored node-local parameters (including reference params)."""
    return [param.model_copy() for param in node.parameters.values() if not param.is_implicit]


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

    raise ValueError(f'No port found for edge {edge.input_node.id}:{edge.output_node.id} metric {metric_id}')


# ---------------------------------------------------------------------------
# Full sync: runtime → DB
# ---------------------------------------------------------------------------


def _collect_edge_rows(ic: InstanceConfig, ctx: Context, node_configs: dict[str, NodeConfig]) -> tuple[int, list[dict[str, Any]]]:
    """Resolve runtime edges into edge-branch row kwargs, in ctx (source-grouped) order."""
    edge_count = 0
    edge_rows: list[dict[str, Any]] = []
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
                        f'No input port found for node {to_nc.identifier} for edge from '
                        + f'{from_nc.identifier}, metric {from_metric_id}'
                    )
                edge_rows.append(
                    dict(
                        instance=ic,
                        source_node=from_nc,
                        source_port_id=from_port.id,
                        node=to_nc,
                        port_id=to_port.id,
                        transformations=edge.to_transforms(),
                        tags=list(edge.tags) if edge.tags else [],
                    )
                )
            edge_count += 1
    return edge_count, edge_rows


def _resolve_dataset_ports(
    ic: InstanceConfig,
    nc: NodeConfig,
    node: Node,
    idx: int,
    ds_instance: DatasetWithFilters,
    db_datasets: dict[str, DatasetModel],
    metrics_by_schema_and_name: dict[tuple[int, str], DatasetMetric],
) -> list[dict[str, Any]]:
    """Resolve one input dataset into ``DatasetPort`` row kwargs (rows are built by the caller)."""
    from nodes.datasets import DBDataset, SerializedDBDataset

    # Resolve the Dataset model object depending on the dataset type.
    if isinstance(ds_instance, DBDataset):
        dataset_obj = ds_instance.db_dataset_obj
        assert dataset_obj is not None
    elif isinstance(ds_instance, SerializedDBDataset):
        assert ds_instance.payload_ref is not None
        dataset_obj = DatasetModel.objects.select_related('schema').get(pk=ds_instance.payload_ref.dataset_pk)
    elif isinstance(ds_instance, DVCDataset):
        dataset_obj = db_datasets.get(ds_instance.id)
    else:
        raise TypeError(f'Unknown dataset type: {type(ds_instance)}')

    if dataset_obj is None:
        raise ValueError(f'No dataset object for {ds_instance.id} on node {node.id}')

    if dataset_obj.schema is None:
        raise ValueError(f'Cannot create dataset port: schema={dataset_obj.schema} for {ds_instance.id} on node {node.id}')

    rows: list[dict[str, Any]] = []
    spec = DatasetPortSpec.from_input_dataset(_input_dataset_def_from_instance(ds_instance))
    pairs = _binding_pairs_for_dataset(node, ds_instance, dataset_obj, metrics_by_schema_and_name)
    for port_column, metric_key in pairs:
        metric = metrics_by_schema_and_name.get((dataset_obj.schema.pk, metric_key))
        if metric is None:
            if ds_instance.column is not None:
                raise ValueError(f'No metric {metric_key} in dataset {ds_instance.id} for node {node.id}')
            logger.debug(
                'No metric %s in dataset %s for node %s; skipping dataset-port binding', metric_key, ds_instance.id, node.id
            )
            continue

        rows.append(
            dict(
                instance=ic,
                node=nc,
                port_id=_dataset_port_id(node, idx, port_column),
                dataset=dataset_obj,
                metric=metric,
                transformations=list(spec.transformations),
                tags=list(spec.tags),
                dataset_spec=spec,
                dataset_index=idx,
            )
        )
    return rows


def _get_db_datasets(ic: InstanceConfig) -> dict[str, DatasetModel]:
    """Build a lookup of dataset identifier -> DB Dataset for an instance."""
    return {
        ds.identifier: ds
        for ds in DatasetModel.objects.get_queryset().for_instance_config(ic).select_related('schema')
        if ds.identifier
    }


def _collect_dataset_schema_pks(ctx: Context, db_datasets: dict[str, DatasetModel]) -> set[int]:
    """Collect schema PKs from both placeholder and DB-backed datasets."""
    from nodes.datasets import DBDataset

    pks: set[int] = set()
    for ds in db_datasets.values():
        assert ds.schema is not None
        pks.add(ds.schema.pk)
    for node in ctx.nodes.values():
        for ds_instance in node.input_dataset_instances:
            if isinstance(ds_instance, DBDataset):
                db_ds = ds_instance.db_dataset_obj
                if db_ds is not None and db_ds.schema is not None:
                    pks.add(db_ds.schema.pk)
                continue
            db_dataset_obj = db_datasets.get(ds_instance.id)
            if db_dataset_obj is not None and db_dataset_obj.schema is not None:
                pks.add(db_dataset_obj.schema.pk)
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


def _update_bindings(ic: InstanceConfig, ctx: Context, node_configs: dict[str, NodeConfig]) -> tuple[int, int]:
    """
    Reconcile ``NodeInputPortBinding`` rows from the runtime graph.

    Returns (edge count, dataset-binding row count). Row UUIDs are the durable
    binding identity and are carried over by structural matching.
    """
    from nodes.datasets import FixedDataset
    from nodes.input_bindings import reconcile_input_bindings
    from nodes.instance_serialization import (
        DatasetPortSnapshot,
        EdgeSnapshot,
        dataset_port_match_keys,
        edge_match_keys,
        existing_dataset_port_identities,
        existing_edge_identities,
        match_preserved_uuids,
        ordered_binding_snapshots,
    )
    from nodes.models import NodeInputPortBinding

    existing_edges = existing_edge_identities(ic)
    existing_ports = existing_dataset_port_identities(ic)

    edge_count, edge_rows = _collect_edge_rows(ic, ctx, node_configs)

    db_datasets = _get_db_datasets(ic)
    all_schema_pks = _collect_dataset_schema_pks(ctx, db_datasets)

    # Build lookup: (schema_pk, metric_name) -> DatasetMetric
    metrics_by_schema_and_name: dict[tuple[int, str], DatasetMetric] = {}
    for metric in DatasetMetric.objects.filter(schema__pk__in=all_schema_pks):
        metrics_by_schema_and_name[(metric.schema.pk, _dataset_metric_binding_key(metric))] = metric

    port_rows: list[dict[str, Any]] = []
    for node in ctx.nodes.values():
        nc = node_configs.get(node.id)
        if nc is None:
            continue

        for idx, ds_instance in enumerate(node.input_dataset_instances):
            if isinstance(ds_instance, FixedDataset):
                continue
            if not isinstance(ds_instance, DatasetWithFilters):
                continue
            port_rows.extend(_resolve_dataset_ports(ic, nc, node, idx, ds_instance, db_datasets, metrics_by_schema_and_name))

    edge_uuids = match_preserved_uuids(
        existing_edges,
        [edge_match_keys(row['source_node'].uuid, row['source_port_id'], row['node'].uuid, row['port_id']) for row in edge_rows],
    )
    port_uuids = match_preserved_uuids(
        existing_ports,
        [
            dataset_port_match_keys(row['node'].uuid, row['dataset'].pk, row['dataset_index'], row['metric'].pk)
            for row in port_rows
        ],
    )

    # Minimal snapshot forms carrying only what the ordering authority reads.
    edge_snaps = [
        EdgeSnapshot(
            from_node=row['source_node'].uuid,
            to_node=row['node'].uuid,
            from_port=row['source_port_id'],
            to_port=row['port_id'],
        )
        for row in edge_rows
    ]
    port_snaps = [
        DatasetPortSnapshot(
            node=row['node'].uuid,
            dataset=row['dataset'].identifier or str(row['dataset'].uuid),
            port_id=row['port_id'],
            metric=row['metric'].name or str(row['metric'].uuid),
            dataset_index=row['dataset_index'],
        )
        for row in port_rows
    ]
    row_by_snap = {id(snap): (row, matched) for snap, row, matched in zip(edge_snaps, edge_rows, edge_uuids, strict=True)}
    row_by_snap.update((id(snap), (row, matched)) for snap, row, matched in zip(port_snaps, port_rows, port_uuids, strict=True))

    desired: list[NodeInputPortBinding] = []
    for item, position in ordered_binding_snapshots(edge_snaps, port_snaps):
        row, matched_uuid = row_by_snap[id(item)]
        identity_kwargs = {'uuid': matched_uuid} if matched_uuid is not None else {}
        desired.append(NodeInputPortBinding(**row, position=position, **identity_kwargs))
    reconcile_input_bindings(ic, desired)
    return edge_count, len(port_rows)


def _promote_dataset_forecast_defaults(ic: InstanceConfig) -> int:
    """
    Promote binding-level forecast years to dataset defaults when unambiguous.

    YAML allows ``forecast_from`` per input-dataset binding. In the DB editor we
    want the common case to be dataset-scoped, with the binding's
    ``dataset_spec`` kept as an override. If all non-null binding years for an
    instance dataset agree, store that year on ``Dataset.spec.forecast_from``
    and clear matching binding overrides so those bindings inherit the dataset
    default.
    """
    from collections import defaultdict

    from nodes.dataset_materialization import refresh_dataset_materialization
    from nodes.models import DatasetMaterialization, NodeInputPortBinding

    ports_by_dataset: dict[int, list[NodeInputPortBinding]] = defaultdict(list)
    ports = (
        NodeInputPortBinding.objects.filter(instance=ic, dataset__isnull=False).select_related('dataset').order_by('dataset_id')
    )
    for port in ports:
        assert port.dataset_id is not None
        ports_by_dataset[port.dataset_id].append(port)

    materializations = {
        materialization.dataset_id: materialization
        for materialization in DatasetMaterialization.objects.select_for_update().filter(
            dataset_id__in=ports_by_dataset,
        )
    }

    promoted = 0
    for dataset_ports in ports_by_dataset.values():
        dataset = dataset_ports[0].dataset
        assert dataset is not None
        # External placeholders (no real DB dataset content) are loaded via plain DVCDataset at
        # runtime, which has no fallback to Dataset.spec.forecast_from (only DBDataset.from_def
        # does). Promoting for these would clear the binding-level value with nothing left to
        # read it back, silently dropping forecast_from and breaking Forecast-column synthesis.
        if dataset.is_external_placeholder:
            continue
        years = {port.dataset_spec.forecast_from for port in dataset_ports if port.dataset_spec.forecast_from is not None}
        if len(years) == 1:
            year = years.pop()
            spec = dict(dataset.spec or {})
            if spec.get('forecast_from') != year:
                spec['forecast_from'] = year
                dataset.spec = spec
                dataset.save(update_fields=['spec'])
                promoted += 1

            changed_ports: list[NodeInputPortBinding] = []
            for port in dataset_ports:
                if port.dataset_spec.forecast_from == year:
                    port.dataset_spec = port.dataset_spec.without_forecast_from()
                    changed_ports.append(port)
            if changed_ports:
                NodeInputPortBinding.objects.bulk_update(changed_ports, ['dataset_spec'])

        forecast_from = (dataset.spec or {}).get('forecast_from')
        materialization = materializations.get(dataset.pk)
        if forecast_from is not None and (materialization is None or materialization.forecast_from != forecast_from):
            refresh_dataset_materialization(dataset)

    return promoted


def sync_instance_to_db(
    instance_id: str,
    yaml_path: str | Path | None = None,
    *,
    promote_forecast_defaults: bool = True,
) -> None:
    """
    Load an instance from YAML and sync its spec to the DB.

    If yaml_path is not given, tries configs/{instance_id}.yaml.
    ``promote_forecast_defaults=False`` keeps binding-level forecast years
    on the DatasetPort specs (used by the parse oracle, which compares
    against pre-promotion state).
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
        instance_spec.features.use_datasets_from_db = True
        ic, _created = InstanceConfig.objects.get_or_create(identifier=instance.id)
        _apply_instance_metadata_columns(ic, instance)
        ic.spec = instance_spec
        ic.config_source = 'database'
        ic.invalidate_cache(save=False)
        ic.save()

        ic.sync_dimensions(update_existing=True, instance=instance)

        # Update or create node configs
        node_qs = ic.nodes.all().defer('spec')
        existing_ncs = {nc.identifier: nc for nc in node_qs}
        node_configs: dict[str, NodeConfig] = {}
        for node_id, node in ctx.nodes.items():
            nc = existing_ncs.get(node_id)
            if nc is None:
                nc = NodeConfig(instance=ic, identifier=node_id)
            nc.update_from_node(node, overwrite=node_id not in existing_ncs, update_relations=False)
            spec = export_node_spec(node)
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
            delete_nodes = stale_nodes.filter(pages__isnull=True, created_by__isnull=True)
            for stale_node in delete_nodes:
                logger.info(f'Stale node {stale_node.identifier} was automatically created, removing')
                stale_node.delete()
            # NodeConfig.objects.filter(instance=ic, identifier__in=stale_ids, pages__isnull=True).defer('spec').delete()

        created_placeholder_ids = sync_instance_dataset_placeholders(ic, ctx)

        edge_count, dataset_port_count = _update_bindings(ic, ctx, node_configs)
        promoted_forecast_defaults = _promote_dataset_forecast_defaults(ic) if promote_forecast_defaults else 0

    logger.info(
        (
            'Synced {id}: {nodes} nodes, {edges} edges, {placeholders} dataset placeholders created, '
            '{ports} dataset ports, {forecast_defaults} dataset forecast defaults promoted'
        ),
        id=instance.id,
        nodes=len(node_configs),
        edges=edge_count,
        placeholders=len(created_placeholder_ids),
        ports=dataset_port_count,
        forecast_defaults=promoted_forecast_defaults,
    )
