"""
Export a runtime Instance to InstanceSpec / NodeSpec for DB storage.

Given a fully loaded Instance (from YAML InstanceLoader), introspect
the live object graph and produce the Pydantic spec models that can be
stored on InstanceConfig.spec and NodeConfig.spec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from kausal_common.i18n.pydantic import TranslatedString

from nodes.actions.action import ActionNode
from nodes.constants import DecisionLevel
from nodes.defs import (
    ActionConfig,
    ActionGroupDef,
    DatasetRepoDef,
    InstanceSpec,
    NodeSpec,
    OutputMetricDef,
    ScenarioDef,
    ScenarioParameterOverrideDef,
    SimpleConfig,
    YearsDef,
)

if TYPE_CHECKING:
    from pathlib import Path

    from kausal_common.i18n.pydantic import I18nString

    from nodes.context import Context
    from nodes.defs import (
        FormulaConfig,
    )
    from nodes.defs.node_defs import NodeSpecExtra
    from nodes.edges import Edge, EdgeDimension
    from nodes.instance import Instance
    from nodes.node import Node, NodeMetric
    from nodes.scenario import Scenario, ScenarioKind
    from params import Parameter


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

    years = YearsDef(
        reference=instance.reference_year,
        min_historical=instance.minimum_historical_year,
        max_historical=instance.maximum_historical_year,
        target=ctx.target_year,
        model_end=ctx.model_end_year,
    )

    dataset_repo = _export_dataset_repo(ctx)
    params = _export_global_params(ctx)
    action_groups = _export_action_groups(instance)
    scenarios = _export_scenarios(ctx)

    return InstanceSpec(
        years=years,
        dataset_repo=dataset_repo,
        dimensions=_export_dimensions(ctx),
        features=instance.features.model_dump(),
        params=params,
        action_groups=action_groups,
        scenarios=scenarios,
    )


def export_node_spec(node: Node) -> NodeSpec:
    """Build a NodeSpec from a live Node."""
    type_config = _export_type_config(node)
    output_metrics = _export_output_metrics(node)
    params = _export_node_params(node)

    kls = type(node)
    node_class = f'{kls.__module__}.{kls.__qualname__}'

    # Capture dataset configs and dimension IDs.
    # Skip internal dimensions — they're created dynamically by node classes at runtime.
    input_datasets = _export_input_datasets(node)
    extra = _export_node_extra(node)
    input_dim_ids = [d for d, dim in node.input_dimensions.items() if not dim.is_internal] if node.input_dimensions else []
    output_dim_ids = [d for d, dim in node.output_dimensions.items() if not dim.is_internal] if node.output_dimensions else []

    return NodeSpec(
        node_class=node_class,
        type_config=type_config,
        output_metrics=output_metrics,
        input_datasets=input_datasets,
        input_dimensions=input_dim_ids,
        output_dimensions=output_dim_ids,
        params=params,
        is_outcome=node.is_outcome,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Instance-level helpers
# ---------------------------------------------------------------------------


def _export_dimensions(ctx: Context) -> list[dict[str, Any]]:
    return [dim.model_dump(exclude_none=True) for dim in ctx.dimensions.values()]


def _export_dataset_repo(ctx: Context) -> DatasetRepoDef:
    repo = ctx.dataset_repo
    if repo is None:
        return DatasetRepoDef()
    return DatasetRepoDef(
        url=repo.repo_url,
        commit=repo.target_commit_id,
        dvc_remote=repo.dvc_remote,
    )


def _export_global_params(ctx: Context) -> list[Parameter]:
    return [param.model_copy() for param in ctx.global_parameters.values()]


def _export_action_groups(instance: Instance) -> list[ActionGroupDef]:
    groups: list[ActionGroupDef] = []
    for idx, ag in enumerate(instance.action_groups):
        groups.append(
            ActionGroupDef(
                id=ag.id,
                name=_to_ts(ag.name),
                color=ag.color or '',
                order=idx,
            )
        )
    return groups


def _export_scenarios(ctx: Context) -> list[ScenarioDef]:
    from nodes.scenario import CustomScenario

    scenarios: list[ScenarioDef] = []
    for scenario in ctx.scenarios.values():
        if isinstance(scenario, CustomScenario):
            continue
        scenarios.append(_export_scenario(scenario))
    return scenarios


def _export_scenario(scenario: Scenario) -> ScenarioDef:
    overrides: list[ScenarioParameterOverrideDef] = []
    for param, value in scenario.get_param_values():
        node_id = None
        if param._node is not None:
            node_id = param._node.id
        overrides.append(
            ScenarioParameterOverrideDef(
                parameter_id=param.global_id,
                value=value,
                node_id=node_id,
            )
        )

    kind: ScenarioKind | None = None
    if scenario.kind is not None:
        kind = scenario.kind

    return ScenarioDef(
        id=scenario.id,
        name=_to_ts(scenario.name),
        kind=kind,
        all_actions_enabled=scenario.all_actions_enabled,
        params=overrides,
    )


# ---------------------------------------------------------------------------
# Node-level helpers
# ---------------------------------------------------------------------------


def _export_type_config(node: Node) -> FormulaConfig | ActionConfig | SimpleConfig:
    if isinstance(node, ActionNode):
        decision_level: str | None = None
        if node.decision_level is not None:
            if isinstance(node.decision_level, DecisionLevel):
                decision_level = node.decision_level.name.lower()
            else:
                decision_level = str(node.decision_level)
        return ActionConfig(decision_level=decision_level)

    assert not hasattr(node, 'formula')

    return SimpleConfig()


def _export_output_metrics(node: Node) -> list[OutputMetricDef]:
    metrics: list[OutputMetricDef] = []
    for metric_id, metric in node.output_metrics.items():
        metrics.append(_export_metric(metric_id, metric))
    return metrics


def _export_metric(metric_id: str, metric: NodeMetric) -> OutputMetricDef:
    assert metric.unit is not None
    unit = str(metric.unit)
    if not unit and metric.unit.dimensionless:
        unit = 'dimensionless'
    return OutputMetricDef(
        id=metric_id,
        label=_to_ts(metric.label),
        unit=unit,
        quantity=metric.quantity or '',
    )


def _export_input_datasets(node: Node) -> list[dict[str, Any]]:
    """
    Export node input datasets as config dicts.

    FixedDatasets are skipped — they are created from historical_values/forecast_values
    at the node config level, not from input_datasets.
    """
    from nodes.datasets import DatasetWithFilters

    result: list[dict[str, Any]] = []
    for ds in node.input_dataset_instances:
        if not isinstance(ds, DatasetWithFilters):
            continue
        entry: dict[str, Any] = {'id': ds.id}
        if ds.tags:
            entry['tags'] = ds.tags
        if hasattr(ds, 'input_dataset') and ds.input_dataset:
            entry['input_dataset'] = ds.input_dataset
        if ds.column:
            entry['column'] = ds.column
        if ds.forecast_from is not None:
            entry['forecast_from'] = ds.forecast_from
        if ds.filters:
            entry['filters'] = ds.filters
        if ds.dropna is not None:
            entry['dropna'] = ds.dropna
        if ds.min_year is not None:
            entry['min_year'] = ds.min_year
        if ds.max_year is not None:
            entry['max_year'] = ds.max_year
        result.append(entry)
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
    if edge.metrics:
        result['metrics'] = list(edge.metrics)
    return result


# ---------------------------------------------------------------------------
# Full sync: runtime → DB
# ---------------------------------------------------------------------------


def sync_instance_to_db(instance_id: str, yaml_path: str | Path | None = None) -> None:  # noqa: C901
    """
    Load an instance from YAML and sync its spec to the DB.

    If yaml_path is not given, tries configs/{instance_id}.yaml.
    """
    from pathlib import Path as _Path

    from django.db import transaction

    from nodes.instance_loader import InstanceLoader
    from nodes.models import InstanceConfig, NodeConfig, NodeEdge

    if yaml_path is None:
        yaml_path = _Path(f'configs/{instance_id}.yaml').resolve()
    else:
        yaml_path = _Path(yaml_path).resolve()

    if not yaml_path.exists():
        # Try globbing
        for p in _Path('configs').glob(f'{instance_id}*.yaml'):
            yaml_path = p.resolve()
            break

    loader = InstanceLoader.from_yaml(yaml_path)
    instance = loader.instance
    ctx = loader.context

    instance_spec = export_instance_spec(instance)

    with transaction.atomic():
        ic, _created = InstanceConfig.objects.get_or_create(identifier=instance.id)
        ic.spec = instance_spec
        ic.config_source = 'database'
        ic.save()

        # Update or create node configs
        existing_ncs = {nc.identifier: nc for nc in NodeConfig.objects.filter(instance=ic)}
        node_configs: dict[str, NodeConfig] = {}
        for node_id, node in ctx.nodes.items():
            nc = existing_ncs.get(node_id)
            if nc is None:
                nc = NodeConfig(instance=ic, identifier=node_id)
            nc.name = getattr(node, 'name', node_id)
            nc.spec = export_node_spec(node)
            nc.save()
            node_configs[node_id] = nc

        # Remove stale node configs
        stale_ids = set(existing_ncs.keys()) - set(node_configs.keys())
        if stale_ids:
            NodeConfig.objects.filter(instance=ic, identifier__in=stale_ids).delete()

        # Recreate edges
        NodeEdge.objects.filter(instance=ic).delete()
        edge_count = 0
        for node in ctx.nodes.values():
            for edge_obj in node.edges:
                if edge_obj.input_node.id != node.id:
                    continue  # only process outgoing edges from this node
                from_nc = node_configs.get(edge_obj.input_node.id)
                to_nc = node_configs.get(edge_obj.output_node.id)
                if from_nc and to_nc:
                    NodeEdge.objects.create(
                        instance=ic,
                        from_node=from_nc,
                        to_node=to_nc,
                        from_port='output',
                        to_port=f'from_{edge_obj.input_node.id}',
                        transformations=edge_to_transforms(edge_obj),
                        tags=list(edge_obj.tags) if edge_obj.tags else [],
                    )
                    edge_count += 1

    logger.info(
        'Synced {id}: {nodes} nodes, {edges} edges',
        id=instance.id,
        nodes=len(node_configs),
        edges=edge_count,
    )
