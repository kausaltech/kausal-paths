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
    FormulaConfig,
    InstanceSpec,
    NodeSpec,
    OutputMetricDef,
    ScenarioDef,
    ScenarioParameterOverrideDef,
    SimpleConfig,
    YearsDef,
)

if TYPE_CHECKING:
    from kausal_common.i18n.pydantic import I18nString

    from nodes.context import Context
    from nodes.defs.node_defs import NodeSpecExtra
    from nodes.instance import Instance
    from nodes.node import Node, NodeMetric
    from nodes.scenario import Scenario, ScenarioKind


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
    features = _export_features(instance)
    params = _export_global_params(ctx)
    action_groups = _export_action_groups(instance)
    scenarios = _export_scenarios(ctx)

    return InstanceSpec(
        years=years,
        dataset_repo=dataset_repo,
        dimensions=_export_dimensions(ctx),
        features=features,
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
    dims: list[dict[str, Any]] = []
    for dim in ctx.dimensions.values():
        dims.append(dim.model_dump(exclude_none=True, exclude={'mtime_hash'}))
    return dims


def _export_dataset_repo(ctx: Context) -> DatasetRepoDef:
    repo = ctx.dataset_repo
    if repo is None:
        return DatasetRepoDef()
    return DatasetRepoDef(
        url=getattr(repo, 'repo_url', '') or '',
        commit=getattr(repo, 'commit', None),
        dvc_remote=getattr(repo, 'dvc_remote', None),
    )


def _export_features(instance: Instance) -> dict[str, object]:
    features = instance.features
    if features is None:
        return {}
    # InstanceFeatures is a dataclass-like object; dump its fields.
    if hasattr(features, '__dict__'):
        return {k: v for k, v in features.__dict__.items() if not k.startswith('_')}
    return {}


def _export_global_params(ctx: Context) -> list[Any]:
    params: list[Any] = []
    for param in ctx.global_parameters.values():
        try:
            params.append(param.model_copy())
        except Exception:
            logger.warning('Could not export global parameter {id}', id=param.local_id)
    return params


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

    # Check if the node has a formula attribute (FormulaNode, etc.)
    formula = getattr(node, 'formula', None)
    if formula is not None:
        return FormulaConfig(formula=str(formula))

    return SimpleConfig()


def _export_output_metrics(node: Node) -> list[OutputMetricDef]:
    metrics: list[OutputMetricDef] = []
    for metric_id, metric in node.output_metrics.items():
        metrics.append(_export_metric(metric_id, metric))
    return metrics


def _export_metric(metric_id: str, metric: NodeMetric) -> OutputMetricDef:
    unit = metric.unit
    unit_str = str(unit) if unit is not None else ''
    if not unit_str and unit is not None and unit.dimensionless:
        unit_str = 'dimensionless'
    return OutputMetricDef(
        id=metric_id,
        label=_to_ts(metric.label),
        unit=unit_str,
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


def _export_node_params(node: Node) -> list[Any]:
    """Export node-local parameters (including reference params)."""
    params: list[Any] = []
    for param in node.parameters.values():
        try:
            params.append(param.model_copy())
        except Exception:
            logger.warning('Could not export parameter {id} on node {node}', id=param.local_id, node=node.id)
    return params
