from typing import TYPE_CHECKING, Annotated

import strawberry as sb

from paths.graphql_helpers import pass_context
from paths.graphql_types import UnitType

from nodes.constants import FORECAST_COLUMN, IMPACT_COLUMN, IMPACT_GROUP, YEAR_COLUMN
from nodes.metric import DimensionalMetric, Metric, YearlyValue

from .metric import DimensionalMetricType

if TYPE_CHECKING:
    from common import polars as ppl
    from nodes.actions.action import ActionNode, ImpactOverview
    from nodes.context import Context
    from nodes.goals import NodeGoalsEntry
    from nodes.graphql.types.node import ActionNodeType, NodeType
    from nodes.node import Node
    from nodes.units import Unit


def get_impact_metric(
    source_node: ActionNode,
    target_node: Node,
    goal: NodeGoalsEntry | None = None,
) -> Metric | None:
    import polars as pl

    df: ppl.PathsDataFrame = source_node.compute_impact(target_node)
    if goal is not None:
        df = goal.filter_df(df)

    df = df.filter(pl.col(IMPACT_COLUMN).eq(IMPACT_GROUP)).drop(IMPACT_COLUMN)
    if df.dim_ids:
        df = df.paths.sum_over_dims()

    try:
        m = target_node.get_default_output_metric()
    except Exception:
        return None

    df = df.select([*df.primary_keys, FORECAST_COLUMN, m.column_id])
    active_normalization = target_node.context.active_normalization
    if active_normalization and active_normalization.get_normalized_unit(m) is not None:
        _, df = active_normalization.normalize_output(m, df)

    return Metric(
        id='%s-%s-impact' % (source_node.id, target_node.id),
        name='Impact',
        df=df,
        unit=df.get_unit(m.column_id),
    )


@sb.type
class ActionImpact:
    action: 'ActionNode' = sb.field(graphql_type=Annotated['ActionNodeType', sb.lazy('nodes.schema')])
    cost_values: list[YearlyValue] | None = sb.field(deprecation_reason='Use costDim instead.')
    impact_values: list[YearlyValue | None] | None = sb.field(deprecation_reason='Use effectDim instead.')
    cost_dim: DimensionalMetric | None = sb.field(graphql_type=DimensionalMetricType | None)
    effect_dim: DimensionalMetric = sb.field(graphql_type=DimensionalMetricType)
    unit_adjustment_multiplier: float | None


@sb.type
class WedgeEntryType:
    id: sb.ID
    label: str
    is_scenario: bool
    metric: DimensionalMetric = sb.field(graphql_type=DimensionalMetricType)


@sb.type
class ImpactOverviewType:
    cost_node: Annotated['NodeType', sb.lazy('nodes.schema')] | None
    effect_node: Annotated['NodeType', sb.lazy('nodes.schema')]

    @sb.field
    @staticmethod
    def id(root: 'ImpactOverview') -> sb.ID:
        cost_id = root.cost_node.id if root.cost_node else 'None'
        return sb.ID('%s:%s' % (cost_id, root.effect_node.id))

    @sb.field
    @staticmethod
    def graph_type(root: 'ImpactOverview') -> str | None:
        if root.spec.graph_type in ['benefit_cost_ratio', 'return_on_investment_gross']:
            return 'return_on_investment'
        return root.spec.graph_type

    @sb.field(graphql_type=UnitType)
    @staticmethod
    def indicator_unit(root: 'ImpactOverview') -> 'Unit':
        return root.spec.indicator_unit

    @sb.field
    @staticmethod
    def indicator_cutpoint(root: 'ImpactOverview') -> float | None:
        return root.spec.indicator_cutpoint

    @sb.field
    @staticmethod
    def cost_cutpoint(root: 'ImpactOverview') -> float | None:
        return root.spec.cost_cutpoint

    @sb.field
    @staticmethod
    def plot_limit_for_indicator(root: 'ImpactOverview') -> float | None:
        return root.spec.plot_limit_for_indicator

    @sb.field
    @staticmethod
    def label(root: 'ImpactOverview') -> str:
        return str(root.spec.label or '')

    @sb.field
    @staticmethod
    def cost_label(root: 'ImpactOverview') -> str | None:
        return str(root.spec.cost_label) if root.spec.cost_label is not None else None

    @sb.field
    @staticmethod
    def effect_label(root: 'ImpactOverview') -> str | None:
        return str(root.spec.effect_label) if root.spec.effect_label is not None else None

    @sb.field
    @staticmethod
    def indicator_label(root: 'ImpactOverview') -> str | None:
        return str(root.spec.indicator_label) if root.spec.indicator_label is not None else None

    @sb.field
    @staticmethod
    def cost_category_label(root: 'ImpactOverview') -> str | None:
        return str(root.spec.cost_category_label) if root.spec.cost_category_label is not None else None

    @sb.field
    @staticmethod
    def effect_category_label(root: 'ImpactOverview') -> str | None:
        return str(root.spec.effect_category_label) if root.spec.effect_category_label is not None else None

    @sb.field
    @staticmethod
    def description(root: 'ImpactOverview') -> str | None:
        return str(root.spec.description) if root.spec.description is not None else None

    @sb.field
    @pass_context
    @staticmethod
    def actions(root: 'ImpactOverview', context: 'Context') -> list[ActionImpact]:
        all_aes = root.calculate(context)
        out: list[ActionImpact] = []
        for ae in all_aes:
            years = ae.df[YEAR_COLUMN]
            if 'Cost' in ae.df.columns:
                cost_values = [
                    YearlyValue(year=year, value=float(val)) for year, val in zip(years, list(ae.df['Cost']), strict=False)
                ]
            else:
                cost_values = None
            effect_dim = DimensionalMetric.from_action_impact(ae, root, 'Effect')
            if effect_dim is None:
                raise ValueError('Effect dimension is None')
            out.append(
                ActionImpact(
                    action=ae.action,
                    cost_values=cost_values,
                    impact_values=[
                        YearlyValue(year=year, value=float(val)) for year, val in zip(years, list(ae.df['Effect']), strict=False)
                    ],
                    cost_dim=DimensionalMetric.from_action_impact(ae, root, 'Cost'),
                    effect_dim=effect_dim,
                    unit_adjustment_multiplier=ae.unit_adjustment_multiplier,
                )
            )
        return out

    @sb.field(graphql_type=UnitType | None)
    @staticmethod
    def cost_unit(root: 'ImpactOverview') -> 'Unit':
        return root.spec.cost_unit or root.spec.indicator_unit

    @sb.field(graphql_type=UnitType | None)
    @staticmethod
    def effect_unit(root: 'ImpactOverview') -> 'Unit':
        return root.spec.effect_unit or root.spec.indicator_unit

    @sb.field
    @staticmethod
    def outcome_dimension(root: 'ImpactOverview') -> str | None:
        return root.spec.outcome_dimension_id

    @sb.field
    @staticmethod
    def stakeholder_dimension(root: 'ImpactOverview') -> str | None:
        return root.spec.stakeholder_dimension_id

    @sb.field
    @pass_context
    @staticmethod
    def wedge(root: 'ImpactOverview', context: 'Context') -> 'list[WedgeEntryType] | None':
        from nodes.defs.action_def import ImpactGraphType

        if root.spec.graph_type != ImpactGraphType.WEDGE_DIAGRAM:
            return None
        entries = root.compute_wedge(context)
        return [
            WedgeEntryType(
                id=sb.ID(e.id),
                label=e.label,
                is_scenario=e.is_scenario,
                metric=e.metric,
            )
            for e in entries
        ]
