from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, cast

import graphene
from django.forms import ValidationError
from django.utils.translation import gettext_lazy as _
from wagtail import blocks
from wagtail.images.blocks import ImageChooserBlock

import polars as pl
from grapple.helpers import register_streamfield_block
from grapple.models import GraphQLField, GraphQLFloat, GraphQLImage, GraphQLStreamfield, GraphQLString
from wagtail_color_panel.blocks import NativeColorBlock

from nodes.blocks import NodeChooserBlock
from nodes.constants import IMPACT_COLUMN, IMPACT_GROUP, VALUE_COLUMN, YEAR_COLUMN
from nodes.metric import DimensionalMetric

if TYPE_CHECKING:
    from collections.abc import Iterable

    from paths.types import GQLInstanceInfo

    from nodes.actions.action import ActionNode
    from nodes.node import Node
    from nodes.scenario import Scenario
    from nodes.schema import (
        ActionImpactType,
        ActionNodeType,
        MetricDimensionCategoryType,
        MetricDimensionCategoryValue,
        MetricDimensionType,
        ScenarioActionImpacts,
        ScenarioType,
        ScenarioValue,
    )
    from nodes.units import Unit


class CardListCardBlock(blocks.StructBlock):
    title = blocks.CharBlock(required=True)
    short_description = blocks.CharBlock(required=False)

    graphql_fields = [
        GraphQLString('title'),
        GraphQLString('short_description'),
    ]


def build_block_type(cls: type, type_prefix: str, interfaces: tuple[graphene.Interface, ...], base_type=graphene.ObjectType):
    from grapple.actions import build_streamfield_type

    ret = build_streamfield_type(cls, type_prefix, interfaces, base_type)  # type: ignore
    del ret._meta.fields['id']  # type: ignore
    return ret


class GraphQLBlockField(GraphQLField):
    def __init__(self, field_name: str, block_cls: type[blocks.Block], *, required: bool | None = None, **kwargs):
        def build_type() -> type:
            return build_block_type(block_cls, '', interfaces=())

        super().__init__(field_name, build_type, required=required, **kwargs)  # type: ignore


@register_streamfield_block
class CardListBlock(blocks.StructBlock):
    title = blocks.CharBlock(required=False)
    cards = blocks.ListBlock(CardListCardBlock(), required=True)

    graphql_fields = [
        GraphQLString('title'),
        GraphQLBlockField('cards', CardListCardBlock, is_list=True, required=True),
    ]


class ProgressBarBlock(blocks.StructBlock):
    """Base class for progress bar blocks with common fields."""

    title = blocks.CharBlock()
    description = blocks.CharBlock(required=False)
    chart_label = blocks.CharBlock()
    color = NativeColorBlock(required=False)

    graphql_fields = [
        GraphQLString('title', required=True),
        GraphQLString('description', required=True),
        GraphQLString('chart_label', required=True),
        GraphQLString('color', required=True),
    ]


@register_streamfield_block
class GoalProgressBarBlock(ProgressBarBlock):
    pass


@register_streamfield_block
class ReferenceProgressBarBlock(ProgressBarBlock):
    pass


@register_streamfield_block
class CurrentProgressBarBlock(ProgressBarBlock):
    pass


@register_streamfield_block
class ScenarioProgressBarBlock(ProgressBarBlock):
    scenario_id = blocks.CharBlock(required=True)  # FIXME: choice block? But where to get the choices?

    graphql_fields = ProgressBarBlock.graphql_fields + [
        GraphQLString('scenario_id', required=True),
    ]

    # TODO Validate that the scenario identifier actually exists in the instance.
    # However, for this we'd need access to the node (from the parent block) here, which seems to be difficult.


@register_streamfield_block
class CategoryBreakdownBlock(blocks.StructBlock):
    title = blocks.CharBlock()
    # TODO: dimension_id should be a choice block. Need to implement some way of getting the choices.
    dimension_id = blocks.CharBlock(
        required=False,
        help_text=_("Break down the given dimension into its categories. If empty, break down into input nodes."),
    )

    # TODO Validate that the dimension id actually exists in the instance.
    # However, for this we'd need access to the node (from the parent block) here, which seems to be difficult.

    graphql_fields = [
        GraphQLString('title', required=True),
        GraphQLString('dimension_id', required=True),
    ]


@register_streamfield_block
class ActionImpactBlock(blocks.StructBlock):
    title = blocks.CharBlock()
    # TODO: scenario_id should be a choice block. Need to implement some way of getting the choices.
    scenario_id = blocks.CharBlock(required=True)

    graphql_fields = [
        GraphQLString('title', required=True),
        GraphQLString('scenario_id', required=True),
    ]


@register_streamfield_block
class CallToActionBlock(blocks.StructBlock):
    title = blocks.CharBlock(label=_('Title'))
    content = blocks.CharBlock(label=_('Content'), required=False)
    link_url = blocks.CharBlock(label=_('Link URL'))

    graphql_fields = [
        GraphQLString('title', required=True),
        GraphQLString('content', required=True),
        GraphQLString('link_url', required=True),
    ]


@register_streamfield_block
class DashboardCardBlock(blocks.StructBlock):
    title = blocks.CharBlock()
    description = blocks.CharBlock(required=False)
    image = ImageChooserBlock(required=False)
    node_config = NodeChooserBlock(required=True)
    goal_index = blocks.IntegerBlock(required=False, min_value=0)  # used in {Goal,Scenario}ProgressBarBlock
    visualizations = blocks.StreamBlock([
        ('goal_progress_bar', GoalProgressBarBlock()),
        ('reference_progress_bar', ReferenceProgressBarBlock()),
        ('current_progress_bar', CurrentProgressBarBlock()),
        ('scenario_progress_bar', ScenarioProgressBarBlock()),
        ('category_breakdown', CategoryBreakdownBlock(
            label=_("Category breakdown"), help_text=_("Break down the value into its components")
        )),
        ('action_impact', ActionImpactBlock(
            label=_("Action impact"), help_text=_("Visualize the impact of actions in a scenario")
        )),
    ])
    call_to_action = CallToActionBlock()

    graphql_fields = [
        GraphQLString('title', required=True),
        GraphQLString('description', required=True),
        GraphQLImage('image', required=False),
        GraphQLField('node', 'nodes.schema.NodeType', required=True),  # pyright: ignore
        GraphQLField('unit', 'paths.schema.UnitType', required=True),  # pyright: ignore
        GraphQLFloat('goal_value', required=False),
        GraphQLFloat('reference_year_value', required=False),
        GraphQLFloat('last_historical_year_value', required=False),
        GraphQLField('scenario_values', 'nodes.schema.ScenarioValue', is_list=True, required=True),  # pyright: ignore
        GraphQLField(
            'metric_dimension_category_values',
            'nodes.schema.MetricDimensionCategoryValue',  # pyright: ignore
            is_list=True,
            required=False,  # can be null if there is no historical data
        ),
        GraphQLField(
            'scenario_action_impacts',
            'nodes.schema.ScenarioActionImpacts',  # pyright: ignore
            is_list=True,
            required=True,
        ),
        GraphQLStreamfield('visualizations', required=True),
        GraphQLBlockField('call_to_action', CallToActionBlock, is_list=False, required=True),
    ]

    def clean(self, value):
        from nodes.models import NodeConfig

        cleaned_data = super().clean(value)
        errors = {}
        node_config = cleaned_data['node_config']
        assert isinstance(node_config, NodeConfig)
        node = node_config.get_node()

        if not node:
            # No good way of explaining this to the user...
            errors['node_config'] = ValidationError(_("This object does not correspond to a node."))
        elif not node.goals or not node.goals.root:
            errors['node_config'] = ValidationError(_("This node has no goals."))
        elif node.context.instance.reference_year is None:
            errors['node_config'] = ValidationError(_("This node's instance has no reference year."))

        goal_index = cleaned_data.get('goal_index')
        if node and node.goals and goal_index is not None and goal_index >= len(node.goals.root):
            errors['goal_index'] = ValidationError(_("This goal index is invalid."))

        if errors:
            raise blocks.StructBlockValidationError(errors)
        return cleaned_data

    def node(self, info: GQLInstanceInfo, values: dict) -> Node:
        from nodes.models import NodeConfig

        node_config = values['node_config']
        assert isinstance(node_config, NodeConfig)
        node = node_config.get_node()
        if not node:
            raise ValueError("Node config has no node.")  # hopefully prevented by validation
        return node

    def unit(self, info: GQLInstanceInfo, values: dict) -> Unit:
        node = self.node(info, values)
        dm = self._dimensional_metric(node)
        return dm.unit

    def goal_value(self, info: GQLInstanceInfo, values: dict) -> float | None:
        """Return the value for the chosen goal for the node's target year."""
        node = self.node(info, values)
        target_year = node.get_target_year()
        if target_year is None:
            raise ValueError("Node has no target year")
        goal_index = values.get('goal_index')
        return self._goal_for_year(node, target_year, goal_index)

    def reference_year_value(self, info: GQLInstanceInfo, values: dict) -> float | None:
        node = self.node(info, values)
        reference_year = info.context.instance.reference_year
        if reference_year is None:
            raise ValueError("Instance has no reference year")
        return self._value_for_year(node, reference_year)

    def last_historical_year_value(self, info: GQLInstanceInfo, values: dict) -> float | None:
        node = self.node(info, values)
        dm  = self._dimensional_metric(node)
        last_historical_year = self._last_historical_year(dm)
        if last_historical_year is None:
            return None
        return self._value_for_year(node, last_historical_year)

    def scenario_values(self, info: GQLInstanceInfo, values: dict) -> Iterable[ScenarioValue]:
        """Return the value for each scenario for the node's target year."""
        from nodes.schema import ScenarioValue
        node = self.node(info, values)
        target_year = node.get_target_year()
        if target_year is None:
            raise ValueError("Node has no target year")
        return [
            ScenarioValue(
                scenario=cast('ScenarioType', s),
                year=target_year,
                value=self._value_for_year(node, target_year, s),
            )
            for s in node.context.scenarios.values()
        ]

    def metric_dimension_category_values(
        self, info: GQLInstanceInfo, values: dict
    ) -> Iterable[MetricDimensionCategoryValue] | None:
        """
        Return the value for each dimension and each category for the node's target year.

        Returns None if there is no historical data.
        """
        from nodes.schema import MetricDimensionCategoryValue
        node = self.node(info, values)
        dm = self._dimensional_metric(node)
        year = self._last_historical_year(dm)
        if year is None:
            return None
        result = []
        df = dm.to_df()
        df = df.filter(pl.col(YEAR_COLUMN) == year)
        for dimension in dm.dimensions:
            for category in dimension.categories:
                cat_df = df.copy()
                cat_df = cat_df.filter(pl.col(dimension.original_id) == category.original_id)
                cat_df = cat_df.paths.sum_over_dims()  # may sum over categories from other dimensions
                if cat_df.is_empty():
                    continue  # Can this be? Handle this better?
                assert len(cat_df) == 1
                value = cat_df.item(0, VALUE_COLUMN)
                result.append(MetricDimensionCategoryValue(
                    dimension=cast('MetricDimensionType', dimension),  # pyright: ignore[reportArgumentType]
                    category=cast('MetricDimensionCategoryType', category),  # pyright: ignore[reportArgumentType]
                    value=value,
                    year=year,
                ))
        return result

    def scenario_action_impacts(
        self, info: GQLInstanceInfo, values: dict
    ) -> Iterable[ScenarioActionImpacts]:
        """Return the impact of each action in the node's target year for each scenario."""
        from nodes.schema import ScenarioActionImpacts
        node = self.node(info, values)
        target_year = node.get_target_year()
        if target_year is None:
            raise ValueError("Node has no target year")
        return [
            ScenarioActionImpacts(
                scenario=cast('ScenarioType', scenario),
                impacts=[self._impact_for_action(action, node, target_year) for action in node.context.get_actions()],
            )
            for scenario in node.context.scenarios.values()
        ]

    def _dimensional_metric(self, node: Node, scenario: Scenario | None = None) -> DimensionalMetric:
        context = scenario.override() if scenario else nullcontext()
        with context:
            dm = DimensionalMetric.from_node(node)
        if not dm:
            raise ValueError("Could not obtain dimensional metric from node")
        return dm

    def _goal_for_year(self, node: Node, year: int, goal_index: int | None) -> float | None:
        if goal_index is None:
            goal_index = 0
        dm = self._dimensional_metric(node)
        if goal_index < 0 or goal_index >= len(dm.goals):
            raise ValueError("Goal index is invalid")
        goal = dm.goals[goal_index]
        year_values = (v for v in goal.values if v.year == year)
        try:
            return next(iter(year_values)).value
        except StopIteration:
            return None

    def _value_for_year(self, node: Node, year: int, scenario: Scenario | None = None) -> float | None:
        dm = self._dimensional_metric(node, scenario)
        df = dm.to_df()
        df = df.filter(pl.col(YEAR_COLUMN) == year)
        df = df.paths.sum_over_dims()
        if df.is_empty():
            return None
        assert len(df) == 1
        return df.item(0, VALUE_COLUMN)

    def _last_historical_year(self, dimensional_metric: DimensionalMetric) -> int | None:
        dm = dimensional_metric
        if dm.forecast_from is None:
            return None
        try:
            return max(y for y in dm.years if y < dm.forecast_from)
        except ValueError:  # no historical years, or forecast_from is None
            return None

    def _impact_for_action(self, action: ActionNode, node: Node, year: int) -> ActionImpactType:
        from nodes.schema import ActionImpactType
        df = action.compute_impact(node)
        df = df.filter(pl.col(IMPACT_COLUMN) == IMPACT_GROUP).drop(IMPACT_COLUMN)
        df = df.filter(pl.col(YEAR_COLUMN) == year)
        df = df.paths.sum_over_dims()
        # Normalize
        active_normalization = node.context.active_normalization
        metric = node.get_default_output_metric()
        if active_normalization and active_normalization.get_normalized_unit(metric) is not None:
            _, df = active_normalization.normalize_output(metric, df)
        return ActionImpactType(
            action=cast('ActionNodeType', action),
            value=df.item(0, VALUE_COLUMN),
            year=year,
        )
