from __future__ import annotations

from typing import TYPE_CHECKING

import graphene
from django.forms import ValidationError
from django.utils.translation import gettext_lazy as _
from wagtail import blocks
from wagtail.images.blocks import ImageChooserBlock

import polars as pl
from grapple.helpers import register_streamfield_block
from grapple.models import GraphQLField, GraphQLFloat, GraphQLStreamfield, GraphQLString
from wagtail_color_panel.blocks import NativeColorBlock

from nodes.blocks import NodeChooserBlock
from nodes.constants import VALUE_COLUMN, YEAR_COLUMN

if TYPE_CHECKING:
    from paths.types import GQLInstanceInfo

    from nodes.node import Node
    from nodes.scenario import Scenario
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
        def build_type():
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
        GraphQLString('description'),
        GraphQLString('chart_label', required=True),
        GraphQLString('color'),
    ]


class NodeProgressBarBlock(ProgressBarBlock):
    """Base class for progress bar blocks with common fields and node field."""

    node_config = NodeChooserBlock(required=True)

    graphql_fields = ProgressBarBlock.graphql_fields + [
        GraphQLField('node', 'nodes.schema.NodeType', required=True),  # pyright: ignore
        GraphQLField('unit', 'paths.schema.UnitType', required=True),  # pyright: ignore
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
        return node.single_metric_unit


@register_streamfield_block
class GoalProgressBarBlock(NodeProgressBarBlock):
    graphql_fields = NodeProgressBarBlock.graphql_fields + [
        GraphQLFloat('value', required=False),
    ]

    def clean(self, value):
        from nodes.models import NodeConfig

        cleaned_data = super().clean(value)
        errors = {}
        node_config = cleaned_data['node_config']
        assert isinstance(node_config, NodeConfig)
        node = node_config.get_node()
        assert node  # ensured by NodeProgressBarBlock.clean()
        if not node.goals or not node.goals.root:
            errors['node_config'] = ValidationError(_("This node has no goals."))
        elif len(node.goals.root) > 1:
            errors['node_config'] = ValidationError(_("This node has more than one goal."))
        if errors:
            raise blocks.StructBlockValidationError(errors)
        return cleaned_data

    def value(self, info: GQLInstanceInfo, values: dict) -> float | None:
        node = self.node(info, values)
        if not node.goals or not node.goals.root:
            return None
        if len(node.goals.root) > 1:
            raise ValueError("Node has more than one goal")
        target_year = node.get_target_year()
        if target_year is None:
            raise ValueError("Node has no target year")
        goal = node.goals.root[0]
        target_year_values = (v for v in goal.get_values() if v.year == target_year)
        try:
            return next(iter(target_year_values)).value
        except StopIteration:
            return None


@register_streamfield_block
class ReferenceProgressBarBlock(NodeProgressBarBlock):
    graphql_fields = NodeProgressBarBlock.graphql_fields + [
        GraphQLFloat('value', required=False),
    ]

    def clean(self, value):
        from nodes.models import NodeConfig

        cleaned_data = super().clean(value)
        errors = {}
        node_config = cleaned_data['node_config']
        assert isinstance(node_config, NodeConfig)
        node = node_config.get_node()
        assert node  # ensured by NodeProgressBarBlock.clean()
        if node.context.instance.reference_year is None:
            errors['node_config'] = ValidationError(_("This node's instance has no reference year."))
        if errors:
            raise blocks.StructBlockValidationError(errors)
        return cleaned_data

    def value(self, info: GQLInstanceInfo, values: dict) -> float | None:
        node = self.node(info, values)
        reference_year = info.context.instance.reference_year
        if reference_year is None:
            raise ValueError("Instance has no reference year")
        df = node.get_output_pl()
        df = df.paths.sum_over_dims()
        df = df.filter(pl.col(YEAR_COLUMN) == reference_year)
        if df.is_empty():
            return None
        if len(df) > 1:
            raise ValueError(f"Node output contains multiple rows for year {reference_year}")
        return df.item(0, VALUE_COLUMN)


@register_streamfield_block
class CurrentProgressBarBlock(NodeProgressBarBlock):
    graphql_fields = NodeProgressBarBlock.graphql_fields + [
        GraphQLFloat('value', required=False),
    ]

    def value(self, info: GQLInstanceInfo, values: dict) -> float | None:
        node = self.node(info, values)
        df = node.get_output_pl()
        last_historical_year = df.paths.get_last_historical_year()
        df = df.filter(pl.col(YEAR_COLUMN) == last_historical_year)
        if df.is_empty():
            return None
        if len(df) > 1:
            raise ValueError(f"Node output contains multiple rows for year {last_historical_year}")
        return df.item(0, VALUE_COLUMN)


@register_streamfield_block
class ScenarioProgressBarBlock(NodeProgressBarBlock):
    scenario_identifier = blocks.CharBlock(required=True)  # FIXME: choice block? But where to get the choices?

    graphql_fields = NodeProgressBarBlock.graphql_fields + [
        GraphQLField('scenario', 'nodes.schema.ScenarioType', required=True),  # pyright: ignore
        GraphQLFloat('value', required=False),
    ]

    def clean(self, value):
        from nodes.models import NodeConfig

        cleaned_data = super().clean(value)
        errors = {}
        node_config = cleaned_data['node_config']
        assert isinstance(node_config, NodeConfig)
        node = node_config.get_node()
        assert node  # ensured by NodeProgressBarBlock.clean()
        scenario_identifier = cleaned_data['scenario_identifier']
        if scenario_identifier not in node.context.scenarios:
            errors['scenario_identifier'] = ValidationError(
                _("This scenario does not exist in the given node's instance.")
            )
        if errors:
            raise blocks.StructBlockValidationError(errors)
        return cleaned_data

    def scenario(self, info: GQLInstanceInfo, values: dict) -> Scenario:
        scenario_identifier = values['scenario_identifier']
        node = self.node(info, values)
        assert info.context.instance == node.context.instance
        return node.context.get_scenario(scenario_identifier)

    # TODO: We'll also need to expose the goal year value for node (dotted line)

    def value(self, info: GQLInstanceInfo, values: dict) -> float | None:
        node = self.node(info, values)
        target_year = node.get_target_year()
        if target_year is None:
            raise ValueError("Node has no target year")
        scenario = self.scenario(info, values)
        with scenario.override():
            df = node.get_output_pl()
        df = df.filter(pl.col(YEAR_COLUMN) == target_year)
        if df.is_empty():
            return None
        if len(df) > 1:
            raise ValueError(f"Node output contains multiple rows for year {target_year}")
        return df.item(0, VALUE_COLUMN)


@register_streamfield_block
class ProgressVisualizationBlock(blocks.StructBlock):
    progress_bars = blocks.StreamBlock([
        ('goal_progress_bar', GoalProgressBarBlock()),
        ('reference_progress_bar', ReferenceProgressBarBlock()),
        ('current_progress_bar', CurrentProgressBarBlock()),
        ('scenario_progress_bar', ScenarioProgressBarBlock()),
    ])

    graphql_fields = [
        GraphQLStreamfield('progress_bars', required=True),
    ]


# class EmissionSourcesVisualizationBlock(blocks.StructBlock):
#     title = blocks.CharBlock()
#     # TODO: Other fields
#
#     graphql_fields = [
#         GraphQLString('title', required=True),
#     ]


@register_streamfield_block
class VisualizationBlock(blocks.StreamBlock):
    progress = ProgressVisualizationBlock()
    # emission_sources = EmissionSourcesVisualizationBlock()  # TODO


@register_streamfield_block
class CallToActionBlock(blocks.StructBlock):
    title = blocks.CharBlock(label=_('Title'))
    content = blocks.CharBlock(label=_('Content'), required=False)
    link_url = blocks.CharBlock(label=_('Link URL'))

    graphql_fields = [
        GraphQLString('title', required=True),
        GraphQLString('content'),
        GraphQLString('link_url', required=True),
    ]


@register_streamfield_block
class DashboardCardBlock(blocks.StructBlock):
    title = blocks.CharBlock()
    description = blocks.CharBlock(required=False)
    image = ImageChooserBlock(required=False)
    visualization = VisualizationBlock()
    call_to_action = CallToActionBlock()

    graphql_fields = [
        GraphQLString('title', required=True),
        GraphQLString('description'),
        GraphQLString('image'),
        GraphQLStreamfield('visualization', required=True),
        GraphQLBlockField('call_to_action', CallToActionBlock, is_list=False, required=True),
    ]
