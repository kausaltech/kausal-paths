from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import graphene
from django import forms
from django.forms import ValidationError
from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from wagtail import blocks
from wagtail.admin.telepath import register as register_telepath_adapter
from wagtail.blocks.struct_block import StructBlockAdapter, StructBlockValidationError
from wagtail.images.blocks import ImageChooserBlock

from grapple.helpers import register_streamfield_block
from grapple.models import GraphQLBoolean, GraphQLField, GraphQLFloat, GraphQLImage, GraphQLInt, GraphQLStreamfield, GraphQLString
from wagtail_color_panel.blocks import NativeColorBlock

from nodes.blocks import NodeChooserBlock
from nodes.constants import IMPACT_COLUMN, IMPACT_GROUP, VALUE_COLUMN, YEAR_COLUMN
from nodes.metric import DimensionalMetric

if TYPE_CHECKING:
    from collections.abc import Iterable

    from paths.types import PathsGQLInfo as GQLInfo

    from frameworks.models import Framework
    from nodes.actions.action import ActionNode
    from nodes.graphql.types.metric import MetricDimensionCategoryType, MetricDimensionType
    from nodes.graphql.types.node import ActionNodeType
    from nodes.graphql.types.scenario import (
        ActionImpactType,
        MetricDimensionCategoryValue,
        ScenarioActionImpacts,
        ScenarioType,
        ScenarioValue,
    )
    from nodes.metric import MetricYearlyGoal
    from nodes.node import Node
    from nodes.scenario import Scenario
    from nodes.units import Unit
    from params.base import Parameter


@register_streamfield_block
class FrameworkLandingBlock(blocks.StructBlock):
    heading = blocks.CharBlock(label=_('Heading'))
    body = blocks.RichTextBlock(label=_('Body'), required=False)
    cta_label = blocks.CharBlock(label=_('Call-to-action button label'), required=False)
    cta_url = blocks.CharBlock(label=_('Call-to-action button URL'), required=False)
    framework_identifier = blocks.CharBlock(
        label=_('Framework identifier'),
        help_text=_('Identifier of the framework to display on the landing page.'),
    )

    graphql_fields = [
        GraphQLString('heading', required=True),
        GraphQLString('body', required=False),
        GraphQLString('cta_label', required=False),
        GraphQLString('cta_url', required=False),
        GraphQLField('framework', 'frameworks.schema.FrameworkType', required=False),
    ]

    def framework(self, info: GQLInfo, values: dict[str, Any]) -> Framework | None:
        from frameworks.models import Framework

        identifier = values.get('framework_identifier')
        if not identifier:
            return None
        fw = Framework.objects.filter(identifier=identifier).first()
        if fw is None:
            return None
        return info.context.cache.for_framework(fw)


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
        help_text=_('Break down the given dimension into its categories. If empty, break down into input nodes.'),
    )

    # TODO Validate that the dimension id actually exists in the instance.
    # However, for this we'd need access to the node (from the parent block) here, which seems to be difficult.

    graphql_fields = [
        GraphQLString('title', required=True),
        GraphQLString('dimension_id', required=True),
    ]


def admin_instance_scenario_choices() -> list[tuple[str, str]]:
    """
    Return the scenarios of the instance being edited in the admin, as block choices.

    The custom scenario is left out: it is each visitor's own, so a block set to it would
    show something different to everyone. Outside an admin request there is no instance
    to ask, and an empty list only matters to form rendering and validation, which do not
    happen there.
    """
    from paths.context import realm_context

    if not realm_context.is_set():
        return []
    context = realm_context.get().realm.get_instance().context
    return [
        (scenario.id, str(scenario.name)) for scenario in context.scenarios.values() if scenario is not context.custom_scenario
    ]


def _parameter_choice_label(param: Parameter[Any, Any]) -> str:
    label = str(param.label) if param.label is not None else param.local_id
    if param.node is None:
        return label
    return f'{param.node.name}: {label}'


def admin_instance_parameter_choices() -> list[tuple[str, str]]:
    """
    Return the customizable parameters of the instance being edited in the admin, as block choices.

    Global parameters come first, then node parameters, each sorted by label. A parameter that is
    not customizable is left out: a visitor could not change it, and `setParameter` refuses it.
    Outside an admin request there is no instance to ask (see `admin_instance_scenario_choices`).
    """
    from paths.context import realm_context

    if not realm_context.is_set():
        return []
    context = realm_context.get().realm.get_instance().context
    params = [param for param in context.get_all_parameters() if param.is_customizable]
    params.sort(key=lambda param: (param.node is not None, _parameter_choice_label(param).lower()))
    return [(param.global_id, _parameter_choice_label(param)) for param in params]


def admin_instance_dimension_choices() -> list[tuple[str, str]]:
    """Return the dimensions of the instance being edited in the admin, as block choices."""
    from paths.context import realm_context

    if not realm_context.is_set():
        return []
    context = realm_context.get().realm.get_instance().context
    dims = sorted(context.dimensions.values(), key=lambda dim: str(dim.label).lower())
    return [(dim.id, f'{dim.label} ({dim.id})') for dim in dims]


CATEGORY_CHOICE_SEPARATOR = ':'


def admin_instance_category_choices() -> list[tuple[str, list[tuple[str, str]]]]:
    """
    Return the categories of the instance being edited in the admin, as block choices grouped by dimension.

    A choice is `<dimension>:<category>`, which is unambiguous because identifiers cannot contain a
    colon. The admin form shows only the groups and categories of the chosen node (see
    `ParameterPanelGraphBlock`); the server accepts any, and `clean` checks them against the node.
    """
    from paths.context import realm_context

    if not realm_context.is_set():
        return []
    context = realm_context.get().realm.get_instance().context
    dims = sorted(context.dimensions.values(), key=lambda dim: str(dim.label).lower())
    return [
        (str(dim.label), [(f'{dim.id}{CATEGORY_CHOICE_SEPARATOR}{cat.id}', str(cat.label)) for cat in dim.categories])
        for dim in dims
        if dim.categories
    ]


def _split_category_choices(choices: Iterable[object]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for choice in choices:
        # Anything else is a value saved in an older shape of the block; it is ignored, not fatal.
        if not isinstance(choice, str):
            continue
        dim_id, sep, cat_id = choice.partition(CATEGORY_CHOICE_SEPARATOR)
        if sep and dim_id and cat_id:
            pairs.append((dim_id, cat_id))
    return pairs


class ParameterPanelGraphFilterType(graphene.ObjectType):
    """The categories a graph shows of one dimension until the visitor changes the filter."""

    dimension = graphene.String(required=True)
    categories = graphene.List(graphene.NonNull(graphene.String), required=True)


class ParameterPanelGraphType(graphene.ObjectType):
    """A node graph in a parameter panel, with the defaults the editor chose for it."""

    node_id = graphene.ID(required=True)
    group_by = graphene.String(description='Dimension to break the graph down by, unless the visitor chooses another.')
    filters = graphene.List(graphene.NonNull(ParameterPanelGraphFilterType), required=True)


@dataclass(frozen=True)
class ParameterPanelGraphFilterData:
    dimension: str
    categories: list[str]


@dataclass(frozen=True)
class ParameterPanelGraphData:
    node_id: str
    group_by: str | None
    filters: list[ParameterPanelGraphFilterData]


class ParameterPanelGraphBlock(blocks.StructBlock):
    """
    One graph in a parameter panel: a node, and how its graph starts out.

    The break-down dimension and the filters are defaults only. The visitor can change both with the
    graph's own controls, as on the node page.
    """

    node = NodeChooserBlock(label=_('Node'))
    group_by = blocks.ChoiceBlock(
        # Wagtail evaluates callable choices at render time; the stubs only admit a static list.
        choices=admin_instance_dimension_choices,  # type: ignore[arg-type]
        required=False,
        label=_('Break down by'),
        help_text=_('Leave empty to use the default dimension of the graph.'),
    )
    filters = blocks.MultipleChoiceBlock(
        choices=admin_instance_category_choices,  # type: ignore[arg-type]
        required=False,
        widget=forms.CheckboxSelectMultiple,
        label=_('Show only these categories'),
        help_text=_(
            'The graph starts filtered to the ticked categories; those of different dimensions combine. '
            'Leave a dimension unticked to show all of it. Visitors can change the filters.'
        ),
    )

    class Meta:
        label = _('Graph')

    def clean(self, value: blocks.StructValue) -> blocks.StructValue:
        """Refuse a break-down dimension or filter category that the chosen node does not have."""
        from paths.context import realm_context

        from nodes.models import NodeConfig

        value = super().clean(value)
        node_config = value.get('node')
        if not isinstance(node_config, NodeConfig) or not realm_context.is_set():
            return value
        context = realm_context.get().realm.get_instance().context
        node = context.nodes.get(node_config.identifier)
        if node is None:
            return value
        dims = node.output_dimensions
        errors: list[str] = []
        group_by = value.get('group_by')
        if group_by and group_by not in dims:
            errors.append(str(_('The node has no dimension "%s" to break down by.') % group_by))
        filters = cast('list[str]', value.get('filters') or [])
        for dim_id, cat_id in _split_category_choices(filters):
            dim = dims.get(dim_id)
            if dim is None:
                errors.append(str(_('The node has no dimension "%s" to filter by.') % dim_id))
            elif cat_id not in dim.cat_map:
                errors.append(
                    str(_('The dimension "%(dim)s" of the node has no category "%(cat)s".') % {'dim': dim_id, 'cat': cat_id})
                )
        if errors:
            raise StructBlockValidationError(non_block_errors=list(dict.fromkeys(errors)))
        return value


class ParameterPanelGraphBlockAdapter(StructBlockAdapter):
    """
    Narrow the graph's choices in the admin form to the chosen node.

    A block's choices are fixed when the form is built, before the editor picks a node, so the
    narrowing happens in the browser: the script asks `node_graph_options` for the node's dimensions
    and the categories it has data for, and hides the rest.
    """

    js_constructor = 'paths.pages.ParameterPanelGraphBlock'

    def js_args(self, block: blocks.Block) -> list[Any]:
        args = super().js_args(block)
        meta = args[2]
        meta['nodeOptionsUrl'] = reverse('pages_node_graph_options')
        meta['texts'] = {
            'chooseNode': str(_('Choose a node to see its dimensions and categories.')),
            'noDimensions': str(_('This node has no dimensions to break down or filter by.')),
        }
        return args

    @cached_property
    def media(self) -> forms.Media:
        return super().media + forms.Media(js=['pages/js/parameter_panel_graph_block.js'])


register_telepath_adapter(ParameterPanelGraphBlockAdapter(), ParameterPanelGraphBlock)


@register_streamfield_block
class ParameterPanelBlock(blocks.StructBlock):
    """
    A panel of parameters, chosen by the editor, that the visitor can change.

    The panel writes to the visitor's session, as the settings panel does, so a change applies
    to every page the visitor reads afterwards, and graphs elsewhere on the page follow it. The
    text shown for each parameter is the parameter's own `description`, so it is written once.
    """

    title = blocks.CharBlock(required=False, label=_('Title'))
    parameter_ids = blocks.ListBlock(
        blocks.ChoiceBlock(
            # Wagtail evaluates callable choices at render time; the stubs only admit a static list.
            choices=admin_instance_parameter_choices,  # type: ignore[arg-type]
            label=_('Parameter'),
        ),
        min_num=1,
        label=_('Parameters'),
        help_text=_('Only parameters that visitors may change are listed. They are shown in this order.'),
    )
    show_reset = blocks.BooleanBlock(
        required=False,
        default=True,
        label=_('Show reset button'),
        help_text=_("Let the visitor return this panel's parameters to their default values."),
    )
    graphs = blocks.ListBlock(
        ParameterPanelGraphBlock(),
        required=False,
        default=[],
        label=_('Graphs'),
        help_text=_(
            "Graphs shown below the parameters, in this order. They follow the visitor's settings, so they "
            'change as the parameters do.'
        ),
    )

    graphql_fields = [
        GraphQLString('title', required=False),
        GraphQLField('parameters', 'params.schema.ParameterInterface', is_list=True, required=True),  # pyright: ignore
        GraphQLBoolean('show_reset', required=True),
        GraphQLField('node_graphs', ParameterPanelGraphType, is_list=True, required=True),
    ]

    class Meta:
        icon = 'cogs'
        label = _('Parameter panel')

    def parameters(self, info: GQLInfo, values: dict[str, Any]) -> list[Parameter[Any, Any]]:
        """
        Return the chosen parameters in the instance the page is served for, in the editor's order.

        Visitors are served the published revision, which may lack a parameter the draft has, or
        have made it fixed since the page was edited. Such a parameter is left out rather than
        failing the page; the editor sees the problem when the block is next edited, because the
        choice is no longer offered.
        """
        context = info.context.require_instance().context
        result: list[Parameter[Any, Any]] = []
        seen: set[str] = set()
        for param_id in values.get('parameter_ids') or []:
            if param_id in seen:
                continue
            seen.add(param_id)
            param = context.get_parameter(param_id, required=False)
            if param is None or not param.is_customizable:
                continue
            result.append(param)
        return result

    def node_graphs(self, info: GQLInfo, values: dict[str, Any]) -> list[ParameterPanelGraphData]:
        """
        Return the graphs to draw, in the editor's order.

        Only the node identifier and the defaults are returned: the client fetches each graph with its
        own query, so the page does not wait for the computation, and a parameter change refetches the
        graphs alone. As for parameters, the served instance may differ from the draft the editor chose
        from, so a graph whose node it lacks is left out, and so is a filter or break-down dimension
        the node no longer has.
        """
        context = info.context.require_instance().context
        result: list[ParameterPanelGraphData] = []
        for graph in values.get('graphs') or []:
            node_config = graph.get('node')
            if node_config is None:
                continue
            node = context.nodes.get(node_config.identifier)
            if node is None:
                continue
            dims = node.output_dimensions
            group_by = graph.get('group_by') or None
            filters: dict[str, list[str]] = {}
            for dim_id, cat_id in _split_category_choices(graph.get('filters') or []):
                dim = dims.get(dim_id)
                if dim is None or cat_id not in dim.cat_map:
                    continue
                cats = filters.setdefault(dim_id, [])
                if cat_id not in cats:
                    cats.append(cat_id)
            result.append(
                ParameterPanelGraphData(
                    node_id=node.id,
                    group_by=group_by if group_by in dims else None,
                    filters=[ParameterPanelGraphFilterData(dimension=d, categories=c) for d, c in filters.items()],
                )
            )
        return result


@register_streamfield_block
class ActionImpactBlock(blocks.StructBlock):
    title = blocks.CharBlock()
    scenario_id = blocks.ChoiceBlock(
        # Wagtail evaluates callable choices at render time; the stubs only admit a static list.
        choices=admin_instance_scenario_choices,  # type: ignore[arg-type]
        label=_('Scenario'),
        help_text=_('Show the impact of each action as computed in this scenario, whichever scenario the visitor has selected.'),
    )

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
    year = blocks.IntegerBlock(
        required=False,
        label=_('Year'),
        help_text=_(
            "The year whose values the card shows. Leave empty for the instance's target year. Use it for a "
            'quantity whose year of interest is another one, such as a budget used up by the end of the model.'
        ),
    )
    visualizations = blocks.StreamBlock([
        ('goal_progress_bar', GoalProgressBarBlock()),
        ('reference_progress_bar', ReferenceProgressBarBlock()),
        ('current_progress_bar', CurrentProgressBarBlock()),
        ('scenario_progress_bar', ScenarioProgressBarBlock()),
        (
            'category_breakdown',
            CategoryBreakdownBlock(label=_('Category breakdown'), help_text=_('Break down the value into its components')),
        ),
        (
            'action_impact',
            ActionImpactBlock(label=_('Action impact'), help_text=_('Visualize the impact of actions in a scenario')),
        ),
    ])
    call_to_action = CallToActionBlock()

    graphql_fields = [
        GraphQLString('title', required=True),
        GraphQLString('description', required=True),
        GraphQLImage('image', required=False),
        GraphQLField('node', 'nodes.graphql.types.node.NodeType', required=True),  # pyright: ignore
        GraphQLField('unit', 'paths.schema.UnitType', required=True),  # pyright: ignore
        GraphQLInt('year', required=False),
        GraphQLFloat('goal_value', required=False, deprecation_reason='Use goalValues instead'),
        GraphQLField(
            'goal_values',
            'nodes.graphql.types.metric.MetricYearlyGoalType',  # pyright: ignore
            is_list=True,
            required=True,
        ),
        GraphQLFloat('reference_year_value', required=False),
        GraphQLFloat('last_historical_year_value', required=False),
        GraphQLField('scenario_values', 'nodes.graphql.types.scenario.ScenarioValue', is_list=True, required=True),  # pyright: ignore
        GraphQLField(
            'metric_dimension_category_values',
            'nodes.graphql.types.scenario.MetricDimensionCategoryValue',  # pyright: ignore
            is_list=True,
            required=False,  # can be null if there is no historical data
        ),
        GraphQLField(
            'scenario_action_impacts',
            'nodes.graphql.types.scenario.ScenarioActionImpacts',  # pyright: ignore
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
            errors['node_config'] = ValidationError(_('This object does not correspond to a node.'))
        elif not node.goals or not node.goals.root:
            errors['node_config'] = ValidationError(_('This node has no goals.'))
        elif node.context.instance.reference_year is None:
            errors['node_config'] = ValidationError(_("This node's instance has no reference year."))

        goal_index = cleaned_data.get('goal_index')
        if node and node.goals and goal_index is not None and goal_index >= len(node.goals.root):
            errors['goal_index'] = ValidationError(_('This goal index is invalid.'))

        if errors:
            raise blocks.StructBlockValidationError(errors)
        return cleaned_data

    def node(self, info: GQLInfo, values: dict) -> Node:
        from nodes.models import NodeConfig

        node_config = values['node_config']
        assert isinstance(node_config, NodeConfig)
        # The page's own model: a page is served for the instance the operation names.
        node = info.context.require_instance().context.nodes.get(node_config.identifier)
        if not node:
            raise ValueError('Node config has no node.')  # hopefully prevented by validation
        return node

    def unit(self, info: GQLInfo, values: dict) -> Unit:
        node = self.node(info, values)
        dm = self._dimensional_metric(node)
        return dm.unit

    def _card_year(self, node: Node, values: dict) -> int:
        """Return the year the card shows: its own `year`, or else the node's target year."""
        year = values.get('year')
        if year is not None:
            return year
        target_year = node.get_target_year()
        if target_year is None:
            raise ValueError('Node has no target year')
        return target_year

    def goal_value(self, info: GQLInfo, values: dict) -> float | None:
        """Return the value for the chosen goal in the card's year."""
        node = self.node(info, values)
        goal_index = values.get('goal_index')
        return self._goal_value_for_year(node, self._card_year(node, values), goal_index)

    def goal_values(self, info: GQLInfo, values: dict) -> list[MetricYearlyGoal]:
        """Return the values for all years for the chosen goal."""
        node = self.node(info, values)
        goal_index = values.get('goal_index')
        return self._goal_values(node, goal_index)

    def reference_year_value(self, info: GQLInfo, values: dict) -> float | None:
        node = self.node(info, values)
        reference_year = node.context.instance.reference_year
        if reference_year is None:
            raise ValueError('Instance has no reference year')
        return self._value_for_year(node, reference_year)

    def last_historical_year_value(self, info: GQLInfo, values: dict) -> float | None:
        node = self.node(info, values)
        dm = self._dimensional_metric(node)
        last_historical_year = self._last_historical_year(dm)
        if last_historical_year is None:
            return None
        return self._value_for_year(node, last_historical_year)

    def scenario_values(self, info: GQLInfo, values: dict) -> Iterable[ScenarioValue]:
        """Return the value for each scenario in the card's year."""
        from nodes.graphql.types.scenario import ScenarioValue

        node = self.node(info, values)
        target_year = self._card_year(node, values)
        return [
            ScenarioValue(
                scenario=cast('ScenarioType', s),
                year=target_year,
                value=self._value_for_year(node, target_year, s),
            )
            for s in node.context.scenarios.values()
        ]

    def metric_dimension_category_values(self, info: GQLInfo, values: dict) -> Iterable[MetricDimensionCategoryValue] | None:
        """
        Return the value for each dimension and each category for the node's target year.

        Returns None if there is no historical data.
        """
        import polars as pl

        from nodes.graphql.types.scenario import MetricDimensionCategoryValue

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
                result.append(
                    MetricDimensionCategoryValue(
                        dimension=cast('MetricDimensionType', dimension),  # pyright: ignore[reportArgumentType]
                        category=cast('MetricDimensionCategoryType', category),  # pyright: ignore[reportArgumentType]
                        value=value,
                        year=year,
                    )
                )
        return result

    def scenario_action_impacts(self, info: GQLInfo, values: dict) -> Iterable[ScenarioActionImpacts]:
        """
        Return the impact of each action in the card's year, per scenario.

        Only the scenarios an `action_impact` visualization of this card names are computed:
        each one is a full set of model runs, and nothing else reads this field.

        Each scenario is computed isolated from the visitor's session, so the chart shows
        the scenario its editor chose and does not move when the visitor edits parameters.
        Whether an action is enabled is reported per entry for the same reason --
        `action.isEnabled` is resolved after this returns, in the visitor's scenario.
        """
        from nodes.graphql.types.scenario import ScenarioActionImpacts

        node = self.node(info, values)
        target_year = self._card_year(node, values)
        context = node.context
        scenario_ids = {child.value['scenario_id'] for child in values['visualizations'] if child.block_type == 'action_impact'}
        result = []
        for scenario_id in sorted(scenario_ids):
            scenario = context.scenarios.get(scenario_id)
            if scenario is None:
                continue
            with scenario.override(set_active=True, isolated=True):
                impacts = [self._impact_for_action(action, node, target_year) for action in context.get_actions()]
            result.append(ScenarioActionImpacts(scenario=cast('ScenarioType', scenario), impacts=impacts))
        return result

    def _dimensional_metric(self, node: Node, scenario: Scenario | None = None) -> DimensionalMetric:
        context = scenario.override(isolated=True) if scenario else nullcontext()
        with context:
            dm = DimensionalMetric.from_node(node)
        if not dm:
            raise ValueError('Could not obtain dimensional metric from node')
        return dm

    def _goal_values(self, node: Node, goal_index: int | None) -> list[MetricYearlyGoal]:
        if goal_index is None:
            goal_index = 0
        dm = self._dimensional_metric(node)
        if goal_index < 0 or goal_index >= len(dm.goals):
            raise ValueError('Goal index is invalid')
        goal = dm.goals[goal_index]
        return goal.values

    def _goal_value_for_year(self, node: Node, year: int, goal_index: int | None) -> float | None:
        values = self._goal_values(node, goal_index)
        year_values = (v for v in values if v.year == year)
        try:
            return next(iter(year_values)).value
        except StopIteration:
            return None

    def _value_for_year(self, node: Node, year: int, scenario: Scenario | None = None) -> float | None:
        import polars as pl

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
        # If forecast_from is None, this indicates there are no forecast years in the relevant DF. Thus
        # all years (if any) are historical years, and the last year should be returned.
        if dm.forecast_from is None:
            try:
                return max(dm.years)
            except ValueError:  # No years.
                return None
        else:
            try:
                return max(y for y in dm.years if y < dm.forecast_from)
            except ValueError:  # No historical years.
                return None

    def _impact_for_action(self, action: ActionNode, node: Node, year: int) -> ActionImpactType:
        import polars as pl

        from nodes.graphql.types.scenario import ActionImpactType

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
            is_enabled=action.is_enabled(),
        )
