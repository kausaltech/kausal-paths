"""
A goal an instance sets for a node: the target path of an outcome, such as net emissions.

A goal is part of the node's definition (`NodeSpec.goals`). An instance that inherits its
nodes from a framework template does not own that definition, so its goal is one of its local
settings for the inherited node (`InheritedNodeSettings.goals`), which composition lays over
the template's. Either way the runtime sees one `NodeGoalsEntry` on the node.

Only what was entered is stored: the points, and for a relative goal the reference year. A
relative goal is turned into amounts when it is read, against the node's own value in the
reference year, so it follows a correction of that year's balance. The reference year has to
be an inventory year for that reason: a figure from elsewhere was not computed by this model.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kausal_common.i18n.pydantic import TranslatedString

from nodes.change_ops import record_change
from nodes.defs.node_defs import ActionConfig
from nodes.goals import GoalValue, NodeGoals, NodeGoalsEntry
from nodes.instance_serialization import InheritedNodeSettings
from nodes.inventory_years import declared_inventory_years
from nodes.models import InstanceConfig, NodeConfig

if TYPE_CHECKING:
    from uuid import UUID

    from nodes.instance_graph import InstanceGraph, NodeMeta


class InstanceGoalError(ValueError):
    """A goal that cannot be set; the message is for the user."""


@dataclass(frozen=True)
class GoalPoint:
    year: int
    value: float


def _target(graph: InstanceGraph, node_id: UUID) -> NodeMeta:
    node = graph.node_by_id.get(node_id)
    if node is None:
        raise InstanceGoalError('The node is not part of the model')
    name = node.identifier or str(node.id)
    if isinstance(node.spec.type_config, ActionConfig):
        raise InstanceGoalError(f'{name} is an action; a goal belongs to an outcome')
    if len(node.spec.output_ports) != 1:
        raise InstanceGoalError(f'{name} has {len(node.spec.output_ports)} outputs; a goal needs a node with one')
    if not node.spec.is_outcome:
        # Only the goals of outcome nodes are listed as goals of the instance.
        raise InstanceGoalError(f'{name} is not an outcome of the model')
    return node


def build_goal(
    ic: InstanceConfig, graph: InstanceGraph, points: list[GoalPoint], *, reference_year: int | None, label: str | None
) -> NodeGoalsEntry:
    """
    Check what was entered and return the goal it describes.

    The goal's path runs in a straight line from the node's value in the last historical year
    through the points given, so the gap to the path can be read in every year up to the last one.
    """
    if not points:
        raise InstanceGoalError('A goal needs at least one year and value')
    years = [point.year for point in points]
    if len(set(years)) != len(years):
        raise InstanceGoalError('A goal has one value for each year')
    spec_years = graph.spec.years
    last_historical = spec_years.max_historical
    last_year = spec_years.model_end or spec_years.target
    for point in points:
        if last_historical is not None and point.year <= last_historical:
            raise InstanceGoalError(f'{point.year} is not after the last inventory year, {last_historical}')
        if last_year is not None and point.year > last_year:
            raise InstanceGoalError(f'{point.year} is after the last year of the model, {last_year}')
    if reference_year is None:
        if any(point.value < 0 for point in points):
            raise InstanceGoalError('An absolute goal is an amount, which cannot be negative')
    else:
        if reference_year not in declared_inventory_years(ic).years:
            raise InstanceGoalError(f'{reference_year} is not an inventory year, so there is no value to refer to')
        if any(point.value < -1 for point in points):
            raise InstanceGoalError('A reduction cannot exceed 100 %')
    return NodeGoalsEntry(
        values=[GoalValue(year=point.year, value=point.value) for point in sorted(points, key=lambda point: point.year)],
        reference_year=reference_year,
        # Given its language here: a goal can be set outside a request, where there is no language context.
        label=TranslatedString(label, default_language=ic.primary_language) if label else None,
        linear_interpolation=True,
        auto_start=True,
        is_main_goal=True,
        default=True,
    )


def _local_node(ic: InstanceConfig, node_id: UUID) -> NodeConfig | None:
    return ic.nodes.get_queryset().with_spec().filter(uuid=node_id).first()


def _goals_snapshot(goals: NodeGoals | None) -> dict[str, object] | None:
    if goals is None or not goals.root:
        return None
    return {'goals': goals.model_dump(mode='json', exclude_none=True)}


def _replace_goals(ic: InstanceConfig, node: NodeMeta, goals: NodeGoals | None) -> None:
    """Store `goals` as the node's, where the instance keeps them. The caller holds an open change operation."""
    local = _local_node(ic, node.id)
    if local is not None:
        spec = local.spec
        assert spec is not None
        before = _goals_snapshot(spec.goals)
        spec.goals = goals if goals is not None else NodeGoals()
        # queryset.update(): NodeConfig.save() goes through ClusterableModel, which can revert i18n fields.
        NodeConfig.objects.filter(pk=local.pk).update(spec=spec)
        record_change(local, action='node.goal.set', before=before, after=_goals_snapshot(goals), target_uuid=node.id)
        return
    ic.refresh_from_db(fields=['node_settings'])
    settings_list: list[InheritedNodeSettings] = []
    before = None
    found = False
    for settings in ic.node_settings:
        if settings.node_uuid != node.id:
            settings_list.append(settings)
            continue
        found = True
        before = _goals_snapshot(settings.goals)
        updated = settings.model_copy(update={'goals': goals})
        # Settings that hold nothing else are dropped, so the template's own goals show again.
        if updated != InheritedNodeSettings(node_uuid=node.id):
            settings_list.append(updated)
    if not found and goals is not None:
        settings_list.append(InheritedNodeSettings(node_uuid=node.id, goals=goals))
    ic.node_settings = settings_list
    ic.save(update_fields=['node_settings'])
    record_change(ic, action='node.goal.set', before=before, after=_goals_snapshot(goals), target_uuid=node.id)


def set_node_goal(ic: InstanceConfig, graph: InstanceGraph, node_id: UUID, goal: NodeGoalsEntry) -> None:
    """Make `goal` the node's one goal for this instance, replacing whichever it had."""
    _replace_goals(ic, _target(graph, node_id), NodeGoals([goal]))


def clear_node_goal(ic: InstanceConfig, graph: InstanceGraph, node_id: UUID) -> None:
    """
    Remove the instance's goal for the node.

    An inherited node goes back to the goals its template gives it, if any.
    """
    _replace_goals(ic, _target(graph, node_id), None)
