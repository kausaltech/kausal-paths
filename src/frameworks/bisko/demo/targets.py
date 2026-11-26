"""
Demo climate targets for the named demo municipalities.

A real municipality has no target until someone sets one: the template gives none, and nothing
here runs for an instance outside `DEMO_MUNICIPALITIES`. The demo cities get one so that the
scenario view has a gap to show.
"""

from frameworks.bisko.demo.inventories import DEMO_MUNICIPALITIES
from frameworks.models import Framework, FrameworkConfig
from nodes.change_ops import change_operation
from nodes.instance_goals import GoalPoint, build_goal, set_node_goal
from nodes.instance_graph import build_instance_graph
from nodes.instance_serialization import build_instance_snapshot
from nodes.inventory_years import declared_inventory_years
from nodes.models import InstanceChangeSource, InstanceConfig

TARGET_NODE = 'net_emissions'
DEMO_TARGET_LABEL = 'Klimaneutral bis 2045 (Demo)'
DEMO_TARGET_PATH = ((2030, -0.42), (2040, -0.8), (2045, -1.0))
"""
Reductions against the first inventory year, shaped like the federal Klimaschutzgesetz
(-65 % by 2030 and -88 % by 2040 against 1990, net neutrality in 2045) restated from a
2022 starting point. Illustrative: no municipality has adopted these numbers.
"""


def seed_demo_target(instance: InstanceConfig) -> str | None:
    """
    Give a demo municipality the demo target for net emissions, unless it already has a target.

    Returns a line describing what was set, or None if nothing was.
    """
    if instance.is_locked:
        return None
    graph = build_instance_graph(build_instance_snapshot(instance))
    node = next((node for node in graph.nodes if node.identifier == TARGET_NODE), None)
    if node is None or node.spec.goals.root:
        return None
    reference_year = min(declared_inventory_years(instance).years)
    last_year = graph.spec.years.model_end or graph.spec.years.target
    points = [GoalPoint(year=year, value=value) for year, value in DEMO_TARGET_PATH if last_year is None or year <= last_year]
    if not points:
        return None
    goal = build_goal(instance, graph, points, reference_year=reference_year, label=DEMO_TARGET_LABEL)
    with change_operation(instance, user=None, action='node.goal.set', source=InstanceChangeSource.CLI):
        set_node_goal(instance, graph, node.id, goal)
    return f'{instance.identifier}: demo target on {TARGET_NODE} against {reference_year}'


def seed_demo_targets(framework: Framework) -> list[str]:
    """Seed the demo target on the named demo municipalities that have an instance; return what was set."""
    configs = FrameworkConfig.objects.filter(
        framework=framework,
        instance_config__organization__identifiers__namespace__identifier='ars',
        instance_config__organization__identifiers__identifier__in=DEMO_MUNICIPALITIES,
    ).select_related('instance_config')
    lines = (seed_demo_target(config.instance_config) for config in configs.order_by('instance_config__identifier'))
    return [line for line in lines if line is not None]
