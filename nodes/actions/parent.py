import pandas as pd
import polars as pl
import networkx as nx

from common import polars as ppl
from nodes.constants import IMPACT_COLUMN, IMPACT_GROUP
from nodes.exceptions import NodeError
from params import Parameter
from .action import ActionNode
from ..node import Node


def first_common_descendant(G, sources, target):
    # Thank you, ChatGPT

    common_nodes = None

    if not len(sources):
        return None

    # Find all shortest paths from each source to the target
    for source in sources:
        paths = nx.all_shortest_paths(G, source, target)
        nodes_in_paths = set(node for path in paths for node in path)

        # Intersect with common_nodes to keep only common ones
        if common_nodes is None:
            common_nodes = nodes_in_paths
        else:
            common_nodes.intersection_update(nodes_in_paths)

    assert common_nodes is not None
    # Exclude the source and target nodes
    common_nodes.difference_update(sources + [target])

    # Return the closest common node to the target
    for node in nx.shortest_path(G, source=sources[0], target=target)[1:]:
        if node in common_nodes:
            return node

    return None


class ParentActionNode(ActionNode):
    subactions: list[ActionNode]

    def __post_init__(self):
        self.subactions = []
        super().__post_init__()

    def add_subaction(self, action: ActionNode):
        assert action not in self.subactions
        self.subactions.append(action)

    def find_first_common_descendant(self, target_node: Node):
        pass

    def notify_parameter_change(self, param: Parameter):
        if param == self.enabled_param:
            for action in self.subactions:
                action.enabled_param.set(param.get())
        return super().notify_parameter_change(param)

    def compute_effect(self) -> ppl.PathsDataFrame:
        for root_node in self.context.get_root_nodes():
            if root_node.unit.is_compatible_with(self.unit):
                break
        else:
            raise NodeError(self, "Unable to find unit-compatible root node")
        # g = self.context.node_graph
        # acts = list(filter(lambda x: root_node.id in nx.descendants(g, x.id), self.subactions))
        df = (
            self.compute_impact(root_node)
                .filter(pl.col(IMPACT_COLUMN) == IMPACT_GROUP).drop(IMPACT_COLUMN)
        )
        return df
