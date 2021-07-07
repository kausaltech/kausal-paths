from factory import Factory, Sequence, SubFactory
from typing import List

from common.i18n import TranslatedString
from nodes.actions import ActionNode
from nodes.context import Context
from nodes.instance import Instance
from nodes.node import Node
from nodes.scenario import CustomScenario, Scenario


class ContextFactory(Factory):
    class Meta:
        model = Context


class InstanceFactory(Factory):
    class Meta:
        model = Instance

    id = 'test'
    name = 'test'
    context = SubFactory(ContextFactory)


class NodeFactory(Factory):
    class Meta:
        model = Node

    id = Sequence(lambda i: f'node{i}')


class ActionNodeFactory(NodeFactory):
    class Meta:
        model = ActionNode


class ScenarioFactory(Factory):
    class Meta:
        model = Scenario

    id = Sequence(lambda i: f'scenario{i}')
    name = TranslatedString('scenario')
    default = False
    all_actions_enabled = False
    nodes: List[Node] = []


class CustomScenarioFactory(ScenarioFactory):
    class Meta:
        model = CustomScenario

    base_scenario = SubFactory(ScenarioFactory)
