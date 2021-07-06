from factory import Factory, PostGeneration, Sequence, SubFactory

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

    context = SubFactory(ContextFactory)
    id = Sequence(lambda i: f'node{i}')

    add_to_context = PostGeneration(lambda obj, create, extracted, **kwargs: obj.context.add_node(obj))


class ActionNodeFactory(NodeFactory):
    class Meta:
        model = ActionNode


class ScenarioFactory(Factory):
    class Meta:
        model = Scenario

    id = Sequence(lambda i: f'scenario{i}')
    name = TranslatedString('scenario')
    context = SubFactory(ContextFactory)
    default = False
    all_actions_enabled = False
    nodes = []

    add_to_context = PostGeneration(lambda obj, create, extracted, **kwargs: obj.context.add_scenario(obj))


class CustomScenarioFactory(ScenarioFactory):
    class Meta:
        model = CustomScenario

    base_scenario = SubFactory(ScenarioFactory)
