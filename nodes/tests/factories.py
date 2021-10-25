from factory import Factory, Sequence, SubFactory
from factory.django import DjangoModelFactory
from typing import List

from common.i18n import TranslatedString
from nodes.actions import ActionNode
from nodes.actions.simple import AdditiveAction
from nodes.context import Context, unit_registry
from nodes.datasets import FixedDataset
from nodes.instance import Instance
from nodes.models import NodeConfig, InstanceConfig
from nodes.node import Node
from nodes.simple import SimpleNode
from nodes.scenario import CustomScenario, Scenario


class ContextFactory(Factory):
    class Meta:
        model = Context

    dataset_repo = None  # TODO: Set appropriately when we have tests for datasets
    target_year = 2030


class InstanceConfigFactory(DjangoModelFactory):
    class Meta:
        model = InstanceConfig

    identifier = 'test'
    lead_title = "lead title"
    lead_paragraph = "Lead paragraph"


class InstanceFactory(Factory):
    class Meta:
        model = Instance

    id = 'test'
    name = 'instance'
    owner = 'owner'
    default_language = 'fi'
    context = SubFactory(ContextFactory)
    reference_year = 1990
    minimum_historical_year = 2010
    maximum_historical_year = 2018

    # pages: Optional[Dict[str, Page]] = None
    # content_refreshed_at: Optional[datetime] = field(init=False)


class NodeConfigFactory(DjangoModelFactory):
    class Meta:
        model = NodeConfig

    instance = SubFactory(InstanceConfigFactory)
    identifier = Sequence(lambda i: f'nodeconfig{i}')
    name = "name"
    short_description = "short description"
    body = "body"


class NodeFactory(Factory):
    class Meta:
        model = Node

    id = Sequence(lambda i: f'node{i}')
    name = TranslatedString('name')
    description = TranslatedString('description')
    color = 'pink'
    unit = unit_registry('kWh').units
    quantity = 'energy'
    target_year_goal = 500.0
    input_datasets = [FixedDataset(
        id='test',
        unit='kWh',
        historical=[(2020, 1.23)],
        forecast=[(2021, 2.34)],
    )]


class ActionNodeFactory(NodeFactory):
    class Meta:
        model = ActionNode


class AdditiveActionFactory(ActionNodeFactory):
    class Meta:
        model = AdditiveAction


class SimpleNodeFactory(NodeFactory):
    class Meta:
        model = SimpleNode


class ScenarioFactory(Factory):
    class Meta:
        model = Scenario

    id = Sequence(lambda i: f'scenario{i}')
    name = TranslatedString('scenario')
    default = False
    all_actions_enabled = False
    notified_nodes: List[Node] = []


class CustomScenarioFactory(ScenarioFactory):
    class Meta:
        model = CustomScenario

    base_scenario = SubFactory(ScenarioFactory)
