from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from factory import Factory, RelatedFactory, SelfAttribute, Sequence, SubFactory, post_generation
from factory.django import DjangoModelFactory

from common.i18n import TranslatedString
from nodes.actions import ActionNode
from nodes.actions.simple import AdditiveAction
from nodes.context import Context, unit_registry
from nodes.datasets import FixedDataset
from nodes.instance import Instance
from nodes.models import InstanceConfig, NodeConfig
from nodes.node import Node
from nodes.scenario import CustomScenario, Scenario
from nodes.simple import SimpleNode


class ContextFactory(Factory[Context]):
    class Meta:
        model = Context

    instance: SubFactory[Any, Instance] = SubFactory(
        'nodes.tests.factories.InstanceFactory',
    )
    dataset_repo: str | None = None  # TODO: Set appropriately when we have tests for datasets
    target_year = 2030

    @classmethod
    def create(cls, **kwargs: Any) -> Context:  # noqa: ANN401
        return super().create(**kwargs)

    @post_generation
    @staticmethod
    def post(obj: Context, create: bool, extracted, **kwargs) -> None:
        obj.instance.set_context(obj)
        default_scenario = ScenarioFactory.create(id='default', context=obj, default=True, all_actions_enabled=True)
        obj.add_scenario(default_scenario)
        obj.activate_scenario(obj.get_default_scenario())


class InstanceFactory(Factory[Instance]):
    class Meta:
        model = Instance

    id = Sequence(lambda i: f'instance{i}')
    name = 'instance'
    owner = 'owner'
    default_language = 'fi'
    context: RelatedFactory[Any, Context] = RelatedFactory(ContextFactory, 'instance')
    reference_year = 1990
    minimum_historical_year = 2010
    maximum_historical_year = 2018

    # pages: Optional[Dict[str, Page]] = None
    # content_refreshed_at: Optional[datetime] = field(init=False)

    @classmethod
    def create(cls, **kwargs: Any) -> Instance:  # noqa: ANN401
        ret = super().create(**kwargs)
        return ret

    @post_generation
    @staticmethod
    def post(obj: Instance, create: bool, extracted, **kwargs):
        obj.modified_at = datetime.now(UTC) + timedelta(hours=1)

class InstanceConfigFactory(DjangoModelFactory[InstanceConfig]):
    class Meta:
        model = InstanceConfig
        exclude = ('instance',)

    identifier = Sequence(lambda i: f'ic{i}')
    lead_title = "lead title"
    lead_paragraph = "Lead paragraph"
    instance: SubFactory[str, Instance] = SubFactory(InstanceFactory, id=SelfAttribute('..identifier'))

    @classmethod
    def create(cls, **kwargs: Any) -> InstanceConfig:
        instance = kwargs.get('instance', None)
        obj: InstanceConfig = super().create(**kwargs)
        if instance:
            from nodes.models import _pytest_instances
            # For tests we want to avoid reading a YAML file to configure the Instance
            _pytest_instances[instance.id] = instance

        return obj

class NodeConfigFactory(DjangoModelFactory):
    class Meta:
        model = NodeConfig

    instance: SubFactory[Any, InstanceConfig] = SubFactory(InstanceConfigFactory)
    identifier = Sequence(lambda i: f'nodeconfig{i}')
    name = "name"
    short_description = "short description"
    description = "description"


class NodeFactory(Factory[Node]):
    class Meta:
        model = Node

    id = Sequence(lambda i: f'node{i}')
    context: SubFactory[Any, Context] = SubFactory(ContextFactory)
    name = TranslatedString('name')
    description = TranslatedString('description')
    color = 'pink'
    unit = unit_registry('kWh').units
    quantity = 'energy'
    goals = [dict(values=[dict(year=2035, value=500)])]
    input_datasets = [FixedDataset(
        id='test',
        unit='kWh',
        historical=[(2020, 1.23)],
        forecast=[(2021, 2.34)],
        tags=[],
    )]

    @post_generation
    @staticmethod
    def post(obj: Node, create: bool, extracted, **kwargs):
        assert obj.context.instance is not None
        obj.context.add_node(obj)
        obj.context.finalize_nodes()


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
    context = SubFactory(ContextFactory)


class CustomScenarioFactory(ScenarioFactory):
    class Meta:
        model = CustomScenario

    base_scenario = SubFactory(ScenarioFactory)
