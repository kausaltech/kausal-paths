from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast
from uuid import NAMESPACE_URL, uuid5

from factory import Factory, LazyAttribute, LazyFunction, RelatedFactory, SelfAttribute, Sequence, SubFactory, post_generation
from factory.django import DjangoModelFactory

from kausal_common.i18n.pydantic import TranslatedString

from nodes.actions.action import ActionNode
from nodes.actions.simple import AdditiveAction
from nodes.context import Context
from nodes.datasets import FixedDataset
from nodes.defs.node_defs import NodeKind, NodeSpec, SimpleConfig
from nodes.defs.port_def import OutputPortDef
from nodes.instance import Instance
from nodes.models import InstanceConfig, NodeConfig, make_empty_instance_spec
from nodes.node import Node
from nodes.scenario import CustomScenario, Scenario, ScenarioKind
from nodes.simple import SimpleNode
from nodes.units import unit_registry
from orgs.models import Organization
from orgs.tests.factories import OrganizationFactory

if TYPE_CHECKING:
    from uuid import UUID

    from factory.builder import Resolver

    from nodes.defs.instance_defs import DatasetRepoSpec


def _port_id(label: str) -> UUID:
    return uuid5(NAMESPACE_URL, f'kausal-paths:test-port:{label}')


class ContextFactory(Factory[Context]):
    class Meta:
        model = Context

    instance: SubFactory[Any, Instance] = SubFactory(
        'nodes.tests.factories.InstanceFactory',
    )
    dataset_repo_spec: DatasetRepoSpec | None = None  # TODO: Set appropriately when we have tests for datasets
    target_year = 2030

    @classmethod
    def create(cls, **kwargs: Any) -> Context:
        return super().create(**kwargs)

    @post_generation
    @staticmethod
    def post(obj: Context, create: bool, extracted, **kwargs) -> None:
        obj.instance.set_context(obj)
        default_scenario = ScenarioFactory.create(id='default', kind=ScenarioKind.DEFAULT, all_actions_enabled=True)
        obj.add_scenario(default_scenario)
        obj.activate_scenario(obj.get_default_scenario())


class InstanceFactory(Factory[Instance]):
    class Meta:
        model = Instance

    id = Sequence(lambda i: f'instance{i}')
    name = Sequence(lambda i: f'instance{i}')
    owner = 'owner'
    default_language = 'fi'
    context: RelatedFactory[Any, Context] = RelatedFactory(ContextFactory, 'instance')
    reference_year = 1990
    minimum_historical_year = 2010
    maximum_historical_year = 2018

    # pages: Optional[Dict[str, Page]] = None
    # content_refreshed_at: Optional[datetime] = field(init=False)

    @classmethod
    def create(cls, **kwargs: Any) -> Instance:
        ret = super().create(**kwargs)
        return ret

    @post_generation
    @staticmethod
    def post(obj: Instance, create: bool, extracted, **kwargs) -> None:
        obj.modified_at = datetime.now(UTC) + timedelta(hours=1)


class InstanceConfigFactory(DjangoModelFactory[InstanceConfig]):
    class Meta:
        model = InstanceConfig
        exclude = ('instance',)

    identifier = Sequence(lambda i: f'ic{i}')
    name = Sequence(lambda i: f'instanceconfig{i}')
    lead_title = 'lead title'
    lead_paragraph = 'Lead paragraph'
    config_source = 'yaml'
    spec = LazyFunction(make_empty_instance_spec)
    instance: SubFactory[str, Instance] = SubFactory(InstanceFactory, id=SelfAttribute('..identifier'))
    organization = SubFactory[Any, Organization](OrganizationFactory)

    @classmethod
    def create(cls, **kwargs: Any) -> InstanceConfig:
        instance = cast('Instance | None', kwargs.get('instance'))
        name = kwargs.pop('name', None)
        if not name:
            assert instance is not None
            name = instance.name

        obj: InstanceConfig = super().create(name=name, **kwargs)
        if instance:
            from nodes.models import _pytest_instances

            # For tests we want to avoid reading a YAML file to configure the Instance
            _pytest_instances[instance.id] = instance

        return obj


class NodeConfigFactory(DjangoModelFactory[NodeConfig]):
    class Meta:
        model = NodeConfig

    instance: SubFactory[Any, InstanceConfig] = SubFactory(InstanceConfigFactory)
    identifier = Sequence(lambda i: f'nodeconfig{i}')
    name = Sequence(lambda i: f'Test node config {i}')
    short_description = 'short description'
    description = 'description'
    spec = NodeSpec(
        kind=NodeKind.SIMPLE,
        type_config=SimpleConfig(node_class='nodes.simple.SimpleNode'),
        output_ports=[OutputPortDef(id=_port_id('default'), unit=unit_registry.parse_units('kt/a'), quantity='emissions')],
    )


def make_fixed_datasets(node: Resolver[Node]):
    return [
        FixedDataset(
            id='test',
            context=node.context,
            unit=unit_registry.parse_units('kWh'),
            historical=[(2020, 1.23)],
            forecast=[(2021, 2.34)],
            tags=[],
        )
    ]


class NodeFactory(Factory[Node]):
    class Meta:
        model = Node

    id = Sequence(lambda i: f'node{i}')
    context: SubFactory[Any, Context] = SubFactory(ContextFactory)
    name = Sequence(lambda i: TranslatedString(f'Test node {i}', default_language='en'))
    description = TranslatedString('description', default_language='en')
    color = 'pink'
    unit = unit_registry.parse_units('kWh')
    quantity = 'energy'
    goals = [dict(values=[dict(year=2035, value=500)])]
    input_datasets = LazyAttribute(make_fixed_datasets)

    @post_generation
    @staticmethod
    def post(obj: Node, create: bool, extracted, **kwargs) -> None:
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


class ScenarioFactory[S: Scenario = Scenario](Factory[S]):
    class Meta:
        model = Scenario

    id = Sequence(lambda i: f'scenario{i}')
    name = TranslatedString('scenario', default_language='en')
    kind: ScenarioKind | None = None
    all_actions_enabled = False


class CustomScenarioFactory(ScenarioFactory[CustomScenario]):
    class Meta:
        model = CustomScenario

    base_scenario = SubFactory[Any, Scenario](ScenarioFactory)
