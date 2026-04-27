from typing import TYPE_CHECKING, Any

from factory import LazyAttribute, Sequence, SubFactory
from factory.django import DjangoModelFactory

from frameworks.models import Framework, FrameworkConfig
from nodes.tests.factories import InstanceConfigFactory

if TYPE_CHECKING:
    from nodes.models import InstanceConfig


class FrameworkFactory(DjangoModelFactory[Framework]):
    identifier = Sequence(lambda i: f'framework{i}')
    name: LazyAttribute[Framework, str] = LazyAttribute(lambda o: 'Framework %d' % o.identifier)
    public_base_fqdn: LazyAttribute[Framework, str] = LazyAttribute(lambda o: 'fw%d.example.com' % o.identifier)

    class Meta:
        model = Framework

    @classmethod
    def create(cls, **kwargs: Any) -> Framework:
        return super().create(**kwargs)


class FrameworkConfigFactory(DjangoModelFactory[FrameworkConfig]):
    framework: SubFactory[FrameworkFactory, Framework] = SubFactory(FrameworkFactory)
    instance_config: SubFactory[InstanceConfigFactory, InstanceConfig] = SubFactory(InstanceConfigFactory)
