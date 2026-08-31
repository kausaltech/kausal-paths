from typing import TYPE_CHECKING, Any

from factory import LazyAttribute, Sequence, SubFactory
from factory.django import DjangoModelFactory

from frameworks.models import Framework, FrameworkConfig
from nodes.tests.factories import InstanceConfigFactory

if TYPE_CHECKING:
    from nodes.models import InstanceConfig


class FrameworkFactory(DjangoModelFactory[Framework]):
    identifier = Sequence(lambda i: f'framework{i}')
    name: LazyAttribute[Framework, str] = LazyAttribute(lambda o: f'Framework {o.identifier}')
    public_base_fqdn: LazyAttribute[Framework, str] = LazyAttribute(lambda o: f'{o.identifier}.example.com')

    class Meta:
        model = Framework

    @classmethod
    def create(cls, **kwargs: Any) -> Framework:
        return super().create(**kwargs)


class FrameworkConfigFactory(DjangoModelFactory[FrameworkConfig]):
    framework: SubFactory[FrameworkFactory, Framework] = SubFactory(FrameworkFactory)
    instance_config: SubFactory[InstanceConfigFactory, InstanceConfig] = SubFactory(InstanceConfigFactory)

    class Meta:
        model = FrameworkConfig

    @classmethod
    def create(cls, **kwargs: Any) -> FrameworkConfig:
        baseline_year = kwargs.pop('baseline_year', None)
        target_year = kwargs.pop('target_year', None)
        fwc = super().create(**kwargs)
        updates: dict[str, int] = {}
        if baseline_year is not None:
            updates['reference'] = baseline_year
        elif fwc.instance_config.ensure_spec().years.reference is None:
            updates['reference'] = 2020
        if target_year is not None:
            updates['target'] = target_year
        if updates:
            fwc.instance_config.update_years(**updates)
        return fwc
