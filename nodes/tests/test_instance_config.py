from __future__ import annotations

import pytest

from frameworks.models import Framework, FrameworkConfig
from nodes.models import InstanceConfig, _pytest_instances
from nodes.tests.factories import InstanceFactory, NodeConfigFactory, SimpleNodeFactory
from orgs.tests.factories import OrganizationFactory

pytestmark = pytest.mark.django_db


def test_framework_backed_yaml_instance_resolves_outcome_node_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = InstanceFactory.create(id='framework-city', name='Framework City')
    SimpleNodeFactory.create(context=instance.context, id='net_emissions', is_outcome=True)
    _pytest_instances.pop(instance.id, None)

    ic = InstanceConfig.objects.create(
        identifier=instance.id,
        name='Framework City',
        primary_language='en',
        organization=OrganizationFactory.create(),
    )
    FrameworkConfig.objects.create(
        framework=Framework.objects.create(identifier='framework-test', name='Framework Test'),
        instance_config=ic,
        organization_name='Framework City',
        baseline_year=2020,
    )
    node_config = NodeConfigFactory.create(instance=ic, identifier='net_emissions')

    def create_model_instance(self: FrameworkConfig, _ic: InstanceConfig):
        return instance

    monkeypatch.setattr(FrameworkConfig, 'create_model_instance', create_model_instance)

    assert [node.identifier for node in ic.get_outcome_nodes()] == ['net_emissions']
    assert instance.context.nodes['net_emissions'].database_id == node_config.pk
