from django.contrib.contenttypes.models import ContentType

import pytest

from kausal_common.datasets.models import DatasetSchemaScope
from kausal_common.datasets.tests.factories import DataPointFactory, DatasetFactory, DatasetSchemaFactory

from nodes.models import NodeConfig
from nodes.roles import instance_admin_role
from nodes.tests.factories import InstanceConfigFactory, NodeConfigFactory
from users.tests.factories import UserFactory

pytestmark = pytest.mark.django_db


def test_protected_node_is_read_only_for_instance_admin_but_not_superuser() -> None:
    instance = InstanceConfigFactory.create(name='Protected node instance', config_source='database')
    node = NodeConfigFactory.create(instance=instance, is_editable=False)
    admin = UserFactory.create()
    superuser = UserFactory.create(is_superuser=True)
    instance_admin_role.assign_user(instance, admin)

    policy = NodeConfig.permission_policy()
    assert policy.user_has_permission_for_instance(admin, 'view', node)
    assert not policy.user_has_permission_for_instance(admin, 'change', node)
    assert not policy.user_has_permission_for_instance(admin, 'delete', node)
    assert not policy.instances_user_has_permission_for(admin, 'change').filter(pk=node.pk).exists()

    assert policy.user_has_permission_for_instance(superuser, 'change', node)
    assert policy.user_has_permission_for_instance(superuser, 'delete', node)
    assert policy.instances_user_has_permission_for(superuser, 'change').filter(pk=node.pk).exists()


def test_protected_schema_makes_dataset_and_data_points_read_only_except_for_superuser() -> None:
    instance = InstanceConfigFactory.create(name='Protected dataset instance', config_source='database')
    schema = DatasetSchemaFactory.create(is_editable=False)
    DatasetSchemaScope.objects.create(
        schema=schema,
        scope_content_type=ContentType.objects.get_for_model(instance),
        scope_id=instance.pk,
    )
    dataset = DatasetFactory.create(schema=schema, scope=instance)
    data_point = DataPointFactory.create(dataset=dataset, metric__schema=schema)
    admin = UserFactory.create()
    superuser = UserFactory.create(is_superuser=True)
    instance_admin_role.assign_user(instance, admin)

    schema_policy = schema.permission_policy()
    assert schema_policy.user_has_permission_for_instance(admin, 'view', schema)
    assert not schema_policy.user_has_permission_for_instance(admin, 'change', schema)
    assert not schema_policy.user_has_permission_for_instance(admin, 'delete', schema)
    assert schema_policy.user_has_permission_for_instance(superuser, 'change', schema)
    assert schema_policy.user_has_permission_for_instance(superuser, 'delete', schema)

    dataset_policy = dataset.permission_policy()
    assert dataset_policy.user_has_permission_for_instance(admin, 'view', dataset)
    assert not dataset_policy.user_has_permission_for_instance(admin, 'change', dataset)
    assert not dataset_policy.user_has_permission_for_instance(admin, 'delete', dataset)
    assert dataset_policy.user_has_permission_for_instance(superuser, 'change', dataset)
    assert dataset_policy.user_has_permission_for_instance(superuser, 'delete', dataset)

    data_point_policy = data_point.permission_policy()
    assert data_point_policy.user_has_permission_for_instance(admin, 'view', data_point)
    assert not data_point_policy.user_has_permission_for_instance(admin, 'change', data_point)
    assert not data_point_policy.user_has_permission_for_instance(admin, 'delete', data_point)
    assert data_point_policy.user_has_permission_for_instance(superuser, 'change', data_point)
    assert data_point_policy.user_has_permission_for_instance(superuser, 'delete', data_point)
