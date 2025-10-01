from __future__ import annotations

from django.contrib.contenttypes.models import ContentType

import pytest

from kausal_common.datasets.models import Dataset, DatasetSchema, DatasetSchemaScope

from nodes.models import InstanceConfig
from nodes.roles import instance_admin_role, instance_super_admin_role
from nodes.tests.factories import InstanceConfigFactory
from paths.context import RealmContext, realm_context
from users.models import User


@pytest.fixture
def dataset_test_data():
    import uuid
    instance1 = InstanceConfigFactory.create(
        identifier=f'instance1-{uuid.uuid4().hex[:8]}',
        name='Instance 1',
    )
    instance2 = InstanceConfigFactory.create(
        identifier=f'instance2-{uuid.uuid4().hex[:8]}',
        name='Instance 2',
    )

    superuser = User.objects.create_superuser(
        email='super@example.com',
        password='password',
    )

    admin_user = User.objects.create_user(
        username='admin',
        email='admin@example.com',
        password='password',
    )
    instance_admin_role.assign_user(instance1, admin_user)

    super_admin_user = User.objects.create_user(
        username='super_admin',
        email='super_admin@example.com',
        password='password',
    )
    instance_super_admin_role.assign_user(instance1, super_admin_user)

    regular_user = User.objects.create_user(
        username='regular',
        email='regular@example.com',
        password='password',
    )

    schema1 = DatasetSchema.objects.create(
        name='Schema 1',
    )
    schema2 = DatasetSchema.objects.create(
        name='Schema 2',
    )
    unused_schema = DatasetSchema.objects.create(
        name='Unused schema'
    )

    content_type = ContentType.objects.get_for_model(InstanceConfig)
    DatasetSchemaScope.objects.create(
        schema=schema1,
        scope_content_type=content_type,
        scope_id=instance1.pk,
    )
    DatasetSchemaScope.objects.create(
        schema=unused_schema,
        scope_content_type=content_type,
        scope_id=instance1.pk,
    )
    DatasetSchemaScope.objects.create(
        schema=schema2,
        scope_content_type=content_type,
        scope_id=instance2.pk,
    )

    dataset1 = Dataset.objects.create(
        schema=schema1,
        scope_content_type=content_type,
        scope_id=instance1.pk,
    )
    dataset2 = Dataset.objects.create(
        schema=schema2,
        scope_content_type=content_type,
        scope_id=instance2.pk,
    )

    return {
        'instance1': instance1,
        'instance2': instance2,
        'superuser': superuser,
        'admin_user': admin_user,
        'super_admin_user': super_admin_user,
        'regular_user': regular_user,
        'schema1': schema1,
        'schema2': schema2,
        'dataset1': dataset1,
        'dataset2': dataset2,
        'unused_schema': unused_schema,
    }


@pytest.fixture
def get_in_admin_context(rf):
    def get(user: User, view, url: str, instance_config: InstanceConfig, kwargs: dict | None = None):
        if kwargs is None:
            kwargs = {}
        request = rf.get(url)
        request.user = user
        ctx = RealmContext(realm = instance_config, user=request.user)
        with realm_context.activate(ctx):
            return view(request, **kwargs)
    return get
