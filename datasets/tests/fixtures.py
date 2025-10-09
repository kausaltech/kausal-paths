from __future__ import annotations

from datetime import date

from django.contrib.auth.models import Group
from django.contrib.contenttypes.models import ContentType

import pytest

from kausal_common.datasets.models import (
    DataPoint,
    DataPointComment,
    Dataset,
    DatasetMetric,
    DatasetSchema,
    DatasetSchemaDimension,
    DatasetSchemaScope,
    DatasetSourceReference,
    DataSource,
    Dimension,
    DimensionCategory,
    DimensionScope,
)

from paths.context import RealmContext, realm_context

from nodes.models import InstanceConfig
from nodes.roles import instance_admin_role, instance_reviewer_role, instance_super_admin_role, instance_viewer_role
from nodes.tests.factories import InstanceConfigFactory
from users.models import User


@pytest.fixture(scope='module')
def dataset_test_data(django_db_setup, django_db_blocker):
  import uuid
  with django_db_blocker.unblock():
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

    reviewer_user = User.objects.create_user(
        username='reviewer',
        email='reviewer@example.com',
        password='password',
    )
    instance_reviewer_role.assign_user(instance1, reviewer_user)

    viewer_user = User.objects.create_user(
        username='viewer',
        email='viewer@example.com',
        password='password',
    )
    instance_viewer_role.assign_user(instance1, viewer_user)

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

    metric1 = DatasetMetric.objects.create(
        schema=schema1,
        label='Metric 1',
        unit='kg',
        order=0,
    )
    metric2 = DatasetMetric.objects.create(
        schema=schema2,
        label='Metric 2',
        unit='liters',
        order=0,
    )

    dimension1 = Dimension.objects.create(
        name='Test Dimension',
    )
    DimensionScope.objects.create(
        dimension=dimension1,
        scope_content_type=content_type,
        scope_id=instance1.pk,
    )

    dimension_category1 = DimensionCategory.objects.create(
        dimension=dimension1,
        label='Category A',
        order=0,
    )

    DatasetSchemaDimension.objects.create(dimension=dimension1, schema=schema1)

    data_point1 = DataPoint.objects.create(
        dataset=dataset1,
        date=date(2023, 1, 1),
        metric=metric1,
        value=100.50,
        created_by=admin_user,
        last_modified_by=admin_user,
    )
    data_point1.dimension_categories.add(dimension_category1)

    data_point2 = DataPoint.objects.create(
        dataset=dataset2,
        date=date(2023, 1, 1),
        metric=metric2,
        value=200.75,
        created_by=superuser,
        last_modified_by=superuser,
    )

    data_source1 = DataSource.objects.create(
        scope_content_type=content_type,
        scope_id=instance1.pk,
        name='Test Data Source 1',
        authority='Authority 1',
    )
    data_source2 = DataSource.objects.create(
        scope_content_type=content_type,
        scope_id=instance2.pk,
        name='Test Data Source 2',
        authority='Authority 2',
    )

    comment1 = DataPointComment.objects.create(
        data_point=data_point1,
        text='Test comment',
        type=DataPointComment.CommentType.PLAIN,
        created_by=admin_user,
        last_modified_by=admin_user,
    )

    source_ref1 = DatasetSourceReference.objects.create(
        dataset=dataset1,
        data_source=data_source1,
    )

    source_ref_on_datapoint = DatasetSourceReference.objects.create(
        data_point=data_point1,
        data_source=data_source1,
    )

    source_ref2 = DatasetSourceReference.objects.create(
        dataset=dataset2,
        data_source=data_source2,
    )

    result = {
        'instance1': instance1,
        'instance2': instance2,
        'superuser': superuser,
        'admin_user': admin_user,
        'super_admin_user': super_admin_user,
        'reviewer_user': reviewer_user,
        'viewer_user': viewer_user,
        'regular_user': regular_user,
        'schema1': schema1,
        'schema2': schema2,
        'dataset1': dataset1,
        'dataset2': dataset2,
        'unused_schema': unused_schema,
        'metric1': metric1,
        'metric2': metric2,
        'dimension1': dimension1,
        'dimension_category1': dimension_category1,
        'data_point1': data_point1,
        'data_point2': data_point2,
        'data_source1': data_source1,
        'data_source2': data_source2,
        'comment1': comment1,
        'source_ref1': source_ref1,
        'source_ref_on_datapoint': source_ref_on_datapoint,
        'source_ref2': source_ref2,
    }
    yield result
    with django_db_blocker.unblock():
        to_delete = list(reversed(result.values())) + list(Group.objects.all())
        for o in to_delete:
            o.delete()

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
