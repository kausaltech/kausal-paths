"""
Datasets and schemas owned by a node: scope, permissions and deletion.

A node-owned dataset is scoped to its `NodeConfig`; the node's instance governs it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from django.contrib.contenttypes.models import ContentType

import pytest

from kausal_common.datasets.models import (
    Dataset,
    DatasetSchema,
    DatasetSchemaScope,
    DatasetSourceReference,
    DataSource,
)
from kausal_common.datasets.tests.factories import DataPointFactory, DatasetFactory, DatasetMetricFactory
from kausal_common.people.models import ObjectRole

from paths.dataset_permission_policy import DataPointCommentPermissionPolicy, DatasetSourceReferencePermissionPolicy

from nodes.models import InstanceConfig, NodeConfig
from nodes.roles import instance_admin_role, instance_reviewer_role, instance_viewer_role
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory
from people.models import DatasetPersonPermission
from people.tests.factories import PersonFactory
from users.tests.factories import UserFactory

if TYPE_CHECKING:
    from users.models import User

pytestmark = pytest.mark.django_db


def _owned_dataset(node: NodeConfig) -> Dataset:
    """Create a dataset and schema scoped to `node`, with one data point."""
    node_ct = ContentType.objects.get_for_model(NodeConfig)
    schema = DatasetSchema.objects.create(name=f'{node.identifier} data')
    DatasetSchemaScope.objects.create(schema=schema, scope_content_type=node_ct, scope_id=node.pk)
    dataset = DatasetFactory.create(schema=schema, scope=node)
    metric = DatasetMetricFactory.create(schema=schema, name='value', unit='kt/a')
    DataPointFactory.create(dataset=dataset, metric=metric)
    return dataset


def _instance() -> InstanceConfig:
    return InstanceConfigFactory.create(instance=InstanceFactory.create())


def _user_with_role(role, ic: InstanceConfig) -> User:
    user = UserFactory.create()
    role.assign_user(ic, user)
    return user


@pytest.fixture
def ic() -> InstanceConfig:
    return _instance()


@pytest.fixture
def node(ic: InstanceConfig) -> NodeConfig:
    return NodeConfigFactory.create(instance=ic, identifier='owner')


@pytest.fixture
def owned(node: NodeConfig) -> Dataset:
    return _owned_dataset(node)


def test_the_node_owns_the_dataset_and_its_instance_governs_it(ic: InstanceConfig, node: NodeConfig, owned: Dataset):
    assert owned.scope_node == node
    assert owned.scope_instance == ic
    instance_dataset = DatasetFactory.create(scope=ic)
    assert instance_dataset.scope_node is None
    assert instance_dataset.scope_instance == ic


def test_node_owned_datasets_are_listed_only_through_their_node(ic: InstanceConfig, node: NodeConfig, owned: Dataset):
    instance_dataset = DatasetFactory.create(scope=ic)
    foreign = _owned_dataset(NodeConfigFactory.create(instance=_instance()))

    assert set(Dataset.objects.qs.for_instance_config(ic)) == {instance_dataset}
    assert set(Dataset.objects.qs.for_node(node)) == {owned}
    assert set(Dataset.objects.qs.governed_by_instance(ic)) == {instance_dataset, owned}
    assert foreign not in set(Dataset.objects.qs.governed_by_instance(ic))


@pytest.mark.parametrize(
    ('role', 'can_view', 'can_change'),
    [
        (instance_admin_role, True, True),
        (instance_viewer_role, True, False),
        (instance_reviewer_role, True, False),
    ],
)
def test_instance_roles_reach_node_owned_datasets_and_schemas(ic: InstanceConfig, owned: Dataset, role, can_view, can_change):
    user = _user_with_role(role, ic)
    stranger = _user_with_role(role, _instance())
    schema = owned.schema
    assert schema is not None
    dataset_policy = Dataset.permission_policy()
    schema_policy = DatasetSchema.permission_policy()

    assert dataset_policy.user_has_perm(user, 'view', owned) is can_view
    assert dataset_policy.user_has_perm(user, 'change', owned) is can_change
    assert (owned in Dataset.objects.qs.viewable_by(user)) is can_view
    assert (owned in Dataset.objects.qs.modifiable_by(user)) is can_change
    assert (schema in DatasetSchema.objects.qs.viewable_by(user)) is can_view
    assert (schema in DatasetSchema.objects.qs.modifiable_by(user)) is can_change
    assert set(schema_policy.get_instance_configs_for_obj(schema)) == {ic.pk}

    assert not dataset_policy.user_has_perm(stranger, 'view', owned)
    assert owned not in Dataset.objects.qs.viewable_by(stranger)
    assert schema not in DatasetSchema.objects.qs.viewable_by(stranger)


def test_a_locked_instance_blocks_changes_to_its_nodes_datasets(ic: InstanceConfig, owned: Dataset):
    admin = _user_with_role(instance_admin_role, ic)
    InstanceConfig.objects.filter(pk=ic.pk).update(is_locked=True)
    owned = Dataset.objects.get(pk=owned.pk)

    assert Dataset.permission_policy().get_permission_block('change', obj=owned) is not None
    assert not Dataset.permission_policy().user_has_perm(admin, 'change', owned)
    assert owned not in Dataset.objects.qs.modifiable_by(admin)
    assert owned.schema not in DatasetSchema.objects.qs.modifiable_by(admin)


def test_reviewers_can_comment_and_editors_can_cite_on_node_owned_data(ic: InstanceConfig, owned: Dataset):
    reviewer = _user_with_role(instance_reviewer_role, ic)
    admin = _user_with_role(instance_admin_role, ic)
    stranger = _user_with_role(instance_admin_role, _instance())
    data_point = owned.data_points.get()

    assert DataPointCommentPermissionPolicy().user_can_create(reviewer, data_point)
    assert DatasetSourceReferencePermissionPolicy().user_can_create(admin, owned)
    assert not DatasetSourceReferencePermissionPolicy().user_can_create(stranger, owned)


def test_deleting_the_node_deletes_its_datasets_and_schemas(ic: InstanceConfig, node: NodeConfig, owned: Dataset):
    schema = owned.schema
    assert schema is not None
    kept = _owned_dataset(NodeConfigFactory.create(instance=ic, identifier='bystander'))

    node.delete()

    assert not Dataset.objects.filter(pk=owned.pk).exists()
    assert not DatasetSchema.objects.filter(pk=schema.pk).exists()
    assert Dataset.objects.filter(pk=kept.pk).exists()


def test_deleting_the_instance_deletes_its_nodes_datasets(ic: InstanceConfig, owned: Dataset):
    schema = owned.schema
    assert schema is not None
    source = DataSource.objects.create(name='Source', scope=ic)
    DatasetSourceReference.objects.create(dataset=owned, data_source=source)

    ic.delete()

    assert not Dataset.objects.filter(pk=owned.pk).exists()
    assert not DatasetSchema.objects.filter(pk=schema.pk).exists()


def test_a_locked_node_locks_its_data_but_not_the_instances(ic: InstanceConfig, node: NodeConfig, owned: Dataset):
    admin = _user_with_role(instance_admin_role, ic)
    instance_dataset = DatasetFactory.create(scope=ic)
    NodeConfig.objects.filter(pk=node.pk).update(is_editable=False)
    owned = Dataset.objects.get(pk=owned.pk)

    policy = Dataset.permission_policy()
    assert policy.user_has_perm(admin, 'view', owned)
    assert not policy.user_has_perm(admin, 'change', owned)
    assert policy.user_has_perm(admin, 'change', instance_dataset)


def test_a_public_instance_does_not_make_its_data_public(ic: InstanceConfig, owned: Dataset):
    assert not ic.has_framework_config()
    somebody = UserFactory.create()
    assert InstanceConfig.permission_policy().user_has_perm(somebody, 'view', ic)

    instance_dataset = DatasetFactory.create(scope=ic)
    assert not Dataset.permission_policy().user_has_perm(somebody, 'view', instance_dataset)
    assert not Dataset.permission_policy().user_has_perm(somebody, 'view', owned)
    assert not Dataset.objects.qs.viewable_by(somebody).filter(pk__in=[owned.pk, instance_dataset.pk]).exists()


@pytest.mark.parametrize(
    ('role', 'can_change'),
    [(ObjectRole.VIEWER, False), (ObjectRole.EDITOR, True), (ObjectRole.ADMIN, True)],
)
def test_a_dataset_grant_reaches_the_data_but_not_the_schema(ic: InstanceConfig, owned: Dataset, role, can_change):
    person = PersonFactory.create()
    DatasetPersonPermission.objects.create(object=owned, person=person, role=role)
    user = person.user
    assert user is not None
    other = _owned_dataset(NodeConfigFactory.create(instance=ic, identifier='other'))
    schema = owned.schema
    assert schema is not None

    policy = Dataset.permission_policy()
    assert policy.user_has_perm(user, 'view', owned)
    assert policy.user_has_perm(user, 'change', owned) is can_change
    assert not policy.user_has_perm(user, 'view', other)
    assert set(Dataset.objects.qs.viewable_by(user)) == {owned}
    # The grantee can read the schema's structure, but not change it.
    assert DatasetSchema.permission_policy().user_has_perm(user, 'view', schema)
    assert not DatasetSchema.permission_policy().user_has_perm(user, 'change', schema)
    assert schema not in DatasetSchema.objects.qs.modifiable_by(user)
    # The grant makes the instance reachable, so the grantee can navigate to the data.
    assert ic in InstanceConfig.permission_policy().adminable_instances(user)
