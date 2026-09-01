"""
Tests for the ``destructively_trim_db`` management command.

These cover the subtle ownership/ordering logic the command relies on:
the two-tier dataset deletion (direct scope vs. schema-scoped placeholders),
orphan schema/dimension cleanup, and the object-existence criterion for
purging Wagtail revisions and log entries.
"""

from django.contrib.contenttypes.models import ContentType
from django.core.management import call_command
from django.utils import timezone
from wagtail.models import ModelLogEntry, Page, Revision as WagtailRevision

import pytest

from kausal_common.datasets.models import (
    DataPoint,
    Dataset,
    DatasetSchema,
    DatasetSchemaDimension,
    DatasetSchemaScope,
    DatasetSourceReference,
    DataSource,
    Dimension,
    DimensionCategory,
    DimensionScope,
)
from kausal_common.datasets.tests.factories import (
    DataPointFactory,
    DatasetFactory,
    DatasetMetricFactory,
    DatasetSchemaDimensionFactory,
    DatasetSchemaFactory,
    DimensionCategoryFactory,
    DimensionFactory,
)

from nodes.models import InstanceConfig
from nodes.tests.factories import InstanceConfigFactory, NodeConfigFactory
from orgs.models import Organization
from orgs.tests.factories import OrganizationFactory
from pages.models import OutcomePage
from people.tests.factories import PersonFactory
from users.models import User

pytestmark = pytest.mark.django_db

_ic_seq = iter(range(1, 1_000_000))


def _ic() -> InstanceConfig:
    """Create an InstanceConfig (the factory requires an explicit name or instance)."""
    n = next(_ic_seq)
    return InstanceConfigFactory.create(identifier=f'trim-ic-{n}', name=f'Trim IC {n}')


def _ct(obj) -> ContentType:
    return ContentType.objects.get_for_model(type(obj))


def _scope(obj) -> dict[str, object]:
    """Generic-FK ``scope`` kwargs pointing at ``obj`` (an InstanceConfig/Organization)."""
    return {'scope_content_type': _ct(obj), 'scope_id': obj.pk}


@pytest.fixture
def run_trim(settings):
    """
    Return a callable that runs the command, keeping the given instances.

    The command refuses to run unless DEBUG is on and the deployment is not
    production, so flip those for the test.
    """
    settings.DEBUG = True
    settings.DEPLOYMENT_TYPE = 'testing'

    def run(keep, *, thorough: bool = False) -> None:
        call_command(
            'destructively_trim_db',
            no_confirm=True,
            thorough=thorough,
            exclude_instance=[ic.identifier for ic in keep],
        )

    return run


def test_direct_dataset_deleted_even_when_schema_shared(run_trim):
    """
    Delete a directly-scoped dataset even when its schema is shared with a kept instance.

    A dataset directly scoped to a deleted instance holds that customer's data, so it must be
    deleted even when its schema is also available to a retained instance.
    """
    deleted = _ic()
    kept = _ic()

    schema = DatasetSchemaFactory.create()
    # The same schema is also made available to the kept instance.
    DatasetSchemaScope.objects.create(schema=schema, **_scope(kept))

    dataset = DatasetFactory.create(schema=schema, **_scope(deleted))
    metric = DatasetMetricFactory.create(schema=schema)
    DataPointFactory.create(dataset=dataset, metric=metric)

    run_trim([kept])

    # The directly-scoped dataset and its data points are gone...
    assert not Dataset.objects.filter(pk=dataset.pk).exists()
    assert not DataPoint.objects.filter(dataset_id=dataset.pk).exists()
    # ...but the shared schema and the kept instance survive.
    assert DatasetSchema.objects.filter(pk=schema.pk).exists()
    assert InstanceConfig.objects.filter(pk=kept.pk).exists()
    assert not InstanceConfig.objects.filter(pk=deleted.pk).exists()


def test_instance_delete_removes_directly_scoped_dataset_with_shared_schema():
    """
    InstanceConfig.delete() removes a directly-scoped dataset even when its schema is shared.

    This exercises InstanceConfig.delete() directly (not via the trim command, whose later cleanup
    would mask the issue): the instance owns the dataset directly, so deleting the instance must
    remove its data even though the schema is also scoped to another instance (which is preserved).
    """
    deleted = _ic()
    other = _ic()

    schema = DatasetSchemaFactory.create()
    # The schema is made available to BOTH instances.
    DatasetSchemaScope.objects.create(schema=schema, **_scope(deleted))
    DatasetSchemaScope.objects.create(schema=schema, **_scope(other))

    dataset = DatasetFactory.create(schema=schema, **_scope(deleted))
    metric = DatasetMetricFactory.create(schema=schema)
    DataPointFactory.create(dataset=dataset, metric=metric)

    deleted.delete()

    # The deleted instance's own (directly-scoped) dataset and its data points are removed...
    assert not Dataset.objects.filter(pk=dataset.pk).exists()
    assert not DataPoint.objects.filter(dataset_id=dataset.pk).exists()
    # ...while the shared schema and the other instance's scope link survive.
    assert DatasetSchema.objects.filter(pk=schema.pk).exists()
    assert DatasetSchemaScope.objects.filter(schema=schema, scope_id=other.pk).exists()
    assert not DatasetSchemaScope.objects.filter(schema=schema, scope_id=deleted.pk).exists()
    assert InstanceConfig.objects.filter(pk=other.pk).exists()


def test_schema_scoped_placeholder_and_orphan_schema_deleted(run_trim):
    """
    Delete schema-scoped placeholder datasets and the now-orphaned schema.

    A placeholder dataset owned only via a DatasetSchemaScope to a deleted instance, and the
    now-orphaned schema, are both removed.
    """
    deleted = _ic()
    kept = _ic()

    schema = DatasetSchemaFactory.create()
    DatasetSchemaScope.objects.create(schema=schema, **_scope(deleted))
    # Owned only via the schema scope (not directly scoped to anything).
    placeholder = DatasetFactory.create(schema=schema, is_external_placeholder=True)

    run_trim([kept])

    assert not Dataset.objects.filter(pk=placeholder.pk).exists()
    assert not DatasetSchema.objects.filter(pk=schema.pk).exists()
    assert not DatasetSchemaScope.objects.filter(schema_id=schema.pk).exists()


def test_shared_schema_scoped_placeholder_preserved(run_trim):
    """
    Preserve a placeholder whose schema is shared with a retained instance.

    A placeholder whose schema is shared between a deleted and a retained instance must remain;
    only the deleted instance's scope link is removed.
    """
    deleted = _ic()
    kept = _ic()

    schema = DatasetSchemaFactory.create()
    DatasetSchemaScope.objects.create(schema=schema, **_scope(deleted))
    DatasetSchemaScope.objects.create(schema=schema, **_scope(kept))
    placeholder = DatasetFactory.create(schema=schema, is_external_placeholder=True)

    run_trim([kept])

    assert Dataset.objects.filter(pk=placeholder.pk).exists()
    assert DatasetSchema.objects.filter(pk=schema.pk).exists()
    assert DatasetSchemaScope.objects.filter(schema_id=schema.pk, scope_id=kept.pk).exists()
    assert not DatasetSchemaScope.objects.filter(schema_id=schema.pk, scope_id=deleted.pk).exists()


def test_deleted_scope_data_source_referenced_by_kept_dataset_preserved(run_trim):
    """
    Preserve a deleted-scope data source while retained data still references it.

    DatasetSourceReference.data_source is PROTECT. After deleted datasets are removed, any
    remaining source reference belongs to retained data, so the source must not be deleted.
    """
    deleted = _ic()
    kept = _ic()

    source = DataSource.objects.create(name='Shared source', **_scope(deleted))
    kept_dataset = DatasetFactory.create(**_scope(kept))
    reference = DatasetSourceReference.objects.create(dataset=kept_dataset, data_source=source)

    run_trim([kept])

    assert DataSource.objects.filter(pk=source.pk).exists()
    assert DatasetSourceReference.objects.filter(pk=reference.pk, dataset=kept_dataset, data_source=source).exists()


def test_revisions_and_log_entries_for_deleted_objects_removed_despite_user(run_trim):
    """
    Purge revisions/log entries for deleted objects even when their author survives.

    Non-thorough mode purges revisions and log entries whose target object no longer exists,
    even when the authoring user survives the trim (and keeps those of retained objects).
    """
    kept = _ic()
    deleted = _ic()
    # A person (and thus its user) attached to the kept instance's org survives the trim.
    author = PersonFactory.create(organization=kept.organization).user
    assert author is not None

    deleted_ds = DatasetFactory.create(**_scope(deleted))
    kept_ds = DatasetFactory.create(**_scope(kept))

    deleted_ds.save_revision(user=author)
    kept_ds.save_revision(user=author)
    for ds in (deleted_ds, kept_ds):
        ModelLogEntry.objects.create(
            content_type=_ct(ds),
            label=str(ds),
            action='wagtail.edit',
            timestamp=timezone.now(),
            object_id=str(ds.pk),
            user=author,
            data={},
        )

    ds_ct = _ct(deleted_ds)
    # Precondition: the deleted dataset's revision is authored by a (non-null) user.
    assert WagtailRevision.objects.filter(content_type=ds_ct, object_id=str(deleted_ds.pk), user=author).exists()

    run_trim([kept])

    # The author survived...
    assert User.objects.filter(pk=author.pk).exists()
    # ...yet the deleted dataset's revision and log entry are gone...
    assert not WagtailRevision.objects.filter(content_type=ds_ct, object_id=str(deleted_ds.pk)).exists()
    assert not ModelLogEntry.objects.filter(content_type=ds_ct, object_id=str(deleted_ds.pk)).exists()
    # ...while the retained dataset keeps both.
    assert WagtailRevision.objects.filter(content_type=ds_ct, object_id=str(kept_ds.pk)).exists()
    assert ModelLogEntry.objects.filter(content_type=ds_ct, object_id=str(kept_ds.pk)).exists()


def test_orphaned_dimension_deleted(run_trim):
    """
    Delete a dimension scoped only to a deleted instance, along with its categories.

    A dimension scoped only to a deleted instance, with no other references, is removed along
    with its categories.
    """
    deleted = _ic()
    kept = _ic()

    dim = DimensionFactory.create()
    DimensionScope.objects.create(dimension=dim, **_scope(deleted))
    cat = DimensionCategoryFactory.create(dimension=dim)

    run_trim([kept])

    assert not Dimension.objects.filter(pk=dim.pk).exists()
    assert not DimensionCategory.objects.filter(pk=cat.pk).exists()


def test_dimension_referenced_by_kept_datapoint_preserved(run_trim):
    """
    Preserve a dimension whose category is still used by a retained data point.

    A dimension scoped only to a deleted instance must NOT be deleted while a retained data
    point still references one of its categories (DataPointDimensionCategory is PROTECT).
    """
    deleted = _ic()
    kept = _ic()

    dim = DimensionFactory.create()
    DimensionScope.objects.create(dimension=dim, **_scope(deleted))
    cat = DimensionCategoryFactory.create(dimension=dim)

    kept_schema = DatasetSchemaFactory.create()
    DatasetSchemaScope.objects.create(schema=kept_schema, **_scope(kept))
    kept_ds = DatasetFactory.create(schema=kept_schema, **_scope(kept))
    metric = DatasetMetricFactory.create(schema=kept_schema)
    DataPointFactory.create(dataset=kept_ds, metric=metric, dimension_categories=[cat])

    run_trim([kept])

    # Its scope link is gone, but the dimension and category survive because they are still in use.
    assert not DimensionScope.objects.filter(dimension=dim).exists()
    assert Dimension.objects.filter(pk=dim.pk).exists()
    assert DimensionCategory.objects.filter(pk=cat.pk).exists()


def test_dimension_linked_to_kept_schema_preserved(run_trim):
    """
    Preserve a dimension still declared by a retained schema.

    A dimension scoped only to a deleted instance must NOT be deleted while a retained schema
    still declares it (DatasetSchemaDimension cascades from Dimension, so deleting it would
    silently corrupt the kept schema).
    """
    deleted = _ic()
    kept = _ic()

    dim = DimensionFactory.create()
    DimensionScope.objects.create(dimension=dim, **_scope(deleted))

    kept_schema = DatasetSchemaFactory.create()
    DatasetSchemaScope.objects.create(schema=kept_schema, **_scope(kept))
    DatasetSchemaDimensionFactory.create(schema=kept_schema, dimension=dim)

    run_trim([kept])

    assert Dimension.objects.filter(pk=dim.pk).exists()
    assert DatasetSchemaDimension.objects.filter(schema=kept_schema, dimension=dim).exists()


def test_unrelated_unscoped_schema_preserved(run_trim):
    """
    Leave pre-existing unscoped, dataset-less schemas alone.

    The orphan-schema cleanup must only touch schemas this trim affected (those that lost a deleted
    scope or backed a deleted dataset), not every scope-less, dataset-less schema in the database —
    e.g. a draft/library schema unrelated to the instances being trimmed.
    """
    deleted = _ic()
    kept = _ic()

    # The deleted instance has a directly-scoped dataset; its bare auto-created schema (no scope of
    # its own) is affected by the trim and should be cleaned up.
    deleted_ds = DatasetFactory.create(**_scope(deleted))
    bare_schema = deleted_ds.schema
    assert bare_schema is not None

    # An unrelated library schema with no scopes and no datasets must survive.
    library_schema = DatasetSchemaFactory.create()

    run_trim([kept])

    assert DatasetSchema.objects.filter(pk=library_schema.pk).exists()
    assert not DatasetSchema.objects.filter(pk=bare_schema.pk).exists()


def test_unrelated_unscoped_dimension_preserved(run_trim):
    """
    Leave pre-existing unscoped dimensions alone.

    Like the schema cleanup, orphan-dimension cleanup must only touch dimensions whose deleted scope
    this trim removed, not every scope-less, unreferenced dimension in the database.
    """
    deleted = _ic()
    kept = _ic()

    # A dimension scoped only to the deleted instance should be removed...
    scoped_dim = DimensionFactory.create()
    DimensionScope.objects.create(dimension=scoped_dim, **_scope(deleted))

    # ...but an unrelated library dimension (no scopes, no schema links, no data points) must survive.
    library_dim = DimensionFactory.create()
    DimensionCategoryFactory.create(dimension=library_dim)

    run_trim([kept])

    assert not Dimension.objects.filter(pk=scoped_dim.pk).exists()
    assert Dimension.objects.filter(pk=library_dim.pk).exists()


def test_ancestor_organizations_of_kept_instance_preserved(run_trim):
    """
    Keep the ancestors of a retained instance's organization.

    A retained instance attached to a child organization must keep its parent/root organizations:
    available_for_instance() only returns the org and its descendants, so without ancestor handling
    the deletion would remove the parent and corrupt the kept instance's treebeard hierarchy.
    """
    parent_org = OrganizationFactory.create()
    child_org = OrganizationFactory.create(parent=parent_org)

    kept = InstanceConfigFactory.create(identifier='trim-kept-child', name='Kept on child org', organization=child_org)
    deleted = _ic()  # has its own unrelated root org, which should be deleted

    run_trim([kept])

    # The kept instance, its (child) organization and the ancestor are all retained and intact.
    assert InstanceConfig.objects.filter(pk=kept.pk).exists()
    assert Organization.objects.filter(pk=child_org.pk).exists()
    assert Organization.objects.filter(pk=parent_org.pk).exists()
    child_org.refresh_from_db()
    assert child_org.get_parent() == parent_org
    # The unrelated deleted instance and its organization are gone.
    assert not InstanceConfig.objects.filter(pk=deleted.pk).exists()


def test_outcome_pages_referencing_deleted_nodes_are_deleted(run_trim):
    """
    Delete outcome pages that reference nodes of a deleted instance before deleting the nodes.

    Wagtail translations or otherwise detached pages can survive deletion of the concrete site
    root subtree, but OutcomePage.outcome_node protects NodeConfig from deletion. Instance
    deletion must clean up those page rows explicitly before deleting the instance's nodes.
    """
    deleted = _ic()
    kept = _ic()
    node = NodeConfigFactory.create(instance=deleted)
    root = Page.get_first_root_node()
    assert root is not None
    outcome_page = root.add_child(
        instance=OutcomePage(
            title='Deleted instance outcome',
            slug='deleted-instance-outcome',
            outcome_node=node,
        ),
    )

    run_trim([kept])

    assert not OutcomePage.objects.filter(pk=outcome_page.pk).exists()
    assert not InstanceConfig.objects.filter(pk=deleted.pk).exists()
    assert InstanceConfig.objects.filter(pk=kept.pk).exists()
