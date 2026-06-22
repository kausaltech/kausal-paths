"""Round-trip tests for dataset provenance (source references + comments) in the snapshot."""

from datetime import date
from decimal import Decimal

from django.contrib.contenttypes.models import ContentType

import pytest

from kausal_common.datasets.models import (
    DataPoint,
    DataPointComment,
    DataPointCommentReviewState,
    DatasetSourceReference,
    DataSource,
)
from kausal_common.datasets.tests.factories import (
    DataPointFactory,
    DatasetFactory,
    DatasetMetricFactory,
    DatasetSchemaFactory,
)

from nodes.instance_serialization import DatasetSnapshot, _import_dataset
from nodes.tests.factories import InstanceConfigFactory
from users.tests.factories import UserFactory

pytestmark = pytest.mark.django_db


def test_dataset_provenance_round_trip():
    author = UserFactory.create()
    src = InstanceConfigFactory.create(name='prov-src', config_source='database')
    ct = ContentType.objects.get_for_model(src)

    schema = DatasetSchemaFactory.create()
    metric = DatasetMetricFactory.create(schema=schema, label='Value')
    ds = DatasetFactory.create(schema=schema, scope_content_type=ct, scope_id=src.pk, identifier='prov/ds')
    dp = DataPointFactory.create(dataset=ds, metric=metric, date=date(2023, 1, 1), value=Decimal(5))

    source = DataSource.objects.create(scope_content_type=ct, scope_id=src.pk, name='Census', url='https://example/')
    DatasetSourceReference.objects.create(dataset=ds, data_source=source)  # dataset-level
    DatasetSourceReference.objects.create(data_point=dp, data_source=source)  # data-point-level
    DataPointComment.objects.create(
        data_point=dp,
        text='looks off',
        is_review=True,
        review_state=DataPointCommentReviewState.UNRESOLVED,
        created_by=author,
        last_modified_by=author,
    )
    # A soft-deleted comment must NOT be carried.
    deleted = DataPointComment.objects.create(data_point=dp, text='ignore me', created_by=author)
    deleted.soft_delete(author)

    # --- Export ---
    snap = DatasetSnapshot.from_model_for_instance(ds, src)
    assert [s.name for s in snap.data_sources] == ['Census']
    assert len(snap.source_references) == 2
    assert {r.point is None for r in snap.source_references} == {True, False}
    assert len(snap.comments) == 1  # soft-deleted one excluded
    comment_snap = snap.comments[0]
    assert comment_snap.text == 'looks off'
    assert comment_snap.created_by == str(author.uuid)  # user carried by uuid
    assert comment_snap.last_modified_by == str(author.uuid)

    # --- Import into a fresh instance ---
    dst = InstanceConfigFactory.create(name='prov-dst', config_source='database')
    dst_ct = ContentType.objects.get_for_model(dst)
    new_ds = _import_dataset(dst, snap, dst_ct, {})

    # DataSource recreated, scoped to the copy, with a fresh uuid.
    new_sources = DataSource.objects.filter(scope_content_type=dst_ct, scope_id=dst.pk, name='Census')
    assert new_sources.count() == 1
    assert new_sources.get().uuid != source.uuid

    # References reattached at both the dataset and data-point level.
    new_dp = DataPoint.objects.get(dataset=new_ds)
    assert DatasetSourceReference.objects.filter(dataset=new_ds, data_point__isnull=True).count() == 1
    assert DatasetSourceReference.objects.filter(data_point=new_dp).count() == 1

    # Comment recreated against the right data point, with the user resolved by uuid.
    new_comments = DataPointComment.objects.filter(data_point__dataset=new_ds)
    assert new_comments.count() == 1
    new_comment = new_comments.get()
    assert new_comment.text == 'looks off'
    assert new_comment.is_review is True
    assert new_comment.review_state == DataPointCommentReviewState.UNRESOLVED
    assert new_comment.created_by_id == author.pk  # resolved from uuid
    assert new_comment.last_modified_by_id == author.pk


def test_dataset_provenance_unknown_user_uuid_is_dropped():
    """A comment whose author uuid isn't present in the target leaves created_by null."""
    src = InstanceConfigFactory.create(name='prov-src2', config_source='database')
    ct = ContentType.objects.get_for_model(src)
    schema = DatasetSchemaFactory.create()
    metric = DatasetMetricFactory.create(schema=schema, label='Value')
    ds = DatasetFactory.create(schema=schema, scope_content_type=ct, scope_id=src.pk, identifier='prov/ds2')
    dp = DataPointFactory.create(dataset=ds, metric=metric, date=date(2023, 1, 1), value=Decimal(5))
    author = UserFactory.create()
    DataPointComment.objects.create(data_point=dp, text='note', created_by=author)

    snap = DatasetSnapshot.from_model_for_instance(ds, src)
    # Drop the author so import can't resolve the uuid.
    author.delete()

    dst = InstanceConfigFactory.create(name='prov-dst2', config_source='database')
    new_ds = _import_dataset(dst, snap, ContentType.objects.get_for_model(dst), {})

    new_comment = DataPointComment.objects.get(data_point__dataset=new_ds)
    assert new_comment.text == 'note'
    assert new_comment.created_by_id is None
