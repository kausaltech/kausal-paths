"""
The per-instance dataset status report.

The verdicts are the whole product here: they decide which commands an operator is told to
run, and in which order. These tests pin each one, including the two that exist because the
naive version of this command got them wrong — an unstamped row must not be invisible, and
an admin-authored row must not be reported as broken.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

import polars as pl
import pytest

from kausal_common.datasets.models import Dataset, DatasetMetric

from nodes.management.commands.dataset_status import candidate_dataset_ids, status_for
from nodes.management.commands.load_dvc_dataset import Command as LoadCommand
from nodes.models import NodeInputPortBinding
from nodes.tests.factories import InstanceConfigFactory, NodeConfigFactory
from nodes.tests.test_load_dvc_dataset_refresh import make_context

pytestmark = pytest.mark.django_db

DS_ID = 'test/status'
COMMIT = 'aaa111'


def _ctx(columns: dict[str, list[float]], units: dict[str, str], commit: str | None = COMMIT) -> Any:
    return make_context(pl.DataFrame({'Year': [2020], **columns}), units, commit=commit)


def _import(ic, ctx) -> Dataset:
    LoadCommand().sync_dataset(ic, ctx, DS_ID)
    return Dataset.objects.get(identifier=DS_ID)


def test_a_freshly_imported_dataset_is_current():
    ic = InstanceConfigFactory.create(name='status-current', config_source='database')
    ctx = _ctx({'Value': [1.0]}, {'Value': 'kt'})
    _import(ic, ctx)

    assert status_for(ic, ctx, DS_ID).verdict == 'current'


def test_a_dataset_with_no_row_is_new():
    ic = InstanceConfigFactory.create(name='status-new', config_source='database')

    status = status_for(ic, _ctx({'Value': [1.0]}, {'Value': 'kt'}), DS_ID)

    assert status.verdict == 'new'


def test_a_changed_commit_means_import():
    ic = InstanceConfigFactory.create(name='status-import', config_source='database')
    _import(ic, _ctx({'Value': [1.0]}, {'Value': 'kt'}))

    status = status_for(ic, _ctx({'Value': [1.0, 2.0][:1]}, {'Value': 'kt'}, commit='bbb222'), DS_ID)

    assert status.verdict == 'import'
    assert 'aaa111' in status.detail
    assert 'bbb222' in status.detail


def test_a_renamed_metric_that_bindings_hold_means_rename_first():
    """The verdict that has to come before the import, or the import refuses."""
    ic = InstanceConfigFactory.create(name='status-rename', config_source='database')
    dataset = _import(ic, _ctx({'Value': [1.0]}, {'Value': 'kt'}))
    NodeInputPortBinding.objects.create(
        instance=ic,
        node=NodeConfigFactory.create(instance=ic),
        port_id=UUID('55555555-5555-5555-5555-555555555555'),
        dataset=dataset,
        metric=DatasetMetric.objects.get(schema=dataset.schema, name='Value'),
    )

    status = status_for(ic, _ctx({'default': [1.0]}, {'default': 'kt'}), DS_ID)

    assert status.verdict == 'rename first'
    assert 'Value' in status.detail


def test_a_row_with_no_dvc_source_is_reported_as_db_only_not_broken():
    """An admin-authored dataset has nothing to be out of date with, and must not read as an error."""
    ic = InstanceConfigFactory.create(name='status-dbonly', config_source='database')
    _import(ic, _ctx({'Value': [1.0]}, {'Value': 'kt'}))
    Dataset.objects.filter(identifier=DS_ID).update(external_ref=None)

    def _fails(_ds_id):
        raise FileNotFoundError('no such dataset in the repo')

    ctx = _ctx({'Value': [1.0]}, {'Value': 'kt'})
    ctx.load_dvc_dataset = _fails

    status = status_for(ic, ctx, DS_ID)
    assert status.verdict == 'db only'
    assert not status.is_stale, 'db-only rows are not work to do'


def test_a_stamped_row_whose_source_will_not_read_is_unreadable():
    ic = InstanceConfigFactory.create(name='status-unreadable', config_source='database')
    _import(ic, _ctx({'Value': [1.0]}, {'Value': 'kt'}))

    def _fails(_ds_id):
        raise FileNotFoundError('no such dataset in the repo')

    ctx = _ctx({'Value': [1.0]}, {'Value': 'kt'})
    ctx.load_dvc_dataset = _fails

    status = status_for(ic, ctx, DS_ID)
    assert status.verdict == 'unreadable'
    assert status.is_stale


def test_rows_without_a_provenance_stamp_are_still_listed():
    """
    The bug this command exists to avoid: silence about the rows most likely to be stale.

    Selecting only ``external_ref``-stamped rows hid 15 of mainz-bisko's 32 datasets,
    ``bisko/weather_correction`` among them — the one that was actually blocking.
    """
    ic = InstanceConfigFactory.create(name='status-unstamped', config_source='database')
    ctx = _ctx({'Value': [1.0]}, {'Value': 'kt'})
    _import(ic, ctx)
    Dataset.objects.filter(identifier=DS_ID).update(external_ref=None)
    # The model declares nothing; the row is the only evidence this dataset exists.
    ctx.get_all_dvc_dataset_ids = set

    assert DS_ID in candidate_dataset_ids(ic, cast('Any', ctx))
