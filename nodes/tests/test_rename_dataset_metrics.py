"""
Renaming a dataset's metric rows to match columns renamed in the DVC data.

The point of renaming in place, rather than clearing the bindings and re-syncing, is that
the bindings never move: they keep pointing at the same metric row, which now answers to the
new name. These tests hold that promise, and the refusals that keep it honest.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

import polars as pl
import pytest

from kausal_common.datasets.models import Dataset, DatasetMetric

from nodes.management.commands.load_dvc_dataset import Command as LoadCommand
from nodes.management.commands.rename_dataset_metrics import build_rename_plan
from nodes.models import NodeInputPortBinding
from nodes.tests.factories import InstanceConfigFactory, NodeConfigFactory
from nodes.tests.test_load_dvc_dataset_refresh import make_context

pytestmark = pytest.mark.django_db

DS_ID = 'test/renamed'


def _import(ic, columns: dict[str, list[float]], units: dict[str, str]) -> Dataset:
    LoadCommand().sync_dataset(ic, make_context(pl.DataFrame({'Year': [2020], **columns}), units, commit='aaa111'), DS_ID)
    return Dataset.objects.get(identifier=DS_ID)


def _bind(ic, dataset: Dataset, metric: DatasetMetric) -> NodeInputPortBinding:
    """Bind the metric to a fresh node's input port; the reference is PROTECTed."""
    node = NodeConfigFactory.create(instance=ic)
    return NodeInputPortBinding.objects.create(
        instance=ic,
        node=node,
        port_id=UUID('44444444-4444-4444-4444-444444444444'),
        dataset=dataset,
        metric=metric,
    )


def test_plan_infers_a_single_rename():
    ic = InstanceConfigFactory.create(name='rename-infer', config_source='database')
    dataset = _import(ic, {'Value': [1.0]}, {'Value': 'kt'})

    plan = build_rename_plan(DS_ID, dataset, ['default'], explicit={})

    assert plan.renames == [('Value', 'default')]
    assert not plan.blockers


def test_plan_counts_the_bindings_that_will_survive():
    ic = InstanceConfigFactory.create(name='rename-counts', config_source='database')
    dataset = _import(ic, {'Value': [1.0]}, {'Value': 'kt'})
    metric = DatasetMetric.objects.get(schema=dataset.schema, name='Value')
    _bind(ic, dataset, metric)
    _bind(ic, dataset, metric)

    plan = build_rename_plan(DS_ID, dataset, ['default'], explicit={})

    assert plan.bindings == {'Value': 2}, 'every binding that holds the metric rides along'


def test_renaming_keeps_the_bindings_pointing_at_the_same_row():
    """The whole reason to rename rather than delete: identity is preserved, so nothing rebinds."""
    ic = InstanceConfigFactory.create(name='rename-keeps', config_source='database')
    dataset = _import(ic, {'Value': [1.0]}, {'Value': 'kt'})
    metric = DatasetMetric.objects.get(schema=dataset.schema, name='Value')
    binding = _bind(ic, dataset, metric)
    metric_uuid = metric.uuid

    metric.name = 'default'
    metric.save(update_fields=['name'])

    binding.refresh_from_db()
    assert binding.metric_id == metric.pk
    assert DatasetMetric.objects.get(pk=metric.pk).uuid == metric_uuid, 'the metric keeps its identity'
    assert DatasetMetric.objects.get(pk=metric.pk).name == 'default'


def test_renaming_clears_the_way_for_the_import():
    """After the rename the incoming column is *kept*, so nothing is dropped and nothing is protected."""
    from nodes.management.commands.load_dvc_dataset import build_dataset_plan

    ic = InstanceConfigFactory.create(name='rename-unblocks', config_source='database')
    dataset = _import(ic, {'Value': [1.0]}, {'Value': 'kt'})
    metric = DatasetMetric.objects.get(schema=dataset.schema, name='Value')
    _bind(ic, dataset, metric)

    blocked = build_dataset_plan(
        ds_id=DS_ID, dataset=dataset, incoming_metric_cols=['default'], incoming_data_points=1, incoming_commit='bbb222'
    )
    assert blocked.blockers, 'precondition: the rename is what unblocks it'

    metric.name = 'default'
    metric.save(update_fields=['name'])

    after = build_dataset_plan(
        ds_id=DS_ID, dataset=dataset, incoming_metric_cols=['default'], incoming_data_points=1, incoming_commit='bbb222'
    )
    assert not after.blockers
    assert after.kept_metrics == ['default']
    assert after.dropped_metrics == []


def test_plan_refuses_an_ambiguous_mapping():
    """Two renamed columns at once could pair up either way, so the operator has to say."""
    ic = InstanceConfigFactory.create(name='rename-ambiguous', config_source='database')
    dataset = _import(ic, {'a': [1.0], 'b': [2.0]}, {'a': 'kt', 'b': 'kt'})

    plan = build_rename_plan(DS_ID, dataset, ['x', 'y'], explicit={})

    assert not plan.renames
    assert any('cannot infer' in b for b in plan.blockers)


def test_plan_refuses_a_rename_onto_an_existing_metric():
    ic = InstanceConfigFactory.create(name='rename-collision', config_source='database')
    dataset = _import(ic, {'a': [1.0], 'b': [2.0]}, {'a': 'kt', 'b': 'kt'})

    plan = build_rename_plan(DS_ID, dataset, ['a', 'b'], explicit={'a': 'b'})

    assert not plan.renames
    assert any('already exists' in b for b in plan.blockers)


def test_plan_says_nothing_to_do_when_the_names_already_match():
    ic = InstanceConfigFactory.create(name='rename-noop', config_source='database')
    dataset = _import(ic, {'Value': [1.0]}, {'Value': 'kt'})

    plan = build_rename_plan(DS_ID, dataset, ['Value'], explicit={})

    assert not plan.renames
    assert not plan.blockers
    assert plan.note is not None


def test_plan_refuses_an_unknown_source_metric():
    ic = InstanceConfigFactory.create(name='rename-unknown', config_source='database')
    dataset = _import(ic, {'Value': [1.0]}, {'Value': 'kt'})

    plan = build_rename_plan(DS_ID, dataset, ['default'], explicit={'nope': 'default'})

    assert not plan.renames
    assert any('no metric named' in b for b in plan.blockers)


def test_a_missing_dataset_is_reported_not_crashed_on():
    plan = build_rename_plan('test/absent', cast('Any', None), ['default'], explicit={})

    assert not plan.renames
    assert not plan.blockers
    assert plan.note is not None
