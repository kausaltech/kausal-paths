"""Tests for re-importing a dataset that already exists in the DB (`load_dvc_dataset --force`)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import polars as pl
import pytest

from kausal_common.datasets.models import DataPoint, Dataset, DatasetMetric

from nodes.management.commands.load_dvc_dataset import Command, build_dataset_plan
from nodes.models import DatasetPort, NodeConfig, NodeInputPortBinding
from nodes.tests.factories import InstanceConfigFactory, NodeConfigFactory

pytestmark = pytest.mark.django_db

DS_ID = 'test/refresh'


def make_context(df: pl.DataFrame, units: dict[str, str], commit: str | None = None) -> Any:
    """Build a stand-in for the runtime Context, carrying just what sync_dataset reads."""
    dvc_dataset = SimpleNamespace(
        df=df,
        units=units,
        index_columns=['Year'],
        metadata={'name': {'en': 'Refreshable dataset'}, 'metrics': [{'column_id': col} for col in units]},
    )
    repo_spec = SimpleNamespace(url='https://example.com/dvc.git', commit=commit, dvc_remote=None) if commit else None
    return cast(
        'Any',
        SimpleNamespace(
            dataset_repo_spec=repo_spec,
            dimensions={},
            instance=SimpleNamespace(default_language='en'),
            load_dvc_dataset=lambda _dataset_id: dvc_dataset,
        ),
    )


def test_force_reimport_replaces_data_in_place():
    """The row survives a --force re-import: same pk and UUID, new data, restamped commit."""
    ic = InstanceConfigFactory.create(name='refresh-instance', config_source='database')

    first = make_context(pl.DataFrame({'Year': [2020, 2021], 'value': [1.0, 2.0]}), {'value': 'kt'}, commit='aaa111')
    Command().sync_dataset(ic, first, DS_ID)

    dataset = Dataset.objects.get(identifier=DS_ID)
    original_pk, original_uuid = dataset.pk, dataset.uuid
    assert DataPoint.objects.filter(dataset=dataset).count() == 2

    second = make_context(
        pl.DataFrame({'Year': [2020, 2021, 2022], 'value': [10.0, 20.0, 30.0]}), {'value': 'kt'}, commit='bbb222'
    )
    Command().sync_dataset(ic, second, DS_ID, force=True)

    dataset.refresh_from_db()
    assert dataset.pk == original_pk, 'the row must be reused, not recreated'
    assert dataset.uuid == original_uuid, 'a new UUID would orphan references in published revisions'
    assert (dataset.external_ref or {})['commit'] == 'bbb222', 'provenance must record the commit the data came from'

    values = sorted(float(dp.value) for dp in DataPoint.objects.filter(dataset=dataset) if dp.value is not None)
    assert values == [10.0, 20.0, 30.0], 'old data points must be gone, not merged with the new ones'


def test_force_reimport_survives_a_protected_reference():
    """A DatasetPort binding the dataset used to make --force fail with ProtectedError."""
    ic = InstanceConfigFactory.create(name='refresh-ported', config_source='database')
    ctx = make_context(pl.DataFrame({'Year': [2020], 'value': [1.0]}), {'value': 'kt'}, commit='aaa111')
    Command().sync_dataset(ic, ctx, DS_ID)

    dataset = Dataset.objects.get(identifier=DS_ID)
    metric = DatasetMetric.objects.get(schema=dataset.schema, name='value')
    node: NodeConfig = NodeConfigFactory.create(instance=ic)
    port = DatasetPort.objects.create(
        instance=ic,
        node=node,
        port_id=UUID('11111111-1111-1111-1111-111111111111'),
        dataset=dataset,
        metric=metric,
    )

    newer = make_context(pl.DataFrame({'Year': [2020, 2021], 'value': [5.0, 6.0]}), {'value': 'kt'}, commit='bbb222')
    Command().sync_dataset(ic, newer, DS_ID, force=True)

    port.refresh_from_db()
    assert port.dataset_id == dataset.pk, 'the binding must still resolve after the refresh'
    assert DataPoint.objects.filter(dataset=dataset).count() == 2


def test_recreate_builds_a_fresh_row():
    """--recreate keeps the old delete-and-rebuild strategy; it used to leave both objects unsaved."""
    ic = InstanceConfigFactory.create(name='refresh-recreate', config_source='database')
    Command().sync_dataset(ic, make_context(pl.DataFrame({'Year': [2020], 'value': [1.0]}), {'value': 'kt'}), DS_ID)
    original_pk = Dataset.objects.get(identifier=DS_ID).pk

    newer = make_context(pl.DataFrame({'Year': [2020, 2021], 'value': [7.0, 8.0]}), {'value': 'kt'})
    Command().sync_dataset(ic, newer, DS_ID, force=True, recreate=True)

    dataset = Dataset.objects.get(identifier=DS_ID)
    assert dataset.pk != original_pk
    assert DataPoint.objects.filter(dataset=dataset).count() == 2


def test_force_reimport_updates_a_changed_unit():
    ic = InstanceConfigFactory.create(name='refresh-units', config_source='database')
    Command().sync_dataset(ic, make_context(pl.DataFrame({'Year': [2020], 'value': [1.0]}), {'value': 'kt'}), DS_ID)

    Command().sync_dataset(ic, make_context(pl.DataFrame({'Year': [2020], 'value': [1.0]}), {'value': 'Mt'}), DS_ID, force=True)

    dataset = Dataset.objects.get(identifier=DS_ID)
    assert DatasetMetric.objects.get(schema=dataset.schema, name='value').unit == 'Mt'


def test_plan_reports_the_diff_without_writing():
    ic = InstanceConfigFactory.create(name='refresh-plan', config_source='database')
    ctx = make_context(pl.DataFrame({'Year': [2020], 'value': [1.0]}), {'value': 'kt'}, commit='aaa111')
    Command().sync_dataset(ic, ctx, DS_ID)
    dataset = Dataset.objects.get(identifier=DS_ID)

    newer = make_context(
        pl.DataFrame({'Year': [2020, 2021], 'value': [1.0, 2.0], 'extra': [3.0, 4.0]}),
        {'value': 'kt', 'extra': 'kt'},
        commit='bbb222',
    )
    Command().sync_dataset(ic, newer, DS_ID, force=True, plan_only=True)

    assert DataPoint.objects.filter(dataset=dataset).count() == 1, '--plan must not touch the data'
    assert not DatasetMetric.objects.filter(schema=dataset.schema, name='extra').exists()


def test_provenance_reports_disagreeing_pins(monkeypatch: pytest.MonkeyPatch):
    """Importing from a different commit than the model expects must never be silent."""
    from nodes.defs.instance_defs import DatasetRepoSpec
    from nodes.management.commands import load_dvc_dataset as mod

    url = 'https://example.com/dvc.git'
    ic = InstanceConfigFactory.create(name='prov-mismatch', config_source='database')
    db_spec = DatasetRepoSpec(url=url, commit='db00000')
    assert ic.spec is not None
    ic.spec.dataset_repo = db_spec
    ctx = cast('Any', SimpleNamespace(dataset_repo_spec=db_spec))
    monkeypatch.setattr(mod, '_yaml_repo_spec', lambda _ic: DatasetRepoSpec(url=url, commit='yaml111'))

    auto = mod.resolve_repo_provenance(ic, ctx, 'auto')
    assert auto.used_source == 'db', "auto follows the instance's declared config source"
    assert auto.spec is not None
    assert auto.spec.commit == 'db00000'
    assert auto.sources_disagree

    override = mod.resolve_repo_provenance(ic, ctx, 'yaml')
    assert override.used_source == 'yaml'
    assert override.spec is not None
    assert override.spec.commit == 'yaml111'


def test_plan_flags_a_dropped_metric_that_ports_still_bind():
    """The diagnosis names the blocker up front instead of leaving it for sync_instance_to_db."""
    ic = InstanceConfigFactory.create(name='refresh-blocked', config_source='database')
    Command().sync_dataset(
        ic, make_context(pl.DataFrame({'Year': [2020], 'old_col': [1.0]}), {'old_col': 'kt'}, commit='aaa111'), DS_ID
    )
    dataset = Dataset.objects.get(identifier=DS_ID)
    metric = DatasetMetric.objects.get(schema=dataset.schema, name='old_col')
    DatasetPort.objects.create(
        instance=ic,
        node=NodeConfigFactory.create(instance=ic),
        port_id=UUID('22222222-2222-2222-2222-222222222222'),
        dataset=dataset,
        metric=metric,
    )

    plan = build_dataset_plan(
        ds_id=DS_ID,
        dataset=dataset,
        incoming_metric_cols=['new_col'],
        incoming_data_points=1,
        incoming_commit='bbb222',
    )

    assert plan.dropped_metrics == [('old_col', 1)]
    assert plan.added_metrics == ['new_col']
    assert plan.blockers, 'dropping a bound metric must be reported as a blocker'


def test_plan_flags_a_dropped_metric_that_only_an_input_binding_holds():
    """
    An input binding protects a metric just as a dataset port does; count both.

    Counting only the ports made the command report a clean plan and then die on
    ``ProtectedError`` partway through the sync — after deleting the data points, with the
    transaction rolled back and nothing to tell the operator what still held the metric.
    """
    ic = InstanceConfigFactory.create(name='refresh-blocked-binding', config_source='database')
    Command().sync_dataset(
        ic, make_context(pl.DataFrame({'Year': [2020], 'old_col': [1.0]}), {'old_col': 'kt'}, commit='aaa111'), DS_ID
    )
    dataset = Dataset.objects.get(identifier=DS_ID)
    metric = DatasetMetric.objects.get(schema=dataset.schema, name='old_col')
    NodeInputPortBinding.objects.create(
        instance=ic,
        node=NodeConfigFactory.create(instance=ic),
        port_id=UUID('33333333-3333-3333-3333-333333333333'),
        dataset=dataset,
        metric=metric,
    )

    plan = build_dataset_plan(
        ds_id=DS_ID,
        dataset=dataset,
        incoming_metric_cols=['new_col'],
        incoming_data_points=1,
        incoming_commit='bbb222',
    )

    assert plan.dropped_metrics == [('old_col', 1)]
    assert plan.blockers, 'an input binding must block the drop, not just a dataset port'


def test_a_valueless_cell_becomes_a_null_data_point_rather_than_being_skipped():
    """
    An empty cell has to survive the import as a null DataPoint.

    BISKO Pruefschritt 1.4 requires that a municipality-confirmed zero be distinguishable
    from a cell nobody has filled in. `DataAvailabilityNode` tests `is_not_null()`, so a null
    reads as "no data" while a 0 reads as a valid entry -- but only if the import keeps the
    cell. Skipping it (the old behaviour) lost the row and forced template datasets to ship
    zeros, which is exactly the finding.
    """
    ic = InstanceConfigFactory.create(name='nullable-instance', config_source='database')
    ctx = make_context(
        pl.DataFrame({'Year': [2020, 2021, 2022], 'value': [1.0, None, 3.0]}),
        {'value': 'kt'},
        commit='ccc333',
    )
    Command().sync_dataset(ic, ctx, DS_ID)

    dataset = Dataset.objects.get(identifier=DS_ID)
    points = {dp.date.year: dp.value for dp in DataPoint.objects.filter(dataset=dataset)}
    assert set(points) == {2020, 2021, 2022}, 'the valueless year must still have a row'
    assert points[2021] is None, 'the empty cell must be null, not absent and not zero'
    assert float(cast('Any', points[2020])) == 1.0
    assert float(cast('Any', points[2022])) == 3.0
