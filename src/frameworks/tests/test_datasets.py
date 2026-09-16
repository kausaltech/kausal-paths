from types import SimpleNamespace
from typing import cast

from django.db import connection
from django.test.utils import CaptureQueriesContext

import polars as pl
import pytest

from common import polars as ppl
from frameworks.datasets import (
    FrameworkMeasureDVCDataset,
    FrameworkMeasureDVCDataset2,
    ObservationDataset,
    collect_measure_datapoints,
)
from frameworks.models import Measure, MeasureDataPoint, MeasureTemplate, Section
from frameworks.tests.factories import FrameworkConfigFactory
from nodes.context import Context, FrameworkConfigData
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


def test_framework_measure_dataset_injects_only_bound_config_datapoints() -> None:
    fwc = FrameworkConfigFactory.create(baseline_year=2020)
    other_fwc = FrameworkConfigFactory.create(framework=fwc.framework, baseline_year=2020)
    section = Section.add_root(instance=Section(framework=fwc.framework, name='Root'))
    template = MeasureTemplate.objects.create(section=section, name='Energy', unit='MWh/a')
    measure = Measure.objects.create(framework_config=fwc, measure_template=template)
    other_measure = Measure.objects.create(framework_config=other_fwc, measure_template=template)
    MeasureDataPoint.objects.create(measure=measure, year=2020, value=42.0, default_value=7.0)
    MeasureDataPoint.objects.create(measure=other_measure, year=2020, value=99.0, default_value=8.0)

    context = cast(
        'Context',
        SimpleNamespace(
            dimensions={},
            framework_config_data=FrameworkConfigData(last_modified_at=fwc.last_modified_at, id=fwc.pk),
            get_parameter_value=lambda *_args, **_kwargs: False,
            instance=SimpleNamespace(reference_year=2020, maximum_historical_year=2020, target_year=2030),
            unit_registry=unit_registry,
            load_dvc_dataset=lambda _id: SimpleNamespace(df=raw),
        ),
    )
    dataset = FrameworkMeasureDVCDataset2(id='framework-energy', context=context)
    raw = ppl.PathsDataFrame._from_pydf(
        pl.DataFrame({'Year': [2020], 'uuid': [str(template.uuid)], 'Value': [1.0]})._df,
        meta=ppl.DataFrameMeta(
            primary_keys=['Year', 'uuid'],
            units={'Value': unit_registry.parse_units('MWh/a')},
        ),
    )

    # Separate bindings and overlay implementations share one raw DB snapshot.
    second_dataset = FrameworkMeasureDVCDataset2(id='framework-energy-copy', context=context)
    observations = ObservationDataset(id='framework-observations', context=context)
    legacy = FrameworkMeasureDVCDataset(id='framework-legacy', context=context)
    with CaptureQueriesContext(connection) as queries:
        result = dataset.before_temporal_fill(raw)
        second_result = second_dataset.before_temporal_fill(raw.ensure_unit('Value', unit_registry.parse_units('kWh/a')))
        observation_result = observations._overlay_observations(raw.with_columns(pl.col('uuid').str.replace_all('-', '_')))
        legacy_result = legacy.before_temporal_fill(
            raw.rename({'uuid': 'UUID'}).with_columns(pl.lit('Energy').alias('Sector'), pl.lit('MWh/a').alias('Unit'))
        )
        assert dataset.get_observation_years() == [2020]
        assert second_dataset.get_observation_years() == [2020]

    assert len(queries) == 1

    assert 'ORDER BY' not in queries[0]['sql']
    assert second_result['Value'].to_list() == [42000.0]
    assert legacy_result['Value'].to_list() == [42.0]
    assert observation_result['Value'].to_list() == [42.0]
    assert observation_result['observed'].to_list() == [True]

    assert result['Value'].to_list() == [42.0]
    assert result['ObservedDataPoint'].to_list() == [True]
    assert result['FromMeasureDataPoint'].to_list() == [True]


def test_measure_datapoint_snapshot_is_lazy_isolated_and_refreshed() -> None:
    fwc = FrameworkConfigFactory.create(baseline_year=2020)
    other_fwc = FrameworkConfigFactory.create(framework=fwc.framework, baseline_year=2020)
    section = Section.add_root(instance=Section(framework=fwc.framework, name='Root'))
    template = MeasureTemplate.objects.create(section=section, name='Energy', unit='MWh/a')
    other_template = MeasureTemplate.objects.create(section=section, name='Share', unit='%')
    measure = Measure.objects.create(framework_config=fwc, measure_template=template)
    share = Measure.objects.create(framework_config=fwc, measure_template=other_template)
    other_measure = Measure.objects.create(framework_config=other_fwc, measure_template=template)
    datapoint = MeasureDataPoint.objects.create(measure=measure, year=2020, value=42.0, default_value=7.0)
    MeasureDataPoint.objects.create(measure=share, year=2030, value=None, default_value=25.0)
    MeasureDataPoint.objects.create(measure=other_measure, year=2020, value=99.0)
    uuid = str(template.uuid)
    share_uuid = str(other_template.uuid)

    with CaptureQueriesContext(connection) as queries:
        data = FrameworkConfigData(id=fwc.pk, last_modified_at=fwc.last_modified_at)
        assert collect_measure_datapoints(data, []).is_empty()
        assert collect_measure_datapoints(None, [uuid]).is_empty()
    assert len(queries) == 0

    with CaptureQueriesContext(connection) as queries:
        energy = collect_measure_datapoints(data, [uuid])
        shares = collect_measure_datapoints(data, [share_uuid])
        assert collect_measure_datapoints(data, ['00000000-0000-0000-0000-000000000000']).is_empty()
    assert len(queries) == 1
    assert energy.rows() == [(uuid, 2020, 42.0, 7.0, 'MWh/a')]
    assert shares.rows() == [(share_uuid, 2030, None, 25.0, '%')]

    # Consumers can mutate their selection without corrupting the shared snapshot.
    energy.replace_column(2, pl.Series('MeasureValue', [1000.0]))
    with CaptureQueriesContext(connection) as queries:
        assert collect_measure_datapoints(data, [uuid])['MeasureValue'].to_list() == [42.0]
    assert len(queries) == 0

    other_data = FrameworkConfigData(id=other_fwc.pk, last_modified_at=other_fwc.last_modified_at)
    with CaptureQueriesContext(connection) as queries:
        assert collect_measure_datapoints(other_data, [uuid])['MeasureValue'].to_list() == [99.0]
    assert len(queries) == 1

    datapoint.value = 52.0
    datapoint.save(update_fields=['value'])
    # A new runtime gets a fresh snapshot even when the timestamp was not updated.
    refreshed = FrameworkConfigData(id=fwc.pk, last_modified_at=fwc.last_modified_at)
    with CaptureQueriesContext(connection) as queries:
        assert collect_measure_datapoints(refreshed, [uuid])['MeasureValue'].to_list() == [52.0]
        assert collect_measure_datapoints(data, [uuid])['MeasureValue'].to_list() == [42.0]
    assert len(queries) == 1


def test_empty_measure_datapoint_snapshot_is_cached() -> None:
    fwc = FrameworkConfigFactory.create(baseline_year=2020)
    data = FrameworkConfigData(id=fwc.pk, last_modified_at=fwc.last_modified_at)
    with CaptureQueriesContext(connection) as queries:
        for _ in range(3):
            frame = collect_measure_datapoints(data, ['00000000-0000-0000-0000-000000000000'])
            assert frame.is_empty()
            assert frame.schema['MeasureValue'] == pl.Float64
    assert len(queries) == 1
