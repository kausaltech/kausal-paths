from types import SimpleNamespace
from typing import cast

import polars as pl
import pytest

from common import polars as ppl
from frameworks.datasets import FrameworkMeasureDVCDataset2
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
            instance=SimpleNamespace(reference_year=2020, maximum_historical_year=2020),
            unit_registry=unit_registry,
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

    result = dataset.before_temporal_fill(raw)

    assert result['Value'].to_list() == [42.0]
    assert result['ObservedDataPoint'].to_list() == [True]
    assert result['FromMeasureDataPoint'].to_list() == [True]
