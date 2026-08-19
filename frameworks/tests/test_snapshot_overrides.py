from types import SimpleNamespace

import pytest

from kausal_common.i18n.pydantic import set_i18n_context

from frameworks.models import Measure, MeasureDataPoint, MeasureTemplate, Section
from frameworks.tests.factories import FrameworkConfigFactory
from nodes.defs.instance_defs import InstanceMetadata, InstanceModelSpec, YearsSpec
from nodes.instance_serialization import InstanceSnapshot
from nodes.scenario import Scenario, ScenarioKind

pytestmark = pytest.mark.django_db


def make_framework_snapshot() -> InstanceSnapshot:
    """Build a minimal parsed-framework-YAML stand-in with demonstration identity."""
    with set_i18n_context('en', []):
        return InstanceSnapshot(
            metadata=InstanceMetadata(
                identifier='nzc',
                name='Demonstration Inventory',
                owner='City of Demonstration',
                primary_language='en',
            ),
            spec=InstanceModelSpec(
                years=YearsSpec(reference=2018, min_historical=2018, max_historical=2020, target=2030, model_end=2060),
            ),
        )


def add_measure_datapoints(fwc, years_with_values, years_without_values=()) -> None:
    section = Section.add_root(instance=Section(framework=fwc.framework, name='Root'))
    template = MeasureTemplate.objects.create(section=section, name='Test measure', unit='MWh/a')
    measure = Measure.objects.create(framework_config=fwc, measure_template=template)
    for year in years_with_values:
        MeasureDataPoint.objects.create(measure=measure, year=year, value=1.0)
    for year in years_without_values:
        MeasureDataPoint.objects.create(measure=measure, year=year, value=None)


def test_apply_snapshot_overrides_identity_and_years():
    fwc = FrameworkConfigFactory.create(baseline_year=2021, target_year=2035, organization_name='Test Org')
    add_measure_datapoints(fwc, years_with_values=[2019, 2022])

    snapshot = make_framework_snapshot()
    result = fwc.apply_snapshot_overrides(snapshot)

    ic = fwc.instance_config
    assert result.metadata.uuid == ic.uuid
    assert result.metadata.identifier == ic.identifier
    assert str(result.metadata.name) == ic.get_name()
    assert str(result.metadata.owner) == 'Test Org'
    years = result.spec.years
    assert years.reference == 2021
    assert years.min_historical == 2019
    assert years.max_historical == 2022
    assert years.target == 2035
    # Framework-YAML value survives; only the city-specific fields are overlaid.
    assert years.model_end == 2060
    # The input snapshot is not mutated.
    assert snapshot.metadata.identifier == 'nzc'
    assert snapshot.spec.years.reference == 2018


def test_apply_snapshot_overrides_defaults_without_datapoints():
    fwc = FrameworkConfigFactory.create(baseline_year=2020, target_year=None)

    result = fwc.apply_snapshot_overrides(make_framework_snapshot())

    years = result.spec.years
    assert years.reference == 2020
    assert years.min_historical == 2020
    assert years.max_historical == 2020
    # No fwc target year: the framework YAML's target stands.
    assert years.target == 2030


def test_scenario_actual_historical_years_authored_wins():
    scenario = Scenario(id='progress_tracking', name='PT', kind=ScenarioKind.PROGRESS_TRACKING, actual_historical_years=[2019])
    # No context bound: an authored value must not need one.
    assert scenario.get_actual_historical_years() == [2019]


def test_scenario_actual_historical_years_lazy_from_context():
    scenario = Scenario(id='progress_tracking', name='PT', kind=ScenarioKind.PROGRESS_TRACKING)
    scenario._context = SimpleNamespace(measure_datapoint_years=[2020, 2021])  # type: ignore[assignment]
    assert scenario.get_actual_historical_years() == [2020, 2021]

    other = Scenario(id='default', name='Default', kind=ScenarioKind.DEFAULT)
    assert other.get_actual_historical_years() is None


def test_context_measure_datapoint_years():
    from nodes.context import Context

    fwc = FrameworkConfigFactory.create(baseline_year=2020)
    add_measure_datapoints(fwc, years_with_values=[2021, 2019], years_without_values=[2018])

    ctx = Context.__new__(Context)
    ctx.instance = SimpleNamespace(config=fwc.instance_config)  # type: ignore[assignment]
    assert ctx.framework_config_data is not None
    assert ctx.framework_config_data.id == fwc.pk
    # Null-valued datapoints do not count as actual data; result is sorted.
    assert ctx.measure_datapoint_years == [2019, 2021]

    unbound = Context.__new__(Context)
    unbound.instance = SimpleNamespace(config=None)  # type: ignore[assignment]
    assert unbound.framework_config_data is None
    assert unbound.measure_datapoint_years is None
