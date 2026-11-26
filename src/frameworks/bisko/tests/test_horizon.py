"""Moving the municipalities' model horizon out to the template's."""

import pytest

from frameworks.bisko.horizon import extend_horizon
from frameworks.tests.factories import FrameworkConfigFactory, FrameworkFactory
from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.models import InstanceChangeOperation, InstanceConfig
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory

pytestmark = pytest.mark.django_db


def _instance_config(years: YearsSpec, *, template: InstanceConfig | None = None) -> InstanceConfig:
    instance = InstanceFactory.create()
    return InstanceConfigFactory.create(
        identifier=instance.id,
        instance=instance,
        config_source='database',
        spec=InstanceModelSpec(years=years),
        template_revision=template.save_revision() if template is not None else None,
    )


def _years(ic: InstanceConfig) -> YearsSpec:
    ic.refresh_from_db()
    return ic.ensure_spec().years


def test_municipalities_take_the_template_horizon_and_keep_their_history() -> None:
    template = _instance_config(YearsSpec(reference=1990, min_historical=1990, max_historical=2023, target=2045, model_end=2045))
    framework = FrameworkFactory.create(template_instance=template)
    municipality = _instance_config(
        YearsSpec(reference=1990, min_historical=2020, max_historical=2022, target=2025, model_end=2025), template=template
    )
    FrameworkConfigFactory.create(framework=framework, instance_config=municipality)

    report = extend_horizon(framework)

    assert report == [f'{municipality.identifier}: target 2025, model_end 2025 -> target 2045, model_end 2045']
    years = _years(municipality)
    assert (years.target, years.model_end) == (2045, 2045)
    assert (years.min_historical, years.max_historical) == (2020, 2022)
    operation = InstanceChangeOperation.objects.get(instance_config=municipality)
    assert operation.action == 'instance.years.extend_horizon'

    assert extend_horizon(framework) == []
    assert InstanceChangeOperation.objects.filter(instance_config=municipality).count() == 1


def test_a_horizon_is_never_shortened_and_locked_or_merely_attached_instances_are_left_alone() -> None:
    template = _instance_config(YearsSpec(reference=1990, min_historical=1990, max_historical=2023, target=2045, model_end=2045))
    framework = FrameworkFactory.create(template_instance=template)
    longer = _instance_config(YearsSpec(reference=1990, max_historical=2022, target=2050, model_end=2050), template=template)
    locked = _instance_config(YearsSpec(reference=1990, max_historical=2022, target=2025, model_end=2025), template=template)
    attached = _instance_config(YearsSpec(reference=1990, max_historical=2022, target=2025, model_end=2025))
    InstanceConfig.objects.filter(pk=locked.pk).update(is_locked=True)
    for municipality in (longer, locked, attached):
        FrameworkConfigFactory.create(framework=framework, instance_config=municipality)

    report = extend_horizon(framework)

    assert report == [f'{locked.identifier}: locked, left at target 2025, model_end 2025']
    assert (_years(longer).target, _years(longer).model_end) == (2050, 2050)
    assert (_years(locked).target, _years(locked).model_end) == (2025, 2025)
    # A model of its own that is only attached to the framework does not inherit the template.
    assert (_years(attached).target, _years(attached).model_end) == (2025, 2025)
