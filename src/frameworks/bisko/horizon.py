"""
The model horizon of the BISKO municipalities.

An instance's years are wholly local to it (see docs/architecture/template-inheritance.md),
so a new target or model end year in the template reaches only the municipalities activated
afterwards. `extend_horizon` moves the existing ones, and `setup_bisko` runs it every time.
"""

from frameworks.models import Framework, FrameworkConfig
from nodes.change_ops import InstanceLockedError, change_operation, record_change
from nodes.models import InstanceChangeSource, InstanceConfig

HORIZON_FIELDS = ('target', 'model_end')


def _horizon_years(instance: InstanceConfig) -> dict[str, int | None]:
    return instance.ensure_spec().years.model_dump(include=set(HORIZON_FIELDS))


def _describe(years: dict[str, int] | dict[str, int | None]) -> str:
    return ', '.join(f'{field} {years[field]}' for field in HORIZON_FIELDS if field in years)


def _later_years(instance: InstanceConfig, horizon: dict[str, int]) -> dict[str, int]:
    """Return the horizon years that lie beyond the instance's own."""
    years = instance.ensure_spec().years
    return {field: year for field, year in horizon.items() if (getattr(years, field) or 0) < year}


def extend_horizon(framework: Framework) -> list[str]:
    """
    Move the target and model end year of every municipality inheriting the template out to the template's.

    Returns one line per municipality changed.

    A year is only ever moved later, and the historical years are not touched, so no balance
    changes. Each municipality's change is one CLI operation in its change history, and its draft
    then differs from its published revision. A locked municipality is reported and left alone. A
    second run does nothing.
    """
    template = framework.template_instance
    if template is None:
        raise ValueError(f'Framework {framework.identifier} has no template instance.')
    template_years = template.ensure_spec().years
    horizon = {field: year for field in HORIZON_FIELDS if (year := getattr(template_years, field)) is not None}
    report: list[str] = []
    # Only the municipalities that inherit the template: an attached model of its own keeps its years.
    configs = FrameworkConfig.objects.filter(framework=framework, instance_config__template_revision__isnull=False)
    for config in configs.select_related('instance_config').order_by('instance_config__identifier'):
        instance = config.instance_config
        if not _later_years(instance, horizon):
            continue
        try:
            with change_operation(instance, user=None, action='instance.years.extend_horizon', source=InstanceChangeSource.CLI):
                instance.refresh_from_db(fields=['spec'])
                updates = _later_years(instance, horizon)
                before = _horizon_years(instance)
                instance.update_years(**updates)
                record_change(instance, action='instance.years.extend_horizon', before=before, after=dict(updates))
        except InstanceLockedError:
            report.append(f'{instance.identifier}: locked, left at {_describe(_horizon_years(instance))}')
            continue
        report.append(f'{instance.identifier}: {_describe(before)} -> {_describe(updates)}')
    return report
