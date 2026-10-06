"""Transactional problem checks for BISKO publication and dependent draft upgrades."""

from typing import TYPE_CHECKING

from django.db import transaction

from loguru import logger

from nodes.instance_problems import collect_instance_problems
from nodes.models import InstanceConfig
from nodes.template_graph import publish_template_instance, upgrade_template_instance

if TYPE_CHECKING:
    from wagtail.models import Revision

    from kausal_common.datasets.models import Dataset


def check_instance_problems(instance: InstanceConfig, *, ignore_problems: bool = False) -> None:
    problems = collect_instance_problems(instance)
    if not problems.messages:
        return
    message = f'{instance.identifier}: instance problems:\n' + '\n'.join(f'- {item}' for item in problems.messages)
    if not ignore_problems:
        raise ValueError(message)
    logger.warning('{}\nProceeding because --ignore-problems was specified.', message)


@transaction.atomic
def publish_checked_template(
    template: InstanceConfig,
    *,
    reference_data: dict[str, Dataset] | None = None,
    ignore_problems: bool = False,
) -> Revision:
    template = InstanceConfig.objects.select_for_update().get(pk=template.pk)
    check_instance_problems(template, ignore_problems=ignore_problems)
    return publish_template_instance(template, reference_data=reference_data, ignore_problems=ignore_problems)


@transaction.atomic
def upgrade_checked_instance(instance: InstanceConfig, revision: Revision, *, ignore_problems: bool = False) -> None:
    instance = InstanceConfig.objects.select_for_update().get(pk=instance.pk)
    upgrade_template_instance(instance, revision)
    instance.refresh_from_db()
    check_instance_problems(instance, ignore_problems=ignore_problems)
