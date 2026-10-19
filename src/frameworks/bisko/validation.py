"""Transactional problem checks for BISKO publication and dependent draft upgrades."""

from typing import TYPE_CHECKING

from django.db import transaction

from loguru import logger

from frameworks.bisko.default_sources import POPULATION_DATASET
from frameworks.bisko.defaults import seed_population_from_pin
from nodes.instance_problems import collect_instance_problems
from nodes.models import InstanceConfig
from nodes.template_graph import publish_template_instance, template_snapshot, upgrade_template_instance

if TYPE_CHECKING:
    from wagtail.models import Revision

    from kausal_common.datasets.models import Dataset


def publish_problem_messages(instance: InstanceConfig) -> list[str]:
    return collect_instance_problems(instance).blocking('publish').messages


def check_instance_problems(instance: InstanceConfig, *, ignore_problems: bool = False, known: set[str] | None = None) -> None:
    """
    Refuse when the instance has problems that block publication.

    `known` are problems the instance had before the operation being checked; they are the
    instance's own unfinished work, not something the operation caused, and do not refuse it.
    """
    messages = [item for item in publish_problem_messages(instance) if item not in (known or set())]
    if not messages:
        return
    message = f'{instance.identifier}: instance problems:\n' + '\n'.join(f'- {item}' for item in messages)
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
    # An upgrade must not make a municipality unpublishable, but data it has not finished
    # entering -- a value still without its grade, say -- is not the upgrade's to refuse.
    known = set(publish_problem_messages(instance))
    upgrade_template_instance(instance, revision)
    instance.refresh_from_db()
    # The upgrade gives a new population input an empty copy; the series at the pin fills it.
    if POPULATION_DATASET in {dataset.identifier for dataset in template_snapshot(instance).all_datasets()}:
        seed_population_from_pin(instance)
    check_instance_problems(instance, ignore_problems=ignore_problems, known=known)
