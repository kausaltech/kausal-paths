"""Transactional problem checks for BISKO publication and dependent draft upgrades."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from django.db import transaction

from loguru import logger

from kausal_common.datasets.models import Dataset

from frameworks.bisko.default_sources import POPULATION_DATASET
from frameworks.bisko.defaults import has_own_population, seed_population_from_pin
from frameworks.bisko.quality import link_grades_of_converting_instance
from frameworks.conversion import convert_to_framework
from frameworks.identity import ensure_municipal_organization
from nodes.instance_problems import collect_instance_problems
from nodes.models import InstanceConfig, NodeConfig
from nodes.template_graph import publish_template_instance, template_snapshot, upgrade_template_instance

if TYPE_CHECKING:
    from wagtail.models import Revision

    from frameworks.models import Framework
    from nodes.data_entry import EntryProblem


def publish_problem_messages(instance: InstanceConfig) -> list[str]:
    return collect_instance_problems(instance).blocking('publish').messages


def check_instance_problems(instance: InstanceConfig, *, ignore_problems: bool = False, known: set[str] | None = None) -> None:
    """
    Refuse when the instance has problems that block publication.

    `known` are problems the instance had before the operation being checked; they are the
    instance's own unfinished work, not something the operation caused, and do not refuse it.
    The refusal names the node and dataset of each data-entry problem, since the message alone
    ("Input data has not been imported") does not say which input to fix.
    """
    problems = collect_instance_problems(instance).blocking('publish')
    known = known or set()
    lines = [
        message
        for message in [
            *(p.message for p in problems.constraints),
            *(p.message for p in problems.datasets),
            *(p.message for p in problems.values),
        ]
        if message not in known
    ]
    entry = [problem for problem in problems.data_entry if problem.message not in known]
    lines.extend(_describe_entry_problems(entry))
    if not lines:
        return
    message = f'{instance.identifier}: instance problems:\n' + '\n'.join(f'- {item}' for item in lines)
    if not ignore_problems:
        raise ValueError(message)
    logger.warning('{}\nProceeding because --ignore-problems was specified.', message)


def _describe_entry_problems(problems: list[EntryProblem]) -> list[str]:
    nodes = dict(NodeConfig.objects.filter(uuid__in={p.node_id for p in problems if p.node_id}).values_list('uuid', 'identifier'))
    datasets = dict(
        Dataset.objects.filter(uuid__in={p.dataset_id for p in problems if p.dataset_id}).values_list('uuid', 'identifier')
    )
    lines = []
    for problem in problems:
        where = [str(nodes.get(problem.node_id, problem.node_id))] if problem.node_id else []
        if problem.dataset_id:
            where.append(str(datasets.get(problem.dataset_id) or problem.dataset_id))
        line = f'{problem.message} ({", ".join(where)})' if where else problem.message
        if line not in lines:
            lines.append(line)
    return lines


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
    has_population = POPULATION_DATASET in {dataset.identifier for dataset in template_snapshot(instance).all_datasets()}
    if has_population and seed_population_from_pin(instance) is None:
        logger.info('{}: the population series at the pin does not cover this municipality', instance.identifier)
    check_instance_problems(instance, ignore_problems=ignore_problems, known=known)


@dataclass(frozen=True)
class Conversion:
    result: dict[str, int | list[int] | None]
    organization: str | None
    """The name of the municipality's organization, if its AGS identifies one."""
    population_years: int | None
    """
    Years of population defaults seeded, or None if there was nothing to seed: the municipality
    has no population input of its own, no AGS, or an AGS the series does not cover.
    """


@transaction.atomic
def convert_checked_instance(
    instance: InstanceConfig, framework: Framework, revision: Revision, *, ignore_problems: bool = False
) -> Conversion:
    """
    Convert a copied model to the published template, then refuse if the result cannot be published.

    Between the conversion and the check, the municipality joins the organization its AGS names
    and gets the population series at the template's pin, as a template upgrade does: the
    population input a converted model receives is otherwise empty. The series covers only
    some Länder, so a converted city outside them enters its population itself.

    Before the conversion, the city's numeric quality columns become grades of the framework's
    scale, which is the form the template's datasets have.
    """
    link_grades_of_converting_instance(instance, framework)
    result = convert_to_framework(instance, framework, revision)
    instance.refresh_from_db()
    organization = ensure_municipal_organization(instance)
    population_years = None
    if organization is not None and has_own_population(instance):
        population_years = seed_population_from_pin(instance)
    check_instance_problems(instance, ignore_problems=ignore_problems)
    return Conversion(result, organization.name if organization is not None else None, population_years)
