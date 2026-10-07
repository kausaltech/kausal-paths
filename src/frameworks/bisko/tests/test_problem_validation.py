from datetime import date
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from kausal_common.datasets.models import DatasetMetricValidationRule
from kausal_common.datasets.tests.factories import DataPointFactory, DatasetFactory, DatasetMetricFactory
from kausal_common.i18n.pydantic import set_i18n_context

from datasets.validation import RuleViolation
from frameworks.bisko.provisioning import reconcile_bisko_default_quality
from frameworks.bisko.validation import publish_checked_template, upgrade_checked_instance
from frameworks.tests.factories import FrameworkFactory
from nodes.defs.data_entry import DataEntryDatasetSpec, DataEntryPlacementSpec, DataEntrySectionSpec, DataEntrySpec
from nodes.defs.graph import QualityLevelKey
from nodes.defs.instance_defs import InstanceModelSpec
from nodes.instance_problems import DataEntryDefinitionError, InstanceProblems
from nodes.template_graph import publish_template_instance
from nodes.tests.factories import InstanceConfigFactory
from nodes.value_validation import ValueValidationViolation

if TYPE_CHECKING:
    from frameworks.models import Framework
    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db


@pytest.fixture
def framework() -> Framework:
    template = InstanceConfigFactory.create(name='Template', config_source='database', spec=InstanceModelSpec())
    return FrameworkFactory.create(identifier='bisko', template_instance=template)


def break_layout(instance: InstanceConfig) -> None:
    with set_i18n_context(instance.primary_language, instance.other_languages):
        spec = instance.ensure_spec()
        spec.data_entry = DataEntrySpec(
            sections=[
                DataEntrySectionSpec(
                    id=uuid4(),
                    name='Broken',
                    tables=[DataEntryPlacementSpec(id=uuid4(), node_id=uuid4(), port_id=uuid4())],
                )
            ]
        )
        instance.spec = spec
        instance.save(update_fields=['spec'])
        instance.invalidate_cache()


def test_invalid_layout_stays_saveable_but_blocks_template_publication(framework: Framework) -> None:
    template = framework.template_instance
    assert template is not None
    break_layout(template)
    before = template.revisions.count()
    with pytest.raises(DataEntryDefinitionError, match='Invalid data-entry definition'):
        publish_template_instance(template)
    with pytest.raises(ValueError, match='instance problems'):
        publish_checked_template(template)
    template.refresh_from_db()
    assert template.live_revision_id is None
    assert template.revisions.count() == before
    revision = publish_checked_template(template, ignore_problems=True)
    template.refresh_from_db()
    assert template.live_revision_id == revision.pk


@pytest.mark.parametrize('ignore_problems', [False, True])
def test_failed_pin_bump_rolls_back_unless_ignored(framework: Framework, ignore_problems: bool) -> None:
    template = framework.template_instance
    assert template is not None
    first = publish_template_instance(template)
    dependent = InstanceConfigFactory.create(
        name='Municipality', config_source='database', template_revision=first, spec=InstanceModelSpec()
    )
    break_layout(template)
    second = publish_template_instance(template, ignore_problems=True)
    if ignore_problems:
        upgrade_checked_instance(dependent, second, ignore_problems=True)
    else:
        with pytest.raises(ValueError, match='instance problems'):
            upgrade_checked_instance(dependent, second)
    dependent.refresh_from_db()
    assert dependent.template_revision_id == (second.pk if ignore_problems else first.pk)


@pytest.mark.parametrize('ignore_problems', [False, True])
def test_implicit_quality_publication_checks_problems(framework: Framework, ignore_problems: bool) -> None:
    template = framework.template_instance
    assert template is not None
    dataset = DatasetFactory.create(scope=template, identifier='reference')
    break_layout(template)
    definition = template.ensure_spec().data_entry
    assert isinstance(definition, DataEntrySpec)
    definition.sections[0].tables.append(DataEntryDatasetSpec(id=uuid4(), dataset_id=dataset.uuid))
    template.save(update_fields=['spec'])
    defaults = {'reference': QualityLevelKey(scheme='quality', level='B')}
    if ignore_problems:
        revision = reconcile_bisko_default_quality(framework, defaults, check_problems=True, ignore_problems=True)
        assert revision is not None
    else:
        with pytest.raises(ValueError, match='instance problems'):
            reconcile_bisko_default_quality(framework, defaults, check_problems=True)
    template.refresh_from_db()
    assert bool(template.live_revision_id) == ignore_problems


@pytest.mark.parametrize('ignore_problems', [False, True])
@pytest.mark.parametrize('enforcement', ['block_publish', 'block_submission'])
def test_publication_check_includes_dataset_validation(framework: Framework, ignore_problems: bool, enforcement: str) -> None:
    template = framework.template_instance
    assert template is not None
    dataset = DatasetFactory.create(scope=template, identifier='reference')
    assert dataset.schema is not None
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='Value', unit='kWh')
    DataPointFactory.create(dataset=dataset, metric=metric, date=date(2023, 1, 1), value=-1)
    DatasetMetricValidationRule.objects.create(metric=metric, rule={'kind': 'value_range', 'min': 0, 'enforcement': enforcement})
    with set_i18n_context(template.primary_language, template.other_languages):
        spec = template.ensure_spec()
        spec.data_entry = DataEntrySpec(
            sections=[
                DataEntrySectionSpec(
                    id=uuid4(),
                    name='Reference',
                    tables=[DataEntryDatasetSpec(id=uuid4(), dataset_id=dataset.uuid)],
                )
            ]
        )
        template.spec = spec
        template.save(update_fields=['spec'])
    allowed = ignore_problems or enforcement == 'block_submission'
    if allowed:
        publish_checked_template(template, ignore_problems=True)
    else:
        with pytest.raises(ValueError, match='instance problems'):
            publish_checked_template(template)
    template.refresh_from_db()
    assert bool(template.live_revision_id) == allowed


def test_pin_upgrade_allows_visible_submission_only_problems(framework: Framework, monkeypatch: pytest.MonkeyPatch) -> None:
    template = framework.template_instance
    assert template is not None
    first = publish_template_instance(template)
    dependent = InstanceConfigFactory.create(
        name='Municipality', config_source='database', template_revision=first, spec=InstanceModelSpec()
    )
    second = publish_template_instance(template)
    problems = InstanceProblems(
        constraints=(),
        datasets=[
            RuleViolation(
                rule_uuid=uuid4(),
                metric_uuid=uuid4(),
                metric='Value',
                kind='no_gaps',
                enforcement='block_submission',
                message='Inventory has gaps',
            )
        ],
        data_entry=[],
        values=[
            ValueValidationViolation(
                node_uuid=uuid4(),
                port_uuid=uuid4(),
                code='missing_required_value',
                message='Municipal input is incomplete',
                years=[2023],
                enforcement='block_submission',
            )
        ],
    )
    monkeypatch.setattr('frameworks.bisko.validation.collect_instance_problems', lambda _instance: problems)
    upgrade_checked_instance(dependent, second)
    dependent.refresh_from_db()
    assert dependent.template_revision_id == second.pk
    assert problems.messages
    assert not problems.blocking('publish').messages
    assert problems.blocking('submit').messages
