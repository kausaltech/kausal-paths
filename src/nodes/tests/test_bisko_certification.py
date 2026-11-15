"""Keep the certification model frozen while its framework successor evolves."""

from copy import deepcopy
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from uuid import uuid4

from django.conf import settings
from django.db import connection

import polars as pl
import pytest

from common.validation import ValidationRuleEnforcement
from datasets.validation import evaluate_rule
from datasets.validation_rules import RequiredCombinationsRule
from nodes.bisko_certification import FROZEN_MODULE_INSTANCES, certification_yaml
from nodes.instance_loader import InstanceYAMLConfig
from nodes.instance_parser import InstanceParseError, parse_instance_snapshot
from nodes.models import make_empty_instance_spec
from nodes.tests.factories import InstanceConfigFactory
from params.param import BoolParameter

pytestmark = pytest.mark.django_db

# The reviewed main revision d5ed3448, with municipal overlap disabled in bisko.
# Changing the certification model must be deliberate.
# Re-recorded 2026-11-15 after two changes judged not to affect the certification: comments
# pointing to setup material that moved from data/ to paths-data (all three files), and bisko's
# weather-correction label set to the wording of pruefstadt-bisko, the instance certified as 1.3.
CERTIFICATION_FILES = {
    'bisko.yaml': '79cbdd845b7b1374ca008325bf7822ea84d549cde13149cc27608f668f22440f',
    'pruefstadt-bisko.yaml': '94c022d7291104b834c52ff77acf9d62743a3b715c60bfd4ad64499b99e0a431',
    'modules/bisko/model.yaml': '4074327559ab76f1a1ce9d3fa08f9736380f9f19f686091ba41ab9ebb14426fc',
}


@pytest.mark.parametrize('identifier', ['bisko', 'pruefstadt-bisko', 'other-instance'])
def test_overlap_migration_preserves_other_parameters_and_instances(identifier: str) -> None:
    parameter_id = 'municipal_facilities_included_in_commerce'
    spec = make_empty_instance_spec().model_copy(
        update={
            'params': [BoolParameter(local_id=parameter_id, value=True), BoolParameter(local_id='other', value=True)],
        }
    )
    instance = InstanceConfigFactory.create(identifier=identifier, name='Migration test', config_source='database', spec=spec)
    expected = spec.model_dump(mode='json')
    if identifier in ('bisko', 'pruefstadt-bisko'):
        expected['params'][0]['value'] = False
    before_cache = instance.cache_invalidated_at
    migration = import_module('nodes.migrations.0089_disable_certification_municipal_overlap')

    migration.forwards(None, connection.schema_editor())
    instance.refresh_from_db()
    assert instance.spec is not None
    assert instance.spec.model_dump(mode='json') == expected
    assert (instance.cache_invalidated_at != before_cache) == (identifier in ('bisko', 'pruefstadt-bisko'))
    after_cache = instance.cache_invalidated_at

    migration.forwards(None, connection.schema_editor())
    instance.refresh_from_db()
    assert instance.spec.model_dump(mode='json') == expected
    assert instance.cache_invalidated_at == after_cache


@pytest.mark.parametrize(('filename', 'digest'), CERTIFICATION_FILES.items())
def test_certification_yaml_remains_at_the_reviewed_revision(filename: str, digest: str) -> None:
    assert sha256((Path(settings.BASE_DIR) / 'configs' / filename).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize(('identifier', 'overlap'), [('bisko', False), ('pruefstadt-bisko', False)])
def test_certification_keeps_each_sector_requirement_and_its_enforcement(identifier: str, overlap: bool) -> None:
    config = InstanceYAMLConfig.load_for_entrypoint(Path(settings.BASE_DIR) / 'configs' / f'{identifier}.yaml')
    assert config.data is not None
    original = deepcopy(config.data)
    instance_uuid = uuid4()
    snapshot = parse_instance_snapshot(config.data, instance_uuid=instance_uuid)
    assert config.data == original
    parameter = next(p for p in snapshot.spec.params if p.local_id == 'municipal_facilities_included_in_commerce')
    assert parameter.value is overlap
    dataset = next(d for d in snapshot.datasets if d.identifier in ('kommune/endenergieverbrauch', 'schwerin/final_energy'))
    assert dataset.shape_id is not None
    shape = next(s for s in snapshot.spec.shapes if s.uuid == dataset.shape_id)
    members = {c.uuid: c for c in shape.combinations}
    rules = [
        v.rule for metric in dataset.metrics for v in metric.validation_rules if isinstance(v.rule, RequiredCombinationsRule)
    ]
    (rule,) = rules
    assert rule.enforcement == ValidationRuleEnforcement.BLOCK_PUBLISH
    assert len(rule.groups) == 21
    assert all(len(group.combinations) == 1 for group in rule.groups)
    required = [members[group.combinations[0]].categories for group in rule.groups]
    assert {c['sector'] for c in required} == {'private_households', 'industry', 'commerce_trade_services'}
    assert len({tuple(sorted(c.items())) for c in required}) == 21
    assert not shape.required  # Requirements stay on the original metric, with their original tier.
    # Explicit zero satisfies a cell; a value elsewhere in the same carrier does not replace it.
    frame = pl.DataFrame([{'Year': 2023, 'Value': 0.0, **dict(cell)} for cell in required])
    coordinates = {key: dict(member.categories) for key, member in members.items()}
    rule_id, metric_id = uuid4(), uuid4()
    assert not evaluate_rule(rule, rule_id, metric_id, 'Value', frame, shape.dimensions, coordinates)
    (missing,) = evaluate_rule(rule, rule_id, metric_id, 'Value', frame.slice(1), shape.dimensions, coordinates)
    assert missing.categories == required[0]
    assert missing.requirement_group == rule.groups[0].id
    assert missing.enforcement == ValidationRuleEnforcement.BLOCK_PUBLISH
    assert missing.years == [2023]
    again = parse_instance_snapshot(config.data, instance_uuid=instance_uuid)
    assert again.spec.shapes == snapshot.spec.shapes


def test_legacy_domains_are_not_enabled_for_other_models() -> None:
    config = {
        'id': 'ordinary',
        'name': 'Ordinary',
        'owner': 'Test',
        'default_language': 'en',
        'reference_year': 2020,
        'minimum_historical_year': 2020,
        'target_year': 2030,
        'datasets': [{'id': 'input', 'category_domain': {'mode': 'open', 'combinations': []}}],
    }
    with pytest.raises(ValueError, match='frozen BISKO module'):
        certification_yaml(config, uuid4())
    with pytest.raises(InstanceParseError, match='category_domain'):
        parse_instance_snapshot(config, instance_uuid=uuid4())


def test_frozen_module_instances_are_exactly_its_includers() -> None:
    """The parser translates the frozen syntax by instance id, so the list must follow the configs."""
    configs = Path(settings.BASE_DIR) / 'configs'
    includers = {
        path.stem for path in configs.glob('*.yaml') if '- file: modules/bisko/model.yaml' in path.read_text(encoding='utf-8')
    }
    assert includers == FROZEN_MODULE_INSTANCES


def test_framework_template_has_an_independent_entrypoint_and_no_legacy_correction() -> None:
    config = InstanceYAMLConfig.load_for_entrypoint(Path(settings.BASE_DIR) / 'configs/bisko-base.yaml')
    assert config.data is not None
    assert config.data['id'] == 'bisko-base'
    snapshot = parse_instance_snapshot(config.data, instance_uuid=uuid4())
    assert all(p.local_id != 'municipal_facilities_included_in_commerce' for p in snapshot.spec.params)
    assert snapshot.spec.data_entry is not None
    assert snapshot.spec.shapes
    assert all(not (shape.identifier or '').startswith('certification/') for shape in snapshot.spec.shapes)
