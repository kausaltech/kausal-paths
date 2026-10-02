"""Pinned declaration inheritance, sparse municipal values, and independent upgrades."""

import json
from datetime import date
from importlib import import_module
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from django.apps import apps
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.db import connection
from django.db.models import ProtectedError
from wagtail.models import Revision

import pytest

from kausal_common.datasets.tests.factories import DataPointFactory, DatasetFactory, DatasetMetricFactory, DatasetSchemaFactory
from kausal_common.i18n.pydantic import TranslatedString

from paths.tests.graphql import PathsTestClient

from frameworks.models import FrameworkConfig
from frameworks.tests.factories import FrameworkFactory
from nodes.defs.instance_defs import InstanceModelSpec, InstanceResultExcelSpec, YearsSpec
from nodes.defs.node_defs import FormulaConfig, SimpleConfig
from nodes.defs.port_def import InputPortDef
from nodes.instance_loader import InstanceLoader
from nodes.instance_serialization import (
    DatasetMetricSource,
    InputBindingSnapshot,
    InstanceExport,
    InstanceSnapshot,
    NodePortSource,
    build_instance_snapshot,
    export_instance,
    import_instance,
)
from nodes.legacy_specs import local_spec_from_template, migrate_inherited_node_settings
from nodes.models import NodeInputPortBinding
from nodes.parameter_values import set_scenario_parameter
from nodes.scenario import Scenario, ScenarioKind
from nodes.template_graph import (
    publish_template_instance,
    replace_input_port_bindings,
    snapshot_content_hash,
    upgrade_template_instance,
)
from nodes.template_settings import InheritedNodeSettings
from nodes.tests.factories import InstanceConfigFactory, NodeConfigFactory
from params.base import ParameterOwner
from params.param import BoolParameter, NumberParameter, StringParameter
from users.tests.factories import UserFactory

if TYPE_CHECKING:
    from django.test import Client

    from nodes.models import InstanceConfig
    from users.models import User

pytestmark = pytest.mark.django_db


@pytest.fixture
def municipal() -> tuple[InstanceConfig, InstanceConfig, User]:
    spec = InstanceModelSpec(
        years=YearsSpec(reference=2020, min_historical=2020, max_historical=2022, target=2030),
        params=[
            NumberParameter(local_id='inventory_factor', value=2, is_customizable=False),
            NumberParameter(local_id='fixed_factor', value=3, owner=ParameterOwner.FRAMEWORK),
            BoolParameter(local_id='weather_correction', value=False),
            StringParameter(local_id='ags_number', value='00000000', is_customizable=False),
        ],
        scenarios=[
            Scenario(id='default', name='Default', kind=ScenarioKind.DEFAULT),
            Scenario(id='weather', name='Weather corrected', param_values={'weather_correction': True}),
        ],
    )
    template = InstanceConfigFactory.create(name='Template', owner='Test owner', config_source='database', spec=spec)
    NodeConfigFactory.create(instance=template, identifier='outcome')
    template.ensure_spec().result_excels = [InstanceResultExcelSpec(name='Balance', base_excel_url=None, node_ids=['outcome'])]
    template.save(update_fields=['spec'])
    framework = FrameworkFactory.create(template_instance=template)
    revision = publish_template_instance(template)
    municipality = InstanceConfigFactory.create(
        name='Municipality',
        owner='Test owner',
        config_source='database',
        template_revision=revision,
        spec=InstanceModelSpec(years=spec.years),
    )
    FrameworkConfig.objects.create(framework=framework, instance_config=municipality)
    return template, municipality, UserFactory.create(is_superuser=True, is_staff=True)


def test_reports_are_inherited_from_pin_and_publication_is_frozen(municipal: tuple[InstanceConfig, InstanceConfig, User]) -> None:
    template, municipality, _ = municipal
    municipality.publish_instance()
    municipality.refresh_from_db()
    old = municipality.live_revision
    assert old is not None
    old_content = old.content
    assert municipality.ensure_spec().result_excels == []
    template.nodes.update(identifier='replacement')
    template.ensure_spec().result_excels[0].node_ids = ['replacement']
    template.save(update_fields=['spec'])
    template.invalidate_cache()
    revision = publish_template_instance(template)
    assert build_instance_snapshot(municipality).spec.result_excels[0].node_ids == ['outcome']
    upgrade_template_instance(municipality, revision)
    municipality.refresh_from_db()
    assert build_instance_snapshot(municipality).spec.result_excels[0].node_ids == ['replacement']
    old.refresh_from_db()
    assert old.content == old_content
    assert municipality.live_revision_id == old.pk


def test_municipal_defaults_carry_into_weather_scenario(municipal: tuple[InstanceConfig, InstanceConfig, User]) -> None:
    _, municipality, user = municipal
    set_scenario_parameter(municipality, 'inventory_factor', 9, user=user)
    municipality.refresh_from_db()
    snapshot = build_instance_snapshot(municipality)
    assert next(p for p in snapshot.spec.params if p.local_id == 'inventory_factor').value == 9
    assert snapshot.spec.scenarios[1].param_values == {'weather_correction': True}
    context = InstanceLoader.from_snapshot(snapshot, instance_config=municipality).instance.context
    context.activate_scenario(context.scenarios['weather'])
    assert context.get_parameter('inventory_factor').value == 9
    assert context.get_parameter('weather_correction').value is True
    assert snapshot.spec.scenarios[0].param_values == {'inventory_factor': 9}
    assert municipality.ensure_spec().params == []


def test_mutation_is_sparse_and_independent_of_visitor_customization(
    municipal: tuple[InstanceConfig, InstanceConfig, User],
) -> None:
    _, municipality, user = municipal
    set_scenario_parameter(municipality, 'inventory_factor', 7, user=user)
    municipality.refresh_from_db()
    assert next(s for s in municipality.ensure_spec().scenarios if s.id == 'default').param_values == {'inventory_factor': 7}
    set_scenario_parameter(municipality, 'inventory_factor', 2, user=user)
    municipality.refresh_from_db()
    assert municipality.ensure_spec().scenarios == []
    with pytest.raises(ValidationError, match='controlled by the framework'):
        set_scenario_parameter(municipality, 'fixed_factor', 8, user=user)
    with pytest.raises(ValidationError, match='permission'):
        set_scenario_parameter(municipality, 'inventory_factor', 8, user=UserFactory.create())


def test_template_parameter_ids_cannot_be_redeclared(municipal: tuple[InstanceConfig, InstanceConfig, User]) -> None:
    _, municipality, _ = municipal
    municipality.ensure_spec().params = [NumberParameter(local_id='inventory_factor', value=7)]
    municipality.save(update_fields=['spec'])
    with pytest.raises(ValueError, match='cannot shadow'):
        build_instance_snapshot(municipality)


@pytest.mark.parametrize('change', ['remove', 'owner', 'type'])
def test_upgrade_retires_meaningless_parameter_values(
    municipal: tuple[InstanceConfig, InstanceConfig, User],
    change: str,
) -> None:
    template, municipality, user = municipal
    set_scenario_parameter(municipality, 'inventory_factor', 8, user=user)
    if change == 'remove':
        template.ensure_spec().params = [p for p in template.ensure_spec().params if p.local_id != 'inventory_factor']
    elif change == 'owner':
        template.ensure_spec().params[0].owner = ParameterOwner.FRAMEWORK
    else:
        template.ensure_spec().params[0] = StringParameter(local_id='inventory_factor', value='new type')
    template.save(update_fields=['spec'])
    template.invalidate_cache()
    revision = publish_template_instance(template)
    upgrade_template_instance(municipality, revision)
    municipality.refresh_from_db()
    assert 'inventory_factor' not in next(s for s in municipality.ensure_spec().scenarios if s.id == 'default').param_values
    assert build_instance_snapshot(municipality).spec.scenarios[0].param_values == {}


def test_migration_discards_copied_declarations_and_retains_local_values(
    municipal: tuple[InstanceConfig, InstanceConfig, User],
) -> None:
    template, municipality, _ = municipal
    base = build_instance_snapshot(template)
    copied = base.spec.model_copy(deep=True)
    copied.params[0].value = 6
    copied.params.append(StringParameter(local_id='municipal_note', value='Local addition'))
    local = local_spec_from_template(copied, base)
    assert [p.local_id for p in local.params] == ['municipal_note']
    assert local.result_excels == []
    assert len(local.scenarios) == 1
    assert next(s for s in local.scenarios if s.id == 'default').param_values == {'inventory_factor': 6}
    municipality.spec = local
    municipality.save(update_fields=['spec'])
    assert [r.model_dump(mode='json') for r in build_instance_snapshot(municipality).spec.result_excels] == [
        r.model_dump(mode='json') for r in base.spec.result_excels
    ]


def test_old_formula_snapshot_is_upgraded_without_mutating_revision_content() -> None:
    template = InstanceConfigFactory.create(name='Template', owner='Test owner', config_source='database')
    NodeConfigFactory.create(instance=template, identifier='formula_node')
    snapshot = build_instance_snapshot(template)
    data = snapshot.model_dump(mode='json')
    data['schema_version'] = 12
    data['nodes'][0]['spec']['type_config'] = {'kind': 'simple', 'node_class': 'formula.FormulaNode'}
    data['nodes'][0]['spec']['params'] = [{'type': 'string', 'local_id': 'formula', 'value': '1'}]
    data['spec']['scenarios'] = [
        {
            'id': 'default',
            'name': 'Default',
            'kind': 'default',
            'param_values': {'formula_node.formula': '1'},
        }
    ]
    upgraded = InstanceSnapshot.from_serialized_data(data)
    assert upgraded.nodes[0].spec is not None
    assert upgraded.nodes[0].spec.type_config == FormulaConfig(formula='1')
    assert upgraded.nodes[0].spec.params == []
    assert upgraded.spec.scenarios[0].param_values == {}
    assert data['nodes'][0]['spec']['params'][0]['value'] == '1'
    data['spec']['scenarios'][0]['param_values']['formula_node.formula'] = '2'
    with pytest.raises(ValueError, match='changes formula'):
        InstanceSnapshot.from_serialized_data(data)


def test_parameter_mutation_authentication_and_audit(
    municipal: tuple[InstanceConfig, InstanceConfig, User],
    client: Client,
) -> None:
    _, municipality, user = municipal
    client.force_login(user)
    gql = PathsTestClient(client)
    gql.set_instance(municipality)
    data = gql.query_data(
        """mutation ($id: ID!) {
          instanceEditor(instanceId: $id) {
            setInstanceParameter(parameterId: "inventory_factor", value: 11) {
              ... on InstanceType { id }
            }
          }
        }""",
        variables={'id': municipality.identifier},
    )
    assert data['instanceEditor']['setInstanceParameter']['id'] == municipality.identifier
    municipality.refresh_from_db()
    assert next(s for s in municipality.ensure_spec().scenarios if s.id == 'default').param_values == {'inventory_factor': 11}
    assert municipality.change_operations.filter(action='instance.parameter.set', user=user).exists()


def test_data_migration_preserves_publications_and_removes_copies(
    municipal: tuple[InstanceConfig, InstanceConfig, User],
) -> None:
    template, municipality, _ = municipal
    base = build_instance_snapshot(template)
    municipality.spec = base.spec.model_copy(deep=True)
    municipality.ensure_spec().params[0].value = 12
    municipality.save(update_fields=['spec'])
    node = template.nodes.get()
    node_spec = node.spec
    assert node_spec is not None
    node_spec.type_config = SimpleConfig(node_class='formula.FormulaNode')
    node_spec.params = [StringParameter(local_id='formula', value='1')]
    node.spec = node_spec
    node.save(update_fields=['spec'])
    revision = municipality.template_revision
    assert revision is not None
    before = revision.content
    template.ensure_spec().scenarios[0].param_values['outcome.formula'] = 'obsolete captured formula'
    template.save(update_fields=['spec'])
    migration = import_module('nodes.migrations.0082_sparse_template_specs')
    migration.migrate_specs(apps, connection.schema_editor())
    municipality.refresh_from_db()
    node = type(node).objects.with_spec().get(pk=node.pk)
    revision.refresh_from_db()
    assert municipality.ensure_spec().params == []
    assert municipality.ensure_spec().result_excels == []
    assert next(s for s in municipality.ensure_spec().scenarios if s.id == 'default').param_values == {'inventory_factor': 12}
    assert node.spec is not None
    assert node.spec.type_config == FormulaConfig(formula='1')
    assert node.spec.params == []
    template.refresh_from_db()
    assert 'outcome.formula' not in template.ensure_spec().scenarios[0].param_values
    assert revision.content == before


def test_upgrade_retires_the_local_entry_of_a_removed_scenario(
    municipal: tuple[InstanceConfig, InstanceConfig, User],
) -> None:
    template, municipality, user = municipal
    set_scenario_parameter(municipality, 'weather_correction', value=False, scenario_id='weather', user=user)
    municipality.refresh_from_db()
    municipality.ensure_spec().scenarios.append(Scenario(id='local', name='Local', param_values={'weather_correction': True}))
    municipality.save(update_fields=['spec'])
    template.ensure_spec().scenarios = [s for s in template.ensure_spec().scenarios if s.id != 'weather']
    template.save(update_fields=['spec'])
    template.invalidate_cache()
    upgrade_template_instance(municipality, publish_template_instance(template))
    municipality.refresh_from_db()
    assert 'weather' not in {s.id for s in municipality.ensure_spec().scenarios}
    # The instance's own scenario is a declaration, not an override, and stays.
    assert {s.id for s in build_instance_snapshot(municipality).spec.scenarios} == {'default', 'local'}


def test_data_migration_converts_yaml_mirrors_and_reports_the_ones_it_cannot(capsys: pytest.CaptureFixture[str]) -> None:
    mirrors = {}
    for identifier, params in (('mirrored', [StringParameter(local_id='formula', value='1')]), ('broken', [])):
        instance = InstanceConfigFactory.create(identifier=identifier, name=identifier, owner='Test', config_source='yaml')
        node = NodeConfigFactory.create(instance=instance, identifier='outcome')
        node_spec = node.spec
        assert node_spec is not None
        node_spec.type_config = SimpleConfig(node_class='formula.FormulaNode')
        node_spec.params = params
        node.spec = node_spec
        node.save(update_fields=['spec'])
        mirrors[identifier] = node
    migration = import_module('nodes.migrations.0082_sparse_template_specs')
    migration.migrate_specs(apps, connection.schema_editor())
    converted = type(mirrors['mirrored']).objects.with_spec().get(pk=mirrors['mirrored'].pk)
    assert converted.spec is not None
    assert converted.spec.type_config == FormulaConfig(formula='1')
    untouched = type(mirrors['broken']).objects.with_spec().get(pk=mirrors['broken'].pk)
    assert untouched.spec is not None
    assert untouched.spec.type_config == SimpleConfig(node_class='formula.FormulaNode')
    assert 'broken: database copy not migrated' in capsys.readouterr().out


def test_migration_retires_stale_copied_report_contents(municipal: tuple[InstanceConfig, InstanceConfig, User]) -> None:
    template, municipality, _ = municipal
    base = build_instance_snapshot(template)
    copied = base.spec.model_copy(deep=True)
    copied.result_excels[0].node_ids = ['retired_node']
    municipality.spec = local_spec_from_template(copied, base)
    municipality.save(update_fields=['spec'])
    assert municipality.ensure_spec().result_excels == []
    assert build_instance_snapshot(municipality).spec.result_excels[0].node_ids == ['outcome']


def test_sparse_scenario_resets_default_scenario_deviations_and_restores_temporary_state(
    municipal: tuple[InstanceConfig, InstanceConfig, User],
) -> None:
    template, municipality, _ = municipal
    template.ensure_spec().scenarios[0].param_values['inventory_factor'] = 4
    template.save(update_fields=['spec'])
    template.invalidate_cache()
    revision = publish_template_instance(template)
    upgrade_template_instance(municipality, revision)
    municipality.refresh_from_db()
    context = InstanceLoader.from_snapshot(build_instance_snapshot(municipality), instance_config=municipality).instance.context
    parameter = context.get_parameter('inventory_factor')
    assert parameter.value == 4
    weather = context.scenarios['weather']
    with weather.override():
        assert parameter.value == 2
    assert parameter.value == 4
    context.activate_scenario(weather)
    assert parameter.value == 2
    assert context.get_parameter('weather_correction').value is True


def test_local_parameter_addition_can_restore_its_declared_default(
    municipal: tuple[InstanceConfig, InstanceConfig, User],
) -> None:
    _, municipality, user = municipal
    municipality.ensure_spec().params.append(NumberParameter(local_id='local_factor', value=2))
    municipality.save(update_fields=['spec'])
    set_scenario_parameter(municipality, 'local_factor', 9, user=user)
    municipality.refresh_from_db()
    set_scenario_parameter(municipality, 'local_factor', 2, user=user)
    municipality.refresh_from_db()
    assert municipality.ensure_spec().scenarios == []
    assert next(p for p in build_instance_snapshot(municipality).spec.params if p.local_id == 'local_factor').value == 2


def test_legacy_node_values_move_to_sparse_defaults_and_formula_settings_retire(
    municipal: tuple[InstanceConfig, InstanceConfig, User],
) -> None:

    template, municipality, _ = municipal
    node = template.nodes.get()
    spec = node.spec
    assert spec is not None
    spec.params = [NumberParameter(local_id='multiplier', value=2)]
    node.spec = spec
    node.save(update_fields=['spec'])
    base = build_instance_snapshot(template)
    settings = [
        InheritedNodeSettings(
            node_uuid=node.uuid,
            parameter_values={'multiplier': 5, 'formula': 'stale formula'},
        )
    ]
    assert migrate_inherited_node_settings(municipality.ensure_spec(), settings, base) == []
    assert next(s for s in municipality.ensure_spec().scenarios if s.id == 'default').param_values == {'outcome.multiplier': 5}


def test_upgrade_to_a_cyclic_draft_is_inspectable_and_repairable(
    municipal: tuple[InstanceConfig, InstanceConfig, User],
) -> None:
    template, municipality, _ = municipal
    first = template.nodes.get_queryset().with_spec().get()
    second = NodeConfigFactory.create(instance=template, identifier='second')
    assert first.spec is not None
    assert second.spec is not None
    for node in (first, second):
        assert node.spec is not None
        node.spec.input_ports = [InputPortDef(id=uuid4(), unit=node.spec.output_ports[0].unit, binding_owner='instance')]
        node.save(update_fields=['spec'])
    revision = publish_template_instance(template)
    upgrade_template_instance(municipality, revision)
    municipality.refresh_from_db()
    binding = InputBindingSnapshot(
        uuid=uuid4(),
        node_id=second.uuid,
        port_id=second.spec.input_ports[0].id,
        source=NodePortSource(node_id=first.uuid, port_id=first.spec.output_ports[0].id),
    )
    replace_input_port_bindings(municipality, second.uuid, binding.port_id, [binding])
    NodeInputPortBinding.objects.create(
        instance=template,
        node=first,
        port_id=first.spec.input_ports[0].id,
        source_node=second,
        source_port_id=second.spec.output_ports[0].id,
    )
    revision = publish_template_instance(template)
    upgrade_template_instance(municipality, revision)
    municipality.refresh_from_db()
    assert any('cycle' in error for error in build_instance_snapshot(municipality).composition_errors)
    with pytest.raises(ValidationError, match='cycle'):
        municipality.publish_instance()
    replace_input_port_bindings(municipality, second.uuid, binding.port_id, None)
    assert build_instance_snapshot(municipality).composition_errors == []


def test_publication_retains_only_local_authoring_inputs(municipal: tuple[InstanceConfig, InstanceConfig, User]) -> None:
    template, municipality, user = municipal
    set_scenario_parameter(municipality, 'inventory_factor', 9, user=user)
    municipality.publish_instance()
    municipality.refresh_from_db()
    revision = municipality.live_revision
    assert revision is not None
    stored = InstanceSnapshot.from_serialized_data(revision.content['model_snapshot']['structured'], compose=False)
    assert stored.snapshot_kind == 'authored'
    assert stored.nodes == []
    assert stored.spec.params == []
    assert stored.spec.result_excels == []
    assert stored.template_revision_id == municipality.template_revision_id
    assert stored.template_content_hash
    resolved = stored.resolve()
    assert resolved.spec.is_composed
    assert len(resolved.nodes) == 1
    assert resolved.spec.scenarios[0].param_values == {'inventory_factor': 9}
    assert resolved.provenance['params/inventory_factor'].instance_uuid == template.uuid
    weather_origin = resolved.parameter_value_origin('inventory_factor', 'weather')
    assert weather_origin is not None
    assert weather_origin.instance_uuid == municipality.uuid
    assert resolved.provenance['scenarios/default/param_values/inventory_factor'].instance_uuid == municipality.uuid
    assert resolved.provenance[f'nodes/{resolved.nodes[0].uuid}'].revision_id == municipality.template_revision_id


@pytest.mark.parametrize('operation', ['save', 'update', 'bulk_update', 'bulk_create'])
def test_composed_spec_cannot_be_persisted(
    municipal: tuple[InstanceConfig, InstanceConfig, User],
    operation: str,
) -> None:
    _, municipality, _ = municipal
    municipality.spec = build_instance_snapshot(municipality).spec

    def persist() -> None:
        if operation == 'save':
            municipality.save(update_fields=['spec'])
        elif operation == 'update':
            type(municipality).objects.filter(pk=municipality.pk).update(spec=municipality.spec)
        elif operation == 'bulk_update':
            type(municipality).objects.bulk_update([municipality], ['spec'])
        else:
            type(municipality).objects.bulk_create([
                type(municipality)(identifier='composed-spec-guard', name='Guard', owner='Test', spec=municipality.spec),
            ])

    with pytest.raises(ValueError, match='Cannot store a composed spec'):
        persist()


def test_scalar_overrides_preserve_field_presence(municipal: tuple[InstanceConfig, InstanceConfig, User]) -> None:
    template, municipality, _ = municipal
    template.ensure_spec().theme_identifier = 'template-theme'
    template.ensure_spec().sample_size = 5
    template.ensure_spec().terms.enabled_label = TranslatedString('Enabled in template', default_language='en')
    template.save(update_fields=['spec'])
    revision = publish_template_instance(template)
    upgrade_template_instance(municipality, revision)
    municipality.refresh_from_db()
    local = municipality.spec
    assert local is not None
    local.theme_identifier = None
    local.sample_size = 0
    local.features.baseline_visible_in_graphs = False
    local.terms.enabled_label = None
    municipality.save(update_fields=['spec'])
    municipality.refresh_from_db()
    assert municipality.ensure_spec().model_dump(mode='json')['features'] == {'baseline_visible_in_graphs': False}
    effective = build_instance_snapshot(municipality)
    assert effective.spec.theme_identifier is None
    assert effective.spec.sample_size == 0
    assert effective.spec.features.baseline_visible_in_graphs is False
    assert effective.spec.features.show_accumulated_effects is True
    assert effective.spec.terms.enabled_label is None
    assert effective.provenance['features/baseline_visible_in_graphs'].instance_uuid == municipality.uuid
    assert effective.provenance['features/show_accumulated_effects'].instance_uuid == template.uuid


def test_restore_revision_restores_local_values_and_template_pin(
    municipal: tuple[InstanceConfig, InstanceConfig, User],
) -> None:
    template, municipality, user = municipal
    set_scenario_parameter(municipality, 'inventory_factor', 9, user=user)
    local = NodeConfigFactory.create(instance=municipality, identifier='local_node', name='Original name')
    municipality.publish_instance()
    municipality.refresh_from_db()
    revision = municipality.live_revision
    assert revision is not None
    pinned = municipality.template_revision_id
    template.ensure_spec().params[0].value = 4
    template.save(update_fields=['spec'])
    upgrade_template_instance(municipality, publish_template_instance(template))
    set_scenario_parameter(municipality, 'inventory_factor', 12, user=user)
    type(local).objects.filter(pk=local.pk).update(name='Changed name')
    extra = NodeConfigFactory.create(instance=municipality, identifier='later_node')
    municipality.restore_revision(revision)
    assert municipality.template_revision_id == pinned
    assert municipality.ensure_spec().params == []
    assert municipality.ensure_spec().result_excels == []
    assert municipality.ensure_spec().scenarios[0].param_values == {'inventory_factor': 9}
    local.refresh_from_db()
    extra.refresh_from_db()
    assert local.name == 'Original name'
    assert extra.is_stale is True
    assert {node.identifier for node in build_instance_snapshot(municipality).nodes} == {'outcome', 'local_node'}


def test_export_round_trip_keeps_template_inheritance(municipal: tuple[InstanceConfig, InstanceConfig, User]) -> None:

    template, municipality, user = municipal
    set_scenario_parameter(municipality, 'inventory_factor', 9, user=user)
    municipality.refresh_from_db()
    exported = export_instance(municipality)
    assert exported.template is not None
    assert exported.template.instance.metadata.uuid == template.uuid
    assert exported.instance.spec.params == []
    assert exported.instance.nodes == []
    loaded = InstanceExport.from_serialized_data(exported.model_dump(mode='json'))
    clone = InstanceConfigFactory.create(name='Clone', owner='Test owner', config_source='database', spec=InstanceModelSpec())
    import_instance(clone, loaded)
    assert clone.template_revision_id == municipality.template_revision_id
    assert clone.nodes.count() == 0
    assert clone.ensure_spec().params == []
    assert build_instance_snapshot(clone).spec.scenarios[0].param_values == {'inventory_factor': 9}


def test_import_bundled_template_remaps_database_revision_id(
    municipal: tuple[InstanceConfig, InstanceConfig, User],
) -> None:

    _, municipality, _ = municipal
    municipality.refresh_from_db()
    exported = export_instance(municipality)
    assert exported.template is not None
    base = exported.template.instance
    base.metadata.uuid = uuid4()
    base.metadata.identifier = 'portable-template'
    base.metadata.name = 'Portable template'
    base.nodes[0].uuid = uuid4()
    exported.instance.template_revision_id = 9999999
    exported.instance.template_content_hash = snapshot_content_hash(base)
    clone = InstanceConfigFactory.create(name='Clone', owner='Test owner', config_source='database', spec=InstanceModelSpec())
    import_instance(clone, exported)
    assert clone.template_revision_id != 9999999
    effective = build_instance_snapshot(clone)
    assert effective.nodes[0].uuid == base.nodes[0].uuid
    assert effective.provenance['params/inventory_factor'].instance_uuid == base.metadata.uuid


def test_publication_protects_old_template_revision(municipal: tuple[InstanceConfig, InstanceConfig, User]) -> None:

    template, municipality, _ = municipal
    municipality.publish_instance()
    municipality.refresh_from_db()
    old = municipality.template_revision
    assert old is not None
    template.ensure_spec().sample_size = 2
    template.save(update_fields=['spec'])
    upgrade_template_instance(municipality, publish_template_instance(template))
    with pytest.raises(ProtectedError):
        old.delete()


def test_old_flattened_snapshot_restores_sparse_authoring_inputs(
    municipal: tuple[InstanceConfig, InstanceConfig, User],
) -> None:

    _, municipality, user = municipal
    set_scenario_parameter(municipality, 'inventory_factor', 9, user=user)
    municipality.refresh_from_db()
    old_data = build_instance_snapshot(municipality).model_dump(mode='json')
    old_data['schema_version'] = 13
    old_data.pop('snapshot_kind')
    old = Revision.objects.create(
        content_type=ContentType.objects.get_for_model(type(municipality)),
        base_content_type=ContentType.objects.get_for_model(type(municipality)),
        object_id=str(municipality.pk),
        content={'model_snapshot': {'structured': old_data}},
    )
    set_scenario_parameter(municipality, 'inventory_factor', 12, user=user)
    municipality.restore_revision(cast('Revision[InstanceConfig]', old))
    assert municipality.ensure_spec().params == []
    assert municipality.ensure_spec().result_excels == []
    assert municipality.nodes.count() == 0
    assert build_instance_snapshot(municipality).spec.scenarios[0].param_values == {'inventory_factor': 9}


def test_old_override_container_is_migrated_with_scalar_presence(
    municipal: tuple[InstanceConfig, InstanceConfig, User],
) -> None:

    _, municipality, _ = municipal
    data = municipality.ensure_spec().model_dump(mode='json')
    data.update(features={'baseline_visible_in_graphs': True, 'show_accumulated_effects': True}, sample_size=0)
    data['overrides'] = {
        'features': {'baseline_visible_in_graphs': False},
        'theme_identifier': 'local-theme',
        'scenarios': {'default': {'param_values': {'inventory_factor': 9}, 'parameter_types': {'inventory_factor': 'number'}}},
    }
    with connection.cursor() as cursor:
        cursor.execute('UPDATE nodes_instanceconfig SET spec = %s::jsonb WHERE id = %s', (json.dumps(data), municipality.pk))
    migration = import_module('nodes.migrations.0083_authored_template_snapshots')
    migration.migrate_authoring_inputs(apps, connection.schema_editor())
    municipality.refresh_from_db()
    stored = municipality.ensure_spec().model_dump(mode='json')
    assert 'overrides' not in stored
    assert 'sample_size' not in stored
    assert stored['features'] == {'baseline_visible_in_graphs': False}
    assert stored['theme_identifier'] == 'local-theme'
    assert build_instance_snapshot(municipality).spec.scenarios[0].param_values == {'inventory_factor': 9}


def test_export_uses_pinned_template_dataset_body_and_import_remaps_payload_revision(
    municipal: tuple[InstanceConfig, InstanceConfig, User],
) -> None:

    template, municipality, _ = municipal
    node = template.nodes.get_queryset().with_spec().get()
    assert node.spec is not None
    unit = str(node.spec.output_ports[0].unit)
    port = InputPortDef(id=uuid4(), unit=node.spec.output_ports[0].unit)
    node.spec.input_ports = [port]
    node.save(update_fields=['spec'])
    schema = DatasetSchemaFactory.create()
    metric = DatasetMetricFactory.create(schema=schema, name='Value', unit=unit)
    dataset = DatasetFactory.create(
        schema=schema,
        scope_content_type=ContentType.objects.get_for_model(type(template)),
        scope_id=template.pk,
        identifier='reference-input',
    )
    point = DataPointFactory.create(dataset=dataset, metric=metric, date=date(2020, 1, 1), value=42)
    NodeInputPortBinding.objects.create(instance=template, node=node, port_id=port.id, dataset=dataset, metric=metric)
    upgrade_template_instance(municipality, publish_template_instance(template))
    municipality.refresh_from_db()
    type(point).objects.filter(pk=point.pk).update(value=99)
    exported = export_instance(municipality)
    assert exported.template is not None
    body = exported.template.datasets[0]
    assert body.data is not None
    assert body.data['data'][0]['Value'] == 42
    base = exported.template.instance
    base.metadata.uuid = uuid4()
    base.metadata.identifier = 'portable-data-template'
    base.metadata.name = 'Portable data template'
    base.nodes[0].uuid = uuid4()
    base.datasets[0] = base.datasets[0].model_copy(update={'id': uuid4()})
    base.dataset_revisions[0] = base.dataset_revisions[0].model_copy(update={'dataset_uuid': base.datasets[0].id})
    binding = base.bindings[0]
    assert isinstance(binding.source, DatasetMetricSource)
    base.bindings[0] = binding.model_copy(
        update={
            'node_id': base.nodes[0].uuid,
            'source': binding.source.model_copy(update={'dataset_uuid': base.datasets[0].id}),
        }
    )
    content_hash = snapshot_content_hash(base)
    exported.instance.template_content_hash = content_hash
    exported.instance.template_revision_id = 9999999
    clone = InstanceConfigFactory.create(
        name='Data clone', owner='Test owner', config_source='database', spec=InstanceModelSpec()
    )
    import_instance(clone, exported)
    effective = build_instance_snapshot(clone)
    pin = effective.dataset_revisions[0]
    assert pin.revision_id != base.dataset_revisions[0].revision_id
    assert Revision.objects.get(pk=pin.revision_id).content['data']['data'][0]['Value'] == 42
    assert effective.template_content_hash == content_hash
    assert isinstance(effective.bindings[0].source, DatasetMetricSource)
    assert effective.bindings[0].source.dataset_revision == pin.revision_id


def test_export_import_retains_inherited_sources_for_local_nodes(
    municipal: tuple[InstanceConfig, InstanceConfig, User],
) -> None:
    template, municipality, _ = municipal
    shared = template.nodes.get_queryset().with_spec().get()
    local = NodeConfigFactory.create(instance=municipality, identifier='local_target')
    assert local.spec is not None
    assert shared.spec is not None
    port = InputPortDef(id=uuid4(), unit=shared.spec.output_ports[0].unit, binding_owner='instance')
    local.spec.input_ports = [port]
    local.save(update_fields=['spec'])
    NodeInputPortBinding.objects.create(
        instance=municipality, node=local, port_id=port.id, source_node=shared, source_port_id=shared.spec.output_ports[0].id
    )
    exported = export_instance(municipality)
    clone = InstanceConfigFactory.create(
        name='Local edge clone', owner='Test owner', config_source='database', spec=InstanceModelSpec()
    )
    import_instance(clone, exported)
    effective = build_instance_snapshot(clone)
    assert len(effective.bindings) == 1
    binding = effective.bindings[0]
    assert isinstance(binding.source, NodePortSource)
    assert binding.source.node_id == shared.uuid
    own = clone.nodes.get()
    assert binding.node_id == own.uuid
    assert own.uuid != local.uuid
    assert clone.binding_overrides.get().node_uuid == own.uuid


def test_node_owned_dataset_round_trip_preserves_local_ownership(
    municipal: tuple[InstanceConfig, InstanceConfig, User],
) -> None:
    _, municipality, _ = municipal
    local = NodeConfigFactory.create(instance=municipality, identifier='dataset_owner')
    assert local.spec is not None
    port = InputPortDef(id=uuid4(), unit=local.spec.output_ports[0].unit)
    local.spec.input_ports = [port]
    local.save(update_fields=['spec'])
    schema = DatasetSchemaFactory.create()
    metric = DatasetMetricFactory.create(schema=schema, name='Value', unit=str(port.unit))
    dataset = DatasetFactory.create(
        schema=schema,
        scope_content_type=ContentType.objects.get_for_model(type(local)),
        scope_id=local.pk,
        identifier='owned-input',
    )
    DataPointFactory.create(dataset=dataset, metric=metric, date=date(2020, 1, 1), value=13)
    NodeInputPortBinding.objects.create(instance=municipality, node=local, port_id=port.id, dataset=dataset, metric=metric)
    exported = export_instance(municipality)
    assert any(item.identifier == 'owned-input' for item in exported.datasets)
    clone = InstanceConfigFactory.create(
        name='Owner clone', owner='Test owner', config_source='database', spec=InstanceModelSpec()
    )
    import_instance(clone, exported)
    own = clone.nodes.get()
    effective = build_instance_snapshot(clone)
    copied = next(item for item in effective.nodes if item.uuid == own.uuid)
    assert len(copied.datasets) == 1
    assert copied.datasets[0].identifier == 'owned-input'
    assert len(effective.bindings) == 1


def test_local_scenario_values_preserve_framework_authored_values(
    municipal: tuple[InstanceConfig, InstanceConfig, User],
) -> None:
    template, municipality, user = municipal
    template.ensure_spec().scenarios[0].param_values['fixed_factor'] = 8
    template.save(update_fields=['spec'])
    upgrade_template_instance(municipality, publish_template_instance(template))
    set_scenario_parameter(municipality, 'inventory_factor', 9, user=user)
    municipality.refresh_from_db()
    effective = build_instance_snapshot(municipality)
    assert effective.composition_errors == []
    assert effective.spec.scenarios[0].param_values == {'fixed_factor': 8, 'inventory_factor': 9}
    origin = effective.parameter_value_origin('fixed_factor', 'default')
    assert origin is not None
    assert origin.instance_uuid == template.uuid


def test_invalid_local_value_does_not_claim_effective_value_provenance(
    municipal: tuple[InstanceConfig, InstanceConfig, User],
) -> None:
    template, municipality, user = municipal
    set_scenario_parameter(municipality, 'inventory_factor', 9, scenario_id='weather', user=user)
    parameter = next(item for item in template.ensure_spec().params if item.local_id == 'inventory_factor')
    assert isinstance(parameter, NumberParameter)
    parameter.max_value = 5
    template.save(update_fields=['spec'])
    upgrade_template_instance(municipality, publish_template_instance(template))
    municipality.refresh_from_db()
    effective = build_instance_snapshot(municipality)
    assert effective.composition_errors
    assert 'inventory_factor' not in effective.spec.scenarios[1].param_values
    origin = effective.parameter_value_origin('inventory_factor', 'weather')
    assert origin is not None
    assert origin.instance_uuid == template.uuid


def test_imported_edition_does_not_publish_an_existing_template_draft(
    municipal: tuple[InstanceConfig, InstanceConfig, User],
) -> None:
    _, municipality, _ = municipal
    exported = export_instance(municipality)
    assert exported.template is not None
    existing = InstanceConfigFactory.create(
        name='Unpublished template draft',
        owner='Test owner',
        config_source='database',
        spec=InstanceModelSpec(sample_size=17),
        live=False,
    )
    assert existing.live_revision_id is None
    exported.template.instance.metadata.uuid = existing.uuid
    exported.instance.template_content_hash = snapshot_content_hash(exported.template.instance)
    clone = InstanceConfigFactory.create(
        name='Edition clone', owner='Test owner', config_source='database', spec=InstanceModelSpec()
    )
    import_instance(clone, exported)
    existing.refresh_from_db()
    assert existing.live_revision_id is None
    assert existing.live is False
    assert existing.ensure_spec().sample_size == 17
    assert existing.nodes.count() == 0
    assert build_instance_snapshot(clone).nodes
