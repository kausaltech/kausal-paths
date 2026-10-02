from datetime import date
from io import StringIO
from typing import TYPE_CHECKING
from unittest.mock import patch
from uuid import uuid4

from django.contrib.contenttypes.models import ContentType
from django.core.management import call_command
from django.core.management.base import CommandError
from django.utils import timezone

import polars as pl
import pytest

from kausal_common.datasets.models import Dataset, DatasetSchemaScope, DimensionScope
from kausal_common.datasets.tests.factories import (
    DataPointFactory,
    DatasetFactory,
    DatasetMetricFactory,
    DatasetSchemaDimensionFactory,
    DatasetSchemaFactory,
    DimensionCategoryFactory,
    DimensionFactory,
)
from kausal_common.people.models import ObjectRole

from paths.tests.graphql import PathsTestClient

from frameworks.bisko.activation import ActivationError, activate_bisko_municipality
from frameworks.bisko.provisioning import GERMAN_ORGANIZATION_CLASSES, setup_bisko
from frameworks.bisko.weather import HEATING_SECTORS, NEUTRAL_SECTORS, WEATHER_DATASET
from frameworks.conversion import share_template_catalogue
from frameworks.models import (
    Framework,
    FrameworkConfig,
    FrameworkOrganizationRoot,
    OrganizationAccessGrant,
    OrganizationAccessGrantEvent,
    Submission,
)
from frameworks.organization_access import accessible_organizations, user_can_access_organization
from frameworks.population import population_aggregates, replace_population_projection
from frameworks.roles import framework_admin_role
from frameworks.tests.factories import FrameworkConfigFactory
from nodes.defs.instance_defs import YearsSpec
from nodes.defs.port_def import InputPortDef
from nodes.instance_serialization import DatasetMetricSource, build_instance_snapshot
from nodes.membership import retention_date
from nodes.models import InstanceMemberAssignment, NodeInputPortBinding
from nodes.template_graph import publish_template_instance, upgrade_template_instance
from nodes.tests.factories import InstanceConfigFactory, NodeConfigFactory
from nodes.units import unit_registry
from orgs.import_bkg import import_bkg_organizations
from orgs.models import Namespace, Organization, OrganizationClass, OrganizationIdentifier
from orgs.tests.factories import OrganizationFactory
from params.param import StringParameter
from users.models import User
from users.tests.factories import UserFactory

if TYPE_CHECKING:
    from pathlib import Path

    from django.test import Client

pytestmark = pytest.mark.django_db


def snapshot(tmp_path: Path) -> Path:
    rows: list[tuple[str, str | None, str | None, str, str, str]] = []
    for state in ('03', '07', '12'):
        rows.append((state, None, None, f'State {state}', 'de_state', 'state'))
        for district_suffix in ('001', '002'):
            district = state + district_suffix
            rows.append((district, None, state, f'District {district}', 'de_district', 'district'))
            for municipality_suffix in ('001', '002'):
                ags = district + municipality_suffix
                ars = district + '0000' + municipality_suffix
                rows.append((ars, ags, district, f'Municipality {ags}', 'de_municipality', 'municipality'))
    path = tmp_path / 'organizations.parquet'
    pl.DataFrame([
        {
            'ars': ars,
            'ags': ags,
            'nuts3': f'DE{ars[:2]}{ars[4]}' if len(ars) == 5 else (f'DE{parent[:2]}{parent[4]}' if parent else None),
            'parent_ars': parent,
            'name': name,
            'classification_identifier': classification,
            'administrative_level': level,
            'primary_language': 'de',
            'primary_language_lowercase': 'de',
            'source_vintage': date(2024, 12, 31),
        }
        for ars, ags, parent, name, classification, level in rows
    ]).write_parquet(path)
    return path


def publish_bisko_template(framework: Framework) -> None:
    template = framework.template_instance
    assert template is not None
    spec = template.ensure_spec()
    spec.years = YearsSpec(reference=2020, min_historical=2010, max_historical=2023, target=2035)
    template.spec = spec
    template.save(update_fields=['spec'])
    node = NodeConfigFactory.create(instance=template, identifier='bisko_shared')
    node.refresh_from_db()
    assert node.spec is not None
    local_port = InputPortDef(id=uuid4(), identifier='local', unit=unit_registry.parse_units('kt/a'), binding_owner='instance')
    reference_port = InputPortDef(id=uuid4(), identifier='reference', unit=unit_registry.parse_units('kt/a'))
    node.spec.input_ports = [local_port, reference_port]
    node.save(update_fields=['spec'])
    for identifier, port in (('kommune/testeingabe', local_port), ('de/reference', reference_port)):
        dataset = DatasetFactory.create(scope=template, identifier=identifier)
        assert dataset.schema is not None
        DatasetSchemaScope.objects.create(
            schema=dataset.schema, scope_content_type=ContentType.objects.get_for_model(framework), scope_id=framework.pk
        )
        metric = DatasetMetricFactory.create(schema=dataset.schema, name='Value', unit='kt/a')
        DataPointFactory.create(dataset=dataset, metric=metric, date=date(2023, 1, 1), value=42)
        NodeInputPortBinding.objects.create(instance=template, node=node, port_id=port.id, dataset=dataset, metric=metric)
    template.invalidate_cache()
    publish_template_instance(template)


def test_catalogue_and_import_reuse_existing_municipality(tmp_path: Path) -> None:
    InstanceConfigFactory.create(identifier='bisko', name='BISKO')
    framework = setup_bisko()
    assert set(OrganizationClass.objects.values_list('identifier', flat=True)) >= {
        identifier for identifier, _ in GERMAN_ORGANIZATION_CLASSES
    }
    assert {'ars', 'ags', 'nuts3'} <= set(Namespace.objects.values_list('identifier', flat=True))
    instance = InstanceConfigFactory.create(name='Existing municipality', config_source='database')
    original = instance.organization
    initial_count = Organization.objects.count()
    OrganizationIdentifier.objects.create(
        namespace=Namespace.objects.get(identifier='ags'), organization=original, identifier='03001001'
    )
    source = snapshot(tmp_path)
    first = import_bkg_organizations(source, framework=framework)
    assert first.created == 20
    assert first.reused == 1
    assert first.moved == 1
    assert first.roots_attached == 3
    original.refresh_from_db()
    parent = original.get_parent()
    assert parent is not None
    assert parent.identifiers.get(namespace__identifier='ars').identifier == '03001'
    assert original.identifiers.get(namespace__identifier='ars').identifier == '030010000001'
    assert instance.organization_id == original.pk
    assert not FrameworkConfig.objects.filter(instance_config=instance).exists()
    assert import_bkg_organizations(source, framework=framework).created == 0
    assert Organization.objects.count() == initial_count + 20


def test_setup_requires_nuts3_after_organizations_are_imported(tmp_path: Path) -> None:
    InstanceConfigFactory.create(identifier='bisko', name='BISKO')
    framework = setup_bisko()
    import_bkg_organizations(snapshot(tmp_path), framework=framework)
    district = OrganizationIdentifier.objects.get(namespace__identifier='ars', identifier='12001').organization
    district.identifiers.get(namespace__identifier='nuts3').delete()

    with pytest.raises(ValueError, match='Regenerate the BKG Parquet'):
        setup_bisko()


def test_import_rejects_old_parquet_without_nuts3(tmp_path: Path) -> None:
    InstanceConfigFactory.create(identifier='bisko', name='BISKO')
    framework = setup_bisko()
    source = snapshot(tmp_path)
    pl.read_parquet(source).drop('nuts3').write_parquet(source)

    with pytest.raises(ValueError, match='Regenerate the staged Parquet'):
        import_bkg_organizations(source, framework=framework)
    assert not OrganizationIdentifier.objects.filter(namespace__identifier='ars').exists()


def test_setup_reconciles_existing_municipal_nuts_code_once(tmp_path: Path) -> None:
    InstanceConfigFactory.create(identifier='bisko', name='BISKO')
    framework = setup_bisko()
    import_bkg_organizations(snapshot(tmp_path), framework=framework)
    municipality = OrganizationIdentifier.objects.get(namespace__identifier='ags', identifier='12001001').organization
    instance = InstanceConfigFactory.create(name='Municipality', organization=municipality, config_source='database')
    spec = instance.ensure_spec()
    spec.params.append(StringParameter(local_id='nuts_code', value='DEA11'))
    instance.spec = spec
    instance.save(update_fields=['spec'])
    FrameworkConfigFactory.create(framework=framework, instance_config=instance)

    setup_bisko()
    instance.refresh_from_db()
    assert next(param.value for param in instance.spec.params if param.local_id == 'nuts_code') == 'DE121'

    with patch.object(type(instance), 'save', side_effect=AssertionError('InstanceConfig.save called')):
        setup_bisko()


def test_repeat_import_only_creates_missing_nuts3_identifiers(tmp_path: Path) -> None:
    InstanceConfigFactory.create(identifier='bisko', name='BISKO')
    framework = setup_bisko()
    source = snapshot(tmp_path)
    import_bkg_organizations(source, framework=framework)
    district = OrganizationIdentifier.objects.get(namespace__identifier='ars', identifier='12001').organization
    OrganizationIdentifier.objects.get(namespace__identifier='nuts3', organization=district).delete()

    with patch.object(Organization, 'save', side_effect=AssertionError('Organization.save called')):
        result = import_bkg_organizations(source, framework=framework)

    assert result.created == result.moved == 0
    assert result.identifiers_created == 1
    assert district.identifiers.get(namespace__identifier='nuts3').identifier == 'DE121'


def test_nine_delegated_accounts_cover_only_their_subtrees(tmp_path: Path) -> None:
    InstanceConfigFactory.create(identifier='bisko', name='BISKO')
    framework = setup_bisko()
    import_bkg_organizations(snapshot(tmp_path), framework=framework)
    ars = Namespace.objects.get(identifier='ars')

    def org(code: str) -> Organization:
        return OrganizationIdentifier.objects.get(namespace=ars, identifier=code).organization

    for state in ('03', '07', '12'):
        own_state = org(state)
        own_district = org(state + '001')
        own_municipality = org(state + '0010000001')
        neighboring_district = org(state + '002')
        neighboring_municipality = org(state + '0010000002')
        other_state = org('07' if state == '03' else '03')
        for level, assigned, expected_count in (
            ('state', own_state, 7),
            ('district', own_district, 3),
            ('municipality', own_municipality, 1),
        ):
            user = UserFactory.create(username=f'bisko-{state}-{level}')
            OrganizationAccessGrant.objects.create(framework=framework, organization=assigned, user=user, role=ObjectRole.EDITOR)
            assert accessible_organizations(framework, user).count() == expected_count
            assert user_can_access_organization(framework, user, assigned, action='change')
            assert not user_can_access_organization(framework, user, other_state)
            if level != 'state':
                assert not user_can_access_organization(framework, user, own_state)
            if level == 'municipality':
                assert not user_can_access_organization(framework, user, neighboring_municipality)
            if level == 'district':
                assert not user_can_access_organization(framework, user, neighboring_district)


def test_provision_bisko_test_accounts_is_repeatable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: PLR0915
    InstanceConfigFactory.create(identifier='bisko', name='BISKO', config_source='database')
    framework = setup_bisko()
    import_bkg_organizations(snapshot(tmp_path), framework=framework)
    publish_bisko_template(framework)
    # A retired municipal dataset remains in the template DB for old revisions,
    # but is absent from the published graph and must not be provisioned locally.
    DatasetFactory.create(scope=framework.template_instance, identifier='kommune/unused')
    output = StringIO()

    call_command('provision_bisko_test_accounts', dry_run=True, stdout=output)
    assert 'Dry run' in output.getvalue()
    assert not User.objects.filter(email__endswith='.fake@kausal.tech').exists()

    monkeypatch.setenv('BISKO_TEST_ACCOUNT_PASSWORD', 'test-password-123')
    output = StringIO()
    call_command('provision_bisko_test_accounts', stdout=output)
    assert output.getvalue().count('created:') == 12
    assert 'test-password-123' not in output.getvalue()

    grants = OrganizationAccessGrant.objects.filter(framework=framework, user__email__endswith='.fake@kausal.tech')
    assert grants.count() == 12
    assert FrameworkConfig.objects.filter(framework=framework).count() == 3
    template = framework.template_instance
    assert template is not None
    template.refresh_from_db()
    for state in ('03', '07', '12'):
        config = FrameworkConfig.objects.get(instance_config__identifier=f'bisko-{state}001001')
        assert config.organization_identifier == f'{state}001001'
        assert config.instance_config.template_revision_id == template.live_revision_id
        assert config.instance_config.config_source == 'database'
        assert config.instance_config.primary_language == 'de'
        assert build_instance_snapshot(config.instance_config).spec.years.reference == 2020
        assert [node.identifier for node in build_instance_snapshot(config.instance_config).nodes] == ['bisko_shared']
        assert not config.instance_config.nodes.exists()
        spec = config.instance_config.spec
        assert spec is not None
        assert {p.local_id: p.value for p in spec.params if p.local_id in ('ags_number', 'lau_code', 'nuts_code')} == {
            'ags_number': f'{state}001001',
            'lau_code': f'DE_{state}001001',
            'nuts_code': f'DE{state}1',
        }
        assert spec.features.enable_user_management
        assert Submission.objects.filter(instance_config=config.instance_config, period_start=2023).count() == 1
        local = Dataset.objects.for_instance_config(config.instance_config).get(identifier='kommune/testeingabe')
        assert local.data_points.filter(date=date(2023, 1, 1), value__isnull=True).exists()
        assert not Dataset.objects.for_instance_config(config.instance_config).filter(identifier='kommune/unused').exists()
        effective = build_instance_snapshot(config.instance_config)
        local_binding = next(
            b for b in effective.bindings if isinstance(b.source, DatasetMetricSource) and b.source.dataset.startswith('kommune/')
        )
        reference_binding = next(
            b for b in effective.bindings if isinstance(b.source, DatasetMetricSource) and b.source.dataset == 'de/reference'
        )
        assert isinstance(local_binding.source, DatasetMetricSource)
        assert isinstance(reference_binding.source, DatasetMetricSource)
        assert local_binding.source.dataset_uuid == local.uuid
        assert reference_binding.source.dataset_uuid != local.uuid
    identifiers = Namespace.objects.get(identifier='ars')
    for state in ('03', '07', '12'):
        for level, ars, role in (
            ('state-admin', state, ObjectRole.ADMIN),
            ('state-editor', state, ObjectRole.EDITOR),
            ('district-editor', state + '001', ObjectRole.EDITOR),
            ('municipality-editor', state + '0010000001', ObjectRole.EDITOR),
        ):
            email = f'bisko-{state}-{level}.fake@kausal.tech'
            user = User.objects.get(email=email)
            assert user.check_password('test-password-123')
            assert (
                grants.get(user=user).organization
                == OrganizationIdentifier.objects.get(namespace=identifiers, identifier=ars).organization
            )
            assert grants.get(user=user).role == role
            assert f'{email} | username={user.username}' in output.getvalue()

    town = FrameworkConfig.objects.get(instance_config__identifier='bisko-03001001').instance_config
    town_dataset = Dataset.objects.for_instance_config(town).get(identifier='kommune/testeingabe')
    assert town_dataset.schema is not None
    town_metric = town_dataset.schema.metrics.get(name='Value')
    point = town_dataset.data_points.get(metric=town_metric, date=date(2023, 1, 1))
    point.value = 99
    point.save(update_fields=['value'])
    override_ids = set(town.binding_overrides.values_list('uuid', flat=True))
    monkeypatch.setenv('BISKO_TEST_ACCOUNT_PASSWORD', 'new-password-123')
    output = StringIO()
    call_command('provision_bisko_test_accounts', stdout=output)
    assert output.getvalue().count('existing:') == 12
    assert grants.count() == 12
    assert FrameworkConfig.objects.filter(framework=framework).count() == 3
    assert list(town_dataset.data_points.values_list('value', flat=True)) == [99]
    assert set(town.binding_overrides.values_list('uuid', flat=True)) == override_ids
    assert User.objects.get(email='bisko-03-state-admin.fake@kausal.tech').check_password('new-password-123')

    template = framework.template_instance
    assert template is not None
    earlier_revision_id = template.live_revision_id
    template.nodes.filter(identifier='bisko_shared').update(name='Updated BISKO method')
    template.publish_instance()
    instance = FrameworkConfig.objects.get(instance_config__identifier='bisko-03001001').instance_config
    instance.refresh_from_db()
    assert instance.template_revision_id == earlier_revision_id
    assert template.live_revision is not None
    upgrade_template_instance(instance, template.live_revision)
    instance.refresh_from_db()
    assert instance.template_revision_id != earlier_revision_id
    assert [str(node.name) for node in build_instance_snapshot(instance).nodes] == ['Updated BISKO method']


def test_provision_bisko_test_accounts_rejects_changed_grant_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    InstanceConfigFactory.create(identifier='bisko', name='BISKO', config_source='database')
    framework = setup_bisko()
    import_bkg_organizations(snapshot(tmp_path), framework=framework)
    publish_bisko_template(framework)
    monkeypatch.setenv('BISKO_TEST_ACCOUNT_PASSWORD', 'test-password-123')
    call_command('provision_bisko_test_accounts', stdout=StringIO())
    grant = OrganizationAccessGrant.objects.get(framework=framework, user__email='bisko-07-state-admin.fake@kausal.tech')
    grant.role = ObjectRole.VIEWER
    grant.save(update_fields=['role'])

    monkeypatch.setenv('BISKO_TEST_ACCOUNT_PASSWORD', 'new-password-123')
    with pytest.raises(CommandError, match='changed or suspended grant'):
        call_command('provision_bisko_test_accounts', stdout=StringIO())
    assert User.objects.get(email='bisko-03-state-admin.fake@kausal.tech').check_password('test-password-123')


def test_activate_bisko_municipality_requires_published_template(tmp_path: Path) -> None:
    InstanceConfigFactory.create(identifier='bisko', name='BISKO')
    framework = setup_bisko()
    import_bkg_organizations(snapshot(tmp_path), framework=framework)
    municipality = OrganizationIdentifier.objects.get(namespace__identifier='ars', identifier='030010000001').organization

    with pytest.raises(ActivationError, match='Publish the BISKO template'):
        activate_bisko_municipality(framework, municipality)
    assert not FrameworkConfig.objects.filter(framework=framework).exists()


def test_weather_defaults_seed_on_activation_and_setup_backfills_empty_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = InstanceConfigFactory.create(identifier='bisko', name='BISKO', config_source='database')
    framework = setup_bisko()
    import_bkg_organizations(snapshot(tmp_path), framework=framework)
    publish_bisko_template(framework)
    schema = DatasetSchemaFactory.create()
    dimension = DimensionFactory.create(name='Sectors')
    DatasetSchemaDimensionFactory.create(schema=schema, dimension=dimension)
    DimensionScope.objects.create(
        dimension=dimension,
        scope_content_type=ContentType.objects.get_for_model(template),
        scope_id=template.pk,
        identifier='sector',
    )
    for identifier in (*HEATING_SECTORS, *NEUTRAL_SECTORS):
        DimensionCategoryFactory.create(dimension=dimension, identifier=identifier)
    metric = DatasetMetricFactory.create(schema=schema, name='default', unit='')
    weather = DatasetFactory.create(scope=template, schema=schema, identifier=WEATHER_DATASET)
    node = NodeConfigFactory.create(instance=template, identifier='weather_input')
    assert node.spec is not None
    port = InputPortDef(
        id=uuid4(), identifier='weather', unit=unit_registry.parse_units('dimensionless'), binding_owner='instance'
    )
    node.spec.input_ports = [port]
    node.save(update_fields=['spec'])
    NodeInputPortBinding.objects.create(instance=template, node=node, port_id=port.id, dataset=weather, metric=metric)
    share_template_catalogue(framework)
    template.invalidate_cache()
    publish_template_instance(template)
    framework.refresh_from_db()

    frame = pl.DataFrame({
        'nuts': ['DE121'] * 46,
        'Year': list(range(1980, 2026)),
        'hdd_eurostat': [500 if year == 2023 else 1000 for year in range(1980, 2026)],
    })
    monkeypatch.setattr('frameworks.bisko.activation.load_weather_source', lambda _framework: (frame, 'test-commit'))
    monkeypatch.setattr('frameworks.bisko.provisioning.load_weather_source', lambda _framework: (frame, 'test-commit'))
    municipality = OrganizationIdentifier.objects.get(namespace__identifier='ags', identifier='12001001').organization
    config, created = activate_bisko_municipality(framework, municipality)
    assert created
    local = Dataset.objects.for_instance_config(config.instance_config).get(identifier=WEATHER_DATASET)
    assert local.data_points.count() == 80

    local.data_points.all().delete()
    setup_bisko()
    assert local.data_points.count() == 80
    point = local.data_points.get(date=date(2023, 1, 1), dimension_categories__identifier='private_households')
    assert point.value == 2
    setup_bisko()
    assert local.data_points.count() == 80


def test_activate_framework_organization_is_scoped_and_repeatable(tmp_path: Path, client: Client) -> None:  # noqa: PLR0915
    InstanceConfigFactory.create(identifier='bisko', name='BISKO', config_source='database')
    framework = setup_bisko()
    import_bkg_organizations(snapshot(tmp_path), framework=framework)
    publish_bisko_template(framework)
    municipality = OrganizationIdentifier.objects.get(namespace__identifier='ars', identifier='030010000001').organization
    state = OrganizationIdentifier.objects.get(namespace__identifier='ars', identifier='03').organization
    user = UserFactory.create()
    OrganizationAccessGrant.objects.create(framework=framework, organization=state, user=user, role=ObjectRole.EDITOR)
    client.force_login(user)
    gql = PathsTestClient(client)
    mutation = """
        mutation ($organization: ID!) {
          activateFrameworkOrganization(frameworkId: "bisko", organizationId: $organization) {
            ... on ActivateOrganizationResult {
              organizationId frameworkConfigId instanceIdentifier created
            }
          }
        }
    """
    variables = {'organization': str(municipality.uuid)}
    first = gql.query_data(mutation, variables=variables)['activateFrameworkOrganization']
    assert first['created'] is True
    assert first['instanceIdentifier'] == 'bisko-03001001'
    second = gql.query_data(mutation, variables=variables)['activateFrameworkOrganization']
    assert second['created'] is False
    assert second['frameworkConfigId'] == first['frameworkConfigId']
    assert FrameworkConfig.objects.filter(framework=framework).count() == 1

    config = FrameworkConfig.objects.get(framework=framework, instance_config__organization=municipality)
    gql.set_instance(config.instance_config)
    editor = gql.query_data("""
        { instance { editor {
          datasets { id identifier isEditable }
          datasetPorts { dataset { id identifier isEditable } }
        } } }
    """)['instance']['editor']
    local_dataset = next(item for item in editor['datasets'] if item['identifier'] == 'kommune/testeingabe')
    assert local_dataset['isEditable'] is True
    bound = {port['dataset']['identifier']: port['dataset'] for port in editor['datasetPorts']}
    assert bound['kommune/testeingabe']['id'] == local_dataset['id']
    assert bound['de/reference']['isEditable'] is False
    assert (
        gql.query_data(
            """
        query ($id: ID!) { instance { editor { dataset(id: $id) { id identifier } } } }
    """,
            variables={'id': local_dataset['id']},
        )['instance']['editor']['dataset']['identifier']
        == 'kommune/testeingabe'
    )
    local_input = Dataset.objects.for_instance_config(config.instance_config).get(identifier='kommune/testeingabe')
    assert local_input.schema is not None
    point = gql.query_data(
        """
        mutation ($instanceId: ID!, $datasetId: ID!, $metricId: UUID!) {
          instanceEditor(instanceId: $instanceId) {
            datasetEditor(datasetId: $datasetId) {
              createDataPoint(input: {
                date: "2022-01-01", value: 42, metricId: $metricId, dimensionCategoryIds: []
              }) {
                __typename
                ... on DataPoint { id value }
                ... on OperationInfo { messages { kind message field code } }
              }
            }
          }
        }
        """,
        variables={
            'instanceId': str(config.instance_config.pk),
            'datasetId': str(local_input.uuid),
            'metricId': str(local_input.schema.metrics.get(name='Value').uuid),
        },
    )['instanceEditor']['datasetEditor']['createDataPoint']
    assert point['__typename'] == 'DataPoint'
    assert list(local_input.data_points.filter(date=date(2022, 1, 1)).values_list('value', flat=True)) == [42]
    gql.set_instance(None)

    admin = UserFactory.create()
    OrganizationAccessGrant.objects.create(framework=framework, organization=state, user=admin, role=ObjectRole.ADMIN)
    client.force_login(admin)
    gql.set_instance(config.instance_config)
    account_data = gql.query_data("""
        { instance {
          memberSeatLimit memberSeatsInUse
          users { role }
          inheritedOrganizationGrants { userEmail role organizationId }
        } }
    """)['instance']
    assert account_data['memberSeatLimit'] == 5
    assert account_data['memberSeatsInUse'] == 0
    assert account_data['users'] == []
    assert {(grant['userEmail'], grant['role']) for grant in account_data['inheritedOrganizationGrants']} == {
        (user.email, 'EDITOR'),
        (admin.email, 'ADMIN'),
    }
    gql.set_instance(None)

    rows = gql.query_data("""
        { framework(identifier: "bisko") { organizations(search: "Municipality 03001001") {
          ars instanceIdentifier municipalityCount activatedMunicipalityCount unactivatedMunicipalityCount
        } } }
    """)['framework']['organizations']
    assert rows == [
        {
            'ars': '030010000001',
            'instanceIdentifier': 'bisko-03001001',
            'municipalityCount': 1,
            'activatedMunicipalityCount': 1,
            'unactivatedMunicipalityCount': 0,
        }
    ]
    state_counts = gql.query_data(
        """query ($id: ID!) { framework(identifier: "bisko") {
          organization(id: $id) { municipalityCount activatedMunicipalityCount unactivatedMunicipalityCount }
        } }""",
        variables={'id': str(state.uuid)},
    )['framework']['organization']
    assert state_counts == {
        'municipalityCount': 4,
        'activatedMunicipalityCount': 1,
        'unactivatedMunicipalityCount': 3,
    }
    configs = gql.query_data(
        """
        query ($organization: ID!) {
          framework(identifier: "bisko") {
            configs(organizationId: $organization) { instanceIdentifier organizationName organizationId }
          }
        }
    """,
        variables={'organization': str(state.uuid)},
    )['framework']['configs']
    assert configs == [
        {
            'instanceIdentifier': 'bisko-03001001',
            'organizationName': 'Municipality 03001001',
            'organizationId': str(municipality.uuid),
        }
    ]

    foreign = OrganizationIdentifier.objects.get(namespace__identifier='ars', identifier='070010000001').organization
    gql.query_errors(mutation, variables={'organization': str(foreign.uuid)}, assert_error_message='Permission denied')
    gql.query_errors(mutation, variables={'organization': str(state.uuid)}, assert_error_message='Only municipalities')


def test_framework_graphql_lists_only_granted_subtree(tmp_path: Path, client: Client) -> None:
    InstanceConfigFactory.create(identifier='bisko', name='BISKO')
    framework = setup_bisko()
    import_bkg_organizations(snapshot(tmp_path), framework=framework)
    district = OrganizationIdentifier.objects.get(namespace__identifier='ars', identifier='03001').organization
    municipality = OrganizationIdentifier.objects.get(namespace__identifier='ars', identifier='030010000001').organization
    instance = InstanceConfigFactory.create(
        identifier='municipality-count', name='Municipality count', organization=municipality, config_source='database'
    )
    FrameworkConfigFactory.create(framework=framework, instance_config=instance)
    result = replace_population_projection(
        framework,
        pl.DataFrame({
            'lau': ['DE_03001001', 'DE_03001002', 'DE_03002001'],
            'Year': [2023, 2023, 2023],
            'population': [100.0, 200.0, 300.0],
        }),
        source_revision='population-test-revision',
    )
    assert result.observations == 3
    state = OrganizationIdentifier.objects.get(namespace__identifier='ars', identifier='03').organization
    assert population_aggregates(framework, 2023).by_path[state.path].value == 600
    user = UserFactory.create(username='district-reader')
    OrganizationAccessGrant.objects.create(framework=framework, organization=district, user=user, role=ObjectRole.VIEWER)
    client.force_login(user)
    gql = PathsTestClient(client)
    query = """
        query ($parent: ID, $search: String) {
          framework(identifier: "bisko") {
            organizations(parentId: $parent, search: $search) {
              __typename id name classificationIdentifier ars ags instanceIdentifier
              municipalityCount activatedMunicipalityCount unactivatedMunicipalityCount
            }
          }
        }
    """
    root = gql.query_data(query)['framework']['organizations']
    assert [entry['ars'] for entry in root] == ['03001']
    assert root[0]['__typename'] == 'FrameworkOrganization'
    assert (root[0]['municipalityCount'], root[0]['activatedMunicipalityCount'], root[0]['unactivatedMunicipalityCount']) == (
        2,
        1,
        1,
    )
    children = gql.query_data(query, variables={'parent': str(district.uuid)})['framework']['organizations']
    assert {entry['ars'] for entry in children} == {'030010000001', '030010000002'}
    assert {entry['ars']: entry['unactivatedMunicipalityCount'] for entry in children} == {
        '030010000001': 0,
        '030010000002': 1,
    }
    assert next(entry for entry in children if entry['ars'] == '030010000001')['instanceIdentifier'] == 'municipality-count'
    searched = gql.query_data(query, variables={'parent': str(district.uuid), 'search': '03001002'})['framework']['organizations']
    assert [entry['ars'] for entry in searched] == ['030010000002']
    assert gql.query_data(query, variables={'search': 'State 07'})['framework']['organizations'] == []
    lookup = """
        query ($id: ID!) {
          framework(identifier: "bisko") {
            organization(id: $id) { id ars municipalityCount activatedMunicipalityCount unactivatedMunicipalityCount }
          }
        }
    """
    found = gql.query_data(lookup, variables={'id': str(district.uuid)})['framework']['organization']
    assert found == {
        'id': str(district.uuid),
        'ars': '03001',
        'municipalityCount': 2,
        'activatedMunicipalityCount': 1,
        'unactivatedMunicipalityCount': 1,
    }
    assert gql.query_data(lookup, variables={'id': str(state.uuid)})['framework']['organization'] is None
    population = gql.query_data(
        """query ($id: ID!) { framework(identifier: "bisko") {
          organization(id: $id) { population(year: 2023) {
            year value partialValue municipalityCount observedMunicipalityCount sourceRevision
          } }
        } }""",
        variables={'id': str(district.uuid)},
    )['framework']['organization']['population']
    assert population == {
        'year': 2023,
        'value': 300,
        'partialValue': 300,
        'municipalityCount': 2,
        'observedMunicipalityCount': 2,
        'sourceRevision': 'population-test-revision',
    }
    config_population = gql.query_data(
        """query ($district: ID!) { framework(identifier: "bisko") {
          configs(organizationId: $district) { population(year: 2023) { value sourceRevision } }
        } }""",
        variables={'district': str(district.uuid)},
    )['framework']['configs']
    assert config_population == [{'population': {'value': 100, 'sourceRevision': 'population-test-revision'}}]
    with patch('frameworks.schema.population_aggregates', wraps=population_aggregates) as load_year:
        siblings = gql.query_data(
            """query ($parent: ID!) { framework(identifier: "bisko") {
              organizations(parentId: $parent) { population(year: 2023) { value } }
            } }""",
            variables={'parent': str(district.uuid)},
        )['framework']['organizations']
    assert [row['population']['value'] for row in siblings] == [100, 200]
    assert load_year.call_count == 1
    state_user = UserFactory.create()
    OrganizationAccessGrant.objects.create(framework=framework, organization=state, user=state_user, role=ObjectRole.VIEWER)
    client.force_login(state_user)
    state_population = gql.query_data(
        """query ($id: ID!) { framework(identifier: "bisko") {
          organization(id: $id) { population(year: 2023) {
            value partialValue municipalityCount observedMunicipalityCount
          } }
        } }""",
        variables={'id': str(state.uuid)},
    )['framework']['organization']['population']
    assert state_population == {
        'value': None,
        'partialValue': 600,
        'municipalityCount': 4,
        'observedMunicipalityCount': 3,
    }
    with pytest.raises(ValueError, match='Duplicate population'):
        replace_population_projection(
            framework,
            pl.DataFrame({'lau': ['DE_03001001', 'DE_03001001'], 'Year': [2023, 2023], 'population': [1.0, 2.0]}),
            source_revision='bad-revision',
        )
    assert population_aggregates(framework, 2023).by_path[state.path].value == 600


def test_district_free_city_population_is_counted_once(client: Client) -> None:
    InstanceConfigFactory.create(identifier='bisko', name='BISKO')
    framework = setup_bisko()
    state = OrganizationFactory.create(name='State', classification=OrganizationClass.objects.get(identifier='de_state'))
    city = OrganizationFactory.create(
        parent=state, name='City district', classification=OrganizationClass.objects.get(identifier='de_district_free_city')
    )
    municipality = OrganizationFactory.create(
        parent=city, name='City municipality', classification=OrganizationClass.objects.get(identifier='de_municipality')
    )
    FrameworkOrganizationRoot.objects.create(framework=framework, organization=state)
    OrganizationIdentifier.objects.create(
        organization=municipality, namespace=Namespace.objects.get(identifier='ags'), identifier='07111000'
    )
    instance = InstanceConfigFactory.create(name='City', identifier='city-count', organization=municipality)
    FrameworkConfigFactory.create(framework=framework, instance_config=instance)
    replace_population_projection(
        framework,
        pl.DataFrame({'lau': ['DE_07111000'], 'Year': [2023], 'population': [115000.0]}),
        source_revision='city-source',
    )
    user = UserFactory.create()
    OrganizationAccessGrant.objects.create(framework=framework, organization=state, user=user, role=ObjectRole.VIEWER)
    client.force_login(user)
    gql = PathsTestClient(client)
    rows = gql.query_data(
        """query ($parent: ID!) { framework(identifier: "bisko") {
          organizations(parentId: $parent) {
            name municipalityCount activatedMunicipalityCount unactivatedMunicipalityCount
            population(year: 2023) { value observedMunicipalityCount }
          }
        } }""",
        variables={'parent': str(state.uuid)},
    )['framework']['organizations']
    assert rows == [
        {
            'name': 'City district',
            'municipalityCount': 1,
            'activatedMunicipalityCount': 1,
            'unactivatedMunicipalityCount': 0,
            'population': {'value': 115000, 'observedMunicipalityCount': 1},
        }
    ]


def test_grants_outside_framework_scope_cannot_cross_into_it() -> None:
    InstanceConfigFactory.create(identifier='bisko', name='BISKO')
    framework = setup_bisko()
    outer = OrganizationFactory.create(name='Outer')
    state = OrganizationFactory.create(parent=outer, name='Custom state')
    FrameworkOrganizationRoot.objects.create(framework=framework, organization=state)
    user = UserFactory.create()
    OrganizationAccessGrant.objects.create(framework=framework, organization=outer, user=user, role=ObjectRole.ADMIN)
    assert not accessible_organizations(framework, user).exists()
    city = InstanceConfigFactory.create(name='Scoped city', organization=state, config_source='database')
    FrameworkConfigFactory.create(framework=framework, instance_config=city)
    assert not type(city).objects.filter(city.permission_policy().role_q(user, 'view'), pk=city.pk).exists()


def test_viewer_grant_cannot_change_organization(tmp_path: Path) -> None:
    InstanceConfigFactory.create(identifier='bisko', name='BISKO')
    framework = setup_bisko()
    import_bkg_organizations(snapshot(tmp_path), framework=framework)
    state = OrganizationIdentifier.objects.get(namespace__identifier='ars', identifier='03').organization
    user = UserFactory.create()
    OrganizationAccessGrant.objects.create(framework=framework, organization=state, user=user, role=ObjectRole.VIEWER)
    assert user_can_access_organization(framework, user, state)
    assert not user_can_access_organization(framework, user, state, action='change')


def test_grant_command_requires_framework_coverage(tmp_path: Path) -> None:
    InstanceConfigFactory.create(identifier='bisko', name='BISKO')
    framework = setup_bisko()
    import_bkg_organizations(snapshot(tmp_path), framework=framework)
    user = UserFactory.create(username='delegated-district')
    call_command('grant_organization_access', user.username, '03001', role='editor')
    assert OrganizationAccessGrant.objects.get(user=user).role == ObjectRole.EDITOR
    outside = OrganizationFactory.create()
    OrganizationIdentifier.objects.create(
        organization=outside, namespace=Namespace.objects.get(identifier='ars'), identifier='99'
    )
    with pytest.raises(CommandError, match='outside framework'):
        call_command('grant_organization_access', user.username, '99')
    assert OrganizationAccessGrant.objects.filter(framework=framework, user=user).count() == 1


def test_organization_role_mutations_require_admin_and_preserve_grant(tmp_path: Path, client: Client) -> None:
    InstanceConfigFactory.create(identifier='bisko', name='BISKO')
    framework = setup_bisko()
    import_bkg_organizations(snapshot(tmp_path), framework=framework)
    district = OrganizationIdentifier.objects.get(namespace__identifier='ars', identifier='03001').organization
    manager = UserFactory.create(username='framework-role-manager')
    member = UserFactory.create(username='organization-member')
    client.force_login(manager)
    gql = PathsTestClient(client)
    mutation = """
    mutation ($fw: ID!, $org: ID!, $user: ID!) {
      assignOrganizationRole(frameworkId: $fw, organizationId: $org, userId: $user, role: EDITOR) {
        ... on OrganizationAccessGrant { organizationId userId role suspendedAt }
      }
    }
    """
    variables = {'fw': framework.identifier, 'org': str(district.uuid), 'user': str(member.uuid)}
    assert gql.query_errors(mutation, variables=variables)
    framework_admin_role.assign_user(framework, manager)
    data = gql.query_data(mutation, variables=variables)['assignOrganizationRole']
    assert data['role'] == 'EDITOR'
    assert data['organizationId'] == str(district.uuid)
    assert user_can_access_organization(framework, member, district, action='change')
    directory = gql.query_data('{ framework(identifier: "bisko") { organizationGrants { organizationId userId role } } }')[
        'framework'
    ]['organizationGrants']
    assert directory == [{'organizationId': str(district.uuid), 'userId': str(member.uuid), 'role': 'EDITOR'}]
    suspend = """
    mutation ($fw: ID!, $org: ID!, $user: ID!) {
      suspendOrganizationRole(frameworkId: $fw, organizationId: $org, userId: $user) {
        ... on OrganizationAccessGrant { suspendedAt retentionUntil }
      }
    }
    """
    suspended = gql.query_data(suspend, variables=variables)['suspendOrganizationRole']
    assert suspended['suspendedAt'] is not None
    assert suspended['retentionUntil'] is not None
    assert not user_can_access_organization(framework, member, district)
    reactivate = """
    mutation ($fw: ID!, $org: ID!, $user: ID!) {
      reactivateOrganizationRole(frameworkId: $fw, organizationId: $org, userId: $user) {
        ... on OrganizationAccessGrant { suspendedAt role }
      }
    }
    """
    restored = gql.query_data(reactivate, variables=variables)['reactivateOrganizationRole']
    assert restored['suspendedAt'] is None
    assert restored['role'] == 'EDITOR'
    assert user_can_access_organization(framework, member, district)
    assert OrganizationAccessGrantEvent.objects.get(action='reactivated').retention_until is not None


def test_district_grant_reaches_member_instance_but_suspension_blocks_it(tmp_path: Path) -> None:
    InstanceConfigFactory.create(identifier='bisko', name='BISKO')
    framework = setup_bisko()
    import_bkg_organizations(snapshot(tmp_path), framework=framework)
    ars = Namespace.objects.get(identifier='ars')
    district = OrganizationIdentifier.objects.get(namespace=ars, identifier='03001').organization
    municipality = OrganizationIdentifier.objects.get(namespace=ars, identifier='030010000001').organization
    neighbor = OrganizationIdentifier.objects.get(namespace=ars, identifier='030020000001').organization
    city = InstanceConfigFactory.create(name='Municipality instance', organization=municipality, config_source='database')
    other = InstanceConfigFactory.create(name='Neighbor instance', organization=neighbor, config_source='database')
    FrameworkConfigFactory.create(framework=framework, instance_config=city)
    FrameworkConfigFactory.create(framework=framework, instance_config=other)
    user = UserFactory.create()
    user.cached_adminable_instances = str(other.pk)
    user.save(update_fields=['cached_adminable_instances'])
    OrganizationAccessGrant.objects.create(framework=framework, organization=district, user=user, role=ObjectRole.EDITOR)
    user.refresh_from_db()
    assert user.cached_adminable_instances is None
    assert user.get_adminable_instances().filter(pk=city.pk).exists()
    policy = city.permission_policy()
    assert policy.user_has_perm(user, 'change', city)
    assert not policy.user_has_perm(user, 'change', other)
    assert list(type(city).objects.filter(policy.role_q(user, 'change')).values_list('pk', flat=True)) == [city.pk]
    schema = DatasetSchemaFactory.create(name='District input')
    own_dataset = DatasetFactory.create(schema=schema, scope=city)
    other_dataset = DatasetFactory.create(schema=schema, scope=other)
    assert own_dataset.permission_policy().user_has_perm(user, 'change', own_dataset)
    assert not other_dataset.permission_policy().user_has_perm(user, 'change', other_dataset)
    InstanceMemberAssignment.objects.create(
        instance_config=city,
        user=user,
        role='viewer',
        suspended_at=timezone.now(),
        retention_until=retention_date(timezone.localdate()),
    )
    assert not policy.user_has_perm(user, 'view', city)
    assert not type(city).objects.filter(policy.role_q(user, 'view'), pk=city.pk).exists()
    assert not own_dataset.permission_policy().user_has_perm(user, 'view', own_dataset)
