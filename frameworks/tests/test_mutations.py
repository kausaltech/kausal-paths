"""Tests for the CADS self-service GraphQL mutations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from frameworks.models import FrameworkConfig, MeasureDataPoint, MeasureTemplate, Section
from frameworks.tests.factories import FrameworkConfigFactory, FrameworkFactory
from nodes.models import InstanceConfig
from users.models import User

if TYPE_CHECKING:
    from django.test import Client

    from paths.tests.graphql import PathsTestClient

    from frameworks.models import Framework
    from frameworks.nzc import NZCPlaceholderInput


gql = str

pytestmark = pytest.mark.django_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def framework() -> Framework:
    return FrameworkFactory.create(
        identifier='testfw',
        name='Test Framework',
        public_base_fqdn='testfw.example.com',
        allow_user_registration=True,
        allow_instance_creation=True,
    )


@pytest.fixture
def closed_framework() -> Framework:
    return FrameworkFactory.create(
        identifier='closedfw',
        name='Closed Framework',
        public_base_fqdn='closedfw.example.com',
        allow_user_registration=False,
        allow_instance_creation=False,
    )


@pytest.fixture
def gql_client(client: Client) -> PathsTestClient:
    from paths.tests.graphql import PathsTestClient

    return PathsTestClient(client)


@pytest.fixture
def authenticated_gql_client(client: Client, framework: Framework) -> PathsTestClient:
    from paths.tests.graphql import PathsTestClient

    user = User.objects.create_user(username='existing', email='admin@test.com', password='testpass123!', is_staff=False)
    client.force_login(user)
    return PathsTestClient(client)


def _framework_admin_gql_client(client: Client, framework: Framework, username: str = 'framework-admin') -> PathsTestClient:
    from paths.tests.graphql import PathsTestClient

    from frameworks.roles import framework_admin_role

    user = User.objects.create_user(username=username, email=f'{username}@test.com', password='testpass123!')
    framework_admin_role.assign_user(framework, user)
    client.force_login(user)
    return PathsTestClient(client)


def _create_net_zero_cities_organization() -> None:
    from orgs.models import Organization
    from orgs.tests.factories import OrganizationFactory

    if not Organization.objects.filter(name='NetZeroCities').exists():
        OrganizationFactory.create(name='NetZeroCities')


def _create_measure_template(framework: Framework) -> MeasureTemplate:
    root = framework.create_root_section()
    section = root.add_child(instance=Section(framework=framework, name='Measures'))
    return MeasureTemplate.objects.create(section=section, name='Test measure', unit='kt/a')


# ---------------------------------------------------------------------------
# Legacy Graphene framework query
# ---------------------------------------------------------------------------


FRAMEWORKS_QUERY = gql("""
query Frameworks($identifier: ID!) {
    frameworks {
        identifier
        name
        allowUserRegistration
        allowInstanceCreation
    }
    framework(identifier: $identifier) {
        identifier
        name
        allowUserRegistration
        allowInstanceCreation
    }
}
""")


def test_framework_query_exposes_frameworks(
    gql_client: PathsTestClient,
    framework: Framework,
    closed_framework: Framework,
) -> None:
    data = gql_client.query_data(FRAMEWORKS_QUERY, variables={'identifier': framework.identifier})

    frameworks_by_id = {fw['identifier']: fw for fw in data['frameworks']}
    assert frameworks_by_id[framework.identifier]['name'] == 'Test Framework'
    assert frameworks_by_id[framework.identifier]['allowUserRegistration'] is True
    assert frameworks_by_id[framework.identifier]['allowInstanceCreation'] is True
    assert frameworks_by_id[closed_framework.identifier]['allowUserRegistration'] is False

    selected = data['framework']
    assert selected['identifier'] == framework.identifier
    assert selected['name'] == framework.name


# ---------------------------------------------------------------------------
# Legacy Graphene framework mutations
# ---------------------------------------------------------------------------


CREATE_FRAMEWORK_CONFIG = gql("""
mutation CreateFrameworkConfig(
    $frameworkId: ID!
    $instanceIdentifier: ID!
    $name: String!
    $baselineYear: Int!
) {
    createFrameworkConfig(
        frameworkId: $frameworkId
        instanceIdentifier: $instanceIdentifier
        name: $name
        baselineYear: $baselineYear
    ) {
        ok
        frameworkConfig {
            organizationName
            baselineYear
            viewUrl
        }
    }
}
""")


def test_create_framework_config_mutation_creates_instance(client: Client) -> None:
    _create_net_zero_cities_organization()
    framework = FrameworkFactory.create(
        identifier='legacyfw',
        name='Legacy Framework',
        public_base_fqdn=None,
    )
    gql_client = _framework_admin_gql_client(client, framework)

    data = gql_client.query_data(
        CREATE_FRAMEWORK_CONFIG,
        variables={
            'frameworkId': framework.identifier,
            'instanceIdentifier': 'legacy-city',
            'name': 'Legacy City',
            'baselineYear': 2020,
        },
    )

    result = data['createFrameworkConfig']
    assert result['ok'] is True
    assert result['frameworkConfig']['organizationName'] == 'Legacy City'
    assert result['frameworkConfig']['baselineYear'] == 2020
    assert result['frameworkConfig']['viewUrl'] is None

    ic = InstanceConfig.objects.get(identifier='legacy-city')
    assert ic.has_framework_config()
    assert ic.framework_config.framework == framework


UPDATE_FRAMEWORK_CONFIG = gql("""
mutation UpdateFrameworkConfig($id: ID!, $organizationName: String!, $baselineYear: Int!, $targetYear: Int!) {
    updateFrameworkConfig(id: $id, organizationName: $organizationName, baselineYear: $baselineYear, targetYear: $targetYear) {
        ok
        frameworkConfig {
            organizationName
            baselineYear
            targetYear
        }
    }
}
""")


def test_update_framework_config_mutation_updates_fields(client: Client, framework: Framework) -> None:
    fwc = FrameworkConfigFactory.create(framework=framework, organization_name='Old City', baseline_year=2020)
    gql_client = _framework_admin_gql_client(client, framework)

    data = gql_client.query_data(
        UPDATE_FRAMEWORK_CONFIG,
        variables={
            'id': str(fwc.pk),
            'organizationName': 'Updated City',
            'baselineYear': 2021,
            'targetYear': 2040,
        },
    )

    result = data['updateFrameworkConfig']
    assert result['ok'] is True
    assert result['frameworkConfig']['organizationName'] == 'Updated City'
    assert result['frameworkConfig']['baselineYear'] == 2021
    assert result['frameworkConfig']['targetYear'] == 2040

    fwc.refresh_from_db()
    assert fwc.organization_name == 'Updated City'
    assert fwc.baseline_year == 2021
    assert fwc.target_year == 2040


DELETE_FRAMEWORK_CONFIG = gql("""
mutation DeleteFrameworkConfig($id: ID!) {
    deleteFrameworkConfig(id: $id) {
        ok
    }
}
""")


def test_delete_framework_config_mutation_deletes_instance(client: Client, framework: Framework) -> None:
    fwc = FrameworkConfigFactory.create(framework=framework)
    instance_config_id = fwc.instance_config_id
    gql_client = _framework_admin_gql_client(client, framework)

    data = gql_client.query_data(DELETE_FRAMEWORK_CONFIG, variables={'id': str(fwc.pk)})

    assert data['deleteFrameworkConfig']['ok'] is True
    assert not FrameworkConfig.objects.filter(pk=fwc.pk).exists()
    assert not InstanceConfig.objects.filter(pk=instance_config_id).exists()


UPDATE_MEASURE_DATA_POINT = gql("""
mutation UpdateMeasureDataPoint(
    $frameworkInstanceId: ID!
    $measureTemplateId: ID!
    $value: Float
    $year: Int
    $internalNotes: String
) {
    updateMeasureDataPoint(
        frameworkInstanceId: $frameworkInstanceId
        measureTemplateId: $measureTemplateId
        value: $value
        year: $year
        internalNotes: $internalNotes
    ) {
        ok
        measureDataPoint {
            year
            value
        }
    }
}
""")


def test_update_measure_data_point_mutation_creates_measure_point(client: Client, framework: Framework) -> None:
    fwc = FrameworkConfigFactory.create(framework=framework, baseline_year=2020)
    measure_template = _create_measure_template(framework)
    gql_client = _framework_admin_gql_client(client, framework)

    data = gql_client.query_data(
        UPDATE_MEASURE_DATA_POINT,
        variables={
            'frameworkInstanceId': str(fwc.pk),
            'measureTemplateId': str(measure_template.pk),
            'value': 42.5,
            'year': 2021,
            'internalNotes': 'Checked by model team',
        },
    )

    result = data['updateMeasureDataPoint']
    assert result['ok'] is True
    assert result['measureDataPoint']['year'] == 2021
    assert result['measureDataPoint']['value'] == 42.5

    data_point = MeasureDataPoint.objects.get(measure__framework_config=fwc, year=2021)
    assert data_point.value == 42.5
    assert data_point.measure.internal_notes == 'Checked by model team'


UPDATE_MEASURE_DATA_POINTS = gql("""
mutation UpdateMeasureDataPoints($frameworkConfigId: ID!, $measures: [MeasureInput!]!) {
    updateMeasureDataPoints(frameworkConfigId: $frameworkConfigId, measures: $measures) {
        ok
        createdDataPoints {
            year
            value
        }
        updatedDataPoints {
            year
            value
        }
        deletedDataPointCount
    }
}
""")


def test_update_measure_data_points_mutation_creates_bulk_points(client: Client, framework: Framework) -> None:
    fwc = FrameworkConfigFactory.create(framework=framework, baseline_year=2020)
    measure_template = _create_measure_template(framework)
    gql_client = _framework_admin_gql_client(client, framework)

    data = gql_client.query_data(
        UPDATE_MEASURE_DATA_POINTS,
        variables={
            'frameworkConfigId': str(fwc.pk),
            'measures': [
                {
                    'measureTemplateId': str(measure_template.pk),
                    'internalNotes': 'Bulk update',
                    'dataPoints': [
                        {'year': 2020, 'value': 10.0},
                        {'year': 2021, 'value': 11.0},
                    ],
                },
            ],
        },
    )

    result = data['updateMeasureDataPoints']
    assert result['ok'] is True
    assert result['deletedDataPointCount'] == 0
    assert result['updatedDataPoints'] == []
    assert sorted((dp['year'], dp['value']) for dp in result['createdDataPoints']) == [(2020, 10.0), (2021, 11.0)]

    values_by_year = {dp.year: dp.value for dp in MeasureDataPoint.objects.filter(measure__framework_config=fwc).order_by('year')}
    assert values_by_year == {2020: 10.0, 2021: 11.0}


CREATE_NZC_FRAMEWORK_CONFIG = gql("""
mutation CreateNzcFrameworkConfig($configInput: FrameworkConfigInput!, $nzcData: NZCCityEssentialData!) {
    createNzcFrameworkConfig(configInput: $configInput, nzcData: $nzcData) {
        ok
        frameworkConfig {
            organizationName
            baselineYear
            targetYear
            viewUrl
        }
    }
}
""")


def test_create_nzc_framework_config_mutation_creates_instance_and_defaults(
    client: Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _create_net_zero_cities_organization()
    framework = FrameworkFactory.create(
        identifier='nzc',
        name='NetZeroCities',
        public_base_fqdn=None,
    )
    gql_client = _framework_admin_gql_client(client, framework)
    dataset_repo = object()
    captured_defaults: list[tuple[str, dict[str, float] | None]] = []
    captured_placeholder_input: dict[str, object] = {}

    class FakeContext:
        def __init__(self, dataset_repo: object) -> None:
            self.dataset_repo = dataset_repo

    class FakeInstance:
        def __init__(self, context: FakeContext) -> None:
            self.context = context

    fake_context = FakeContext(dataset_repo)
    fake_instance = FakeInstance(fake_context)

    def fake_get_instance(self: InstanceConfig, *args: object, **kwargs: object) -> FakeInstance:
        return fake_instance

    def fake_get_nzc_default_values(repo: object, placeholder_input: NZCPlaceholderInput) -> dict[str, float]:
        assert repo is dataset_repo
        captured_placeholder_input.update(
            population=placeholder_input.population,
            renewmix=placeholder_input.renewmix,
            temperature=placeholder_input.temperature,
        )
        return {'template-uuid': 12.5}

    def fake_create_measure_defaults(self: FrameworkConfig, defaults: dict[str, float] | None = None) -> None:
        captured_defaults.append((self.instance_config.identifier, defaults))

    monkeypatch.setattr(InstanceConfig, 'get_instance', fake_get_instance)
    monkeypatch.setattr('frameworks.nzc.get_nzc_default_values', fake_get_nzc_default_values)
    monkeypatch.setattr(FrameworkConfig, 'create_measure_defaults', fake_create_measure_defaults)

    data = gql_client.query_data(
        CREATE_NZC_FRAMEWORK_CONFIG,
        variables={
            'configInput': {
                'frameworkId': framework.identifier,
                'instanceIdentifier': 'nzc-created-city',
                'name': 'NZC Created City',
                'baselineYear': 2020,
                'targetYear': 2035,
            },
            'nzcData': {
                'population': 123456,
                'temperature': 'LOW',
                'renewableMix': 'HIGH',
            },
        },
    )

    result = data['createNzcFrameworkConfig']
    assert result['ok'] is True
    assert result['frameworkConfig']['organizationName'] == 'NZC Created City'
    assert result['frameworkConfig']['baselineYear'] == 2020
    assert result['frameworkConfig']['targetYear'] == 2035
    assert result['frameworkConfig']['viewUrl'] is None

    ic = InstanceConfig.objects.get(identifier='nzc-created-city')
    assert ic.has_framework_config()
    assert ic.framework_config.framework == framework
    assert captured_placeholder_input == {
        'population': 123456,
        'renewmix': 'high',
        'temperature': 'low',
    }
    assert captured_defaults == [('nzc-created-city', {'template-uuid': 12.5})]


# ---------------------------------------------------------------------------
# registerUser
# ---------------------------------------------------------------------------

REGISTER_USER = gql("""
mutation RegisterUser($input: RegisterUserInput!) {
    registerUser(input: $input) {
        ... on RegisterUserResult {
            userId
            email
        }
        ... on OperationInfo {
            messages { message }
        }
    }
}
""")


def test_register_user_success(gql_client: PathsTestClient, framework: Framework) -> None:
    data = gql_client.query_data(
        REGISTER_USER,
        variables={
            'input': {
                'frameworkId': framework.identifier,
                'email': 'newuser@example.com',
                'password': 'SecurePass123!',
                'firstName': 'Test',
                'lastName': 'User',
            },
        },
    )
    result = data['registerUser']
    assert result['email'] == 'newuser@example.com'
    assert result['userId'] is not None

    user = User.objects.get(email='newuser@example.com')
    assert user.first_name == 'Test'
    assert user.last_name == 'User'
    assert user.is_staff is False
    assert user.check_password('SecurePass123!')


def test_register_user_duplicate_email(gql_client: PathsTestClient, framework: Framework) -> None:
    User.objects.create_user(username='existing', email='existing@example.com', password='pass123!')

    gql_client.query_errors(
        REGISTER_USER,
        variables={
            'input': {
                'frameworkId': framework.identifier,
                'email': 'existing@example.com',
                'password': 'SecurePass123!',
            },
        },
        assert_error_message='already exists',
    )


def test_register_user_weak_password(gql_client: PathsTestClient, framework: Framework) -> None:
    gql_client.query_errors(
        REGISTER_USER,
        variables={
            'input': {
                'frameworkId': framework.identifier,
                'email': 'newuser@example.com',
                'password': '123',
            },
        },
        assert_error_message='Invalid password',
    )


def test_register_user_framework_disallows(gql_client: PathsTestClient, closed_framework: Framework) -> None:
    gql_client.query_errors(
        REGISTER_USER,
        variables={
            'input': {
                'frameworkId': closed_framework.identifier,
                'email': 'newuser@example.com',
                'password': 'SecurePass123!',
            },
        },
        assert_error_message='not allowed',
    )


def test_register_user_unknown_framework(gql_client: PathsTestClient) -> None:
    gql_client.query_errors(
        REGISTER_USER,
        variables={
            'input': {
                'frameworkId': 'nonexistent',
                'email': 'newuser@example.com',
                'password': 'SecurePass123!',
            },
        },
        assert_error_message='not found',
    )


# ---------------------------------------------------------------------------
# createInstance
# ---------------------------------------------------------------------------

CREATE_INSTANCE = gql("""
mutation CreateInstance($input: CreateInstanceInput!) {
    createInstance(input: $input) {
        ... on CreateInstanceResult {
            instanceId
            instanceName
        }
        ... on OperationInfo {
            messages { message }
        }
    }
}
""")


def test_create_instance_success(authenticated_gql_client: PathsTestClient, framework: Framework) -> None:
    data = authenticated_gql_client.query_data(
        CREATE_INSTANCE,
        variables={
            'input': {
                'frameworkId': framework.identifier,
                'name': 'My City Model',
                'identifier': 'my-city',
                'organizationName': 'My City',
            },
        },
    )
    result = data['createInstance']
    assert result['instanceId'] == 'my-city'
    assert result['instanceName'] == 'My City Model'

    ic = InstanceConfig.objects.get(identifier='my-city')
    assert ic.config_source == 'database'
    assert ic.has_framework_config()
    assert ic.framework_config.framework == framework
    assert ic.admin_group is not None


def test_create_instance_unauthenticated(gql_client: PathsTestClient, framework: Framework) -> None:
    gql_client.query_errors(
        CREATE_INSTANCE,
        variables={
            'input': {
                'frameworkId': framework.identifier,
                'name': 'Anon City',
                'identifier': 'anon-city',
                'organizationName': 'Anon',
            },
        },
        assert_error_message='Authentication required',
    )


def test_create_instance_framework_disallows(authenticated_gql_client: PathsTestClient, closed_framework: Framework) -> None:
    authenticated_gql_client.query_errors(
        CREATE_INSTANCE,
        variables={
            'input': {
                'frameworkId': closed_framework.identifier,
                'name': 'Blocked City',
                'identifier': 'blocked-city',
                'organizationName': 'Blocked',
            },
        },
        assert_error_message='not allowed',
    )


# ---------------------------------------------------------------------------
# FrameworkLandingBlock query
# ---------------------------------------------------------------------------


PAGES_QUERY = gql("""
query Pages {
    pages {
        ... on InstanceRootPage {
            body {
                ... on FrameworkLandingBlock {
                    heading
                    body
                    ctaLabel
                    ctaUrl
                    framework {
                        identifier
                        allowUserRegistration
                        allowInstanceCreation
                    }
                }
            }
        }
    }
}
""")


def test_landing_block_exposes_framework(client: Client, framework: Framework) -> None:
    import json

    from wagtail.models import Locale, Page, Site

    from paths.tests.graphql import PathsTestClient

    from nodes.defs.instance_defs import InstanceSpec, YearsSpec
    from nodes.models import InstanceConfig
    from orgs.tests.factories import OrganizationFactory
    from pages.models import InstanceRootPage

    org = OrganizationFactory.create()
    spec = InstanceSpec(
        primary_language='en',
        owner='Test',
        years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030),
    )
    ic = InstanceConfig.objects.create(
        name='Landing Test',
        identifier='landing-test',
        primary_language='en',
        other_languages=[],
        organization=org,
        config_source='database',
        spec=spec,
    )
    locale, _ = Locale.objects.get_or_create(language_code='en')
    root = Page.get_first_root_node()
    assert root is not None
    body = json.dumps([
        {
            'type': 'framework_landing',
            'value': {
                'heading': 'Welcome',
                'body': '<p>Hello</p>',
                'cta_label': 'Go',
                'cta_url': '/register',
                'framework_identifier': framework.identifier,
            },
        }
    ])
    page = root.add_child(
        instance=InstanceRootPage(
            locale=locale,
            title='Landing Test',
            slug='landing-test',
            url_path='',
            body=body,
        )
    )
    site = Site.objects.create(site_name='Landing Test', hostname='landing-test.localhost', root_page=page)
    ic.site = site
    ic.save(update_fields=['site'])

    gql_client = PathsTestClient(client)
    gql_client.set_instance(ic)
    data = gql_client.query_data(PAGES_QUERY)

    pages = data['pages']
    assert len(pages) >= 1
    root_page_data = pages[0]
    assert 'body' in root_page_data
    blocks = root_page_data['body']
    assert len(blocks) == 1
    block = blocks[0]
    assert block['heading'] == 'Welcome'
    assert block['ctaLabel'] == 'Go'
    fw_data = block['framework']
    assert fw_data is not None
    assert fw_data['identifier'] == framework.identifier
    assert fw_data['allowUserRegistration'] is True
    assert fw_data['allowInstanceCreation'] is True


def test_create_instance_duplicate_identifier(authenticated_gql_client: PathsTestClient, framework: Framework) -> None:
    # Create first
    authenticated_gql_client.query_data(
        CREATE_INSTANCE,
        variables={
            'input': {
                'frameworkId': framework.identifier,
                'name': 'First City',
                'identifier': 'dupe-city',
                'organizationName': 'First',
            },
        },
    )
    # Try duplicate
    authenticated_gql_client.query_errors(
        CREATE_INSTANCE,
        variables={
            'input': {
                'frameworkId': framework.identifier,
                'name': 'Second City',
                'identifier': 'dupe-city',
                'organizationName': 'Second',
            },
        },
        assert_error_message='already exists',
    )
