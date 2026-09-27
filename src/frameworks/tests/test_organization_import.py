from datetime import date
from io import StringIO
from typing import TYPE_CHECKING

from django.core.management import call_command
from django.core.management.base import CommandError
from django.utils import timezone

import polars as pl
import pytest

from kausal_common.datasets.tests.factories import DatasetFactory, DatasetSchemaFactory
from kausal_common.people.models import ObjectRole

from paths.tests.graphql import PathsTestClient

from frameworks.models import (
    FrameworkConfig,
    FrameworkOrganizationRoot,
    OrganizationAccessGrant,
    OrganizationAccessGrantEvent,
)
from frameworks.organization_access import accessible_organizations, user_can_access_organization
from frameworks.provisioning import GERMAN_ORGANIZATION_CLASSES, setup_bisko
from frameworks.roles import framework_admin_role
from frameworks.tests.factories import FrameworkConfigFactory
from nodes.membership import retention_date
from nodes.models import InstanceMemberAssignment
from nodes.tests.factories import InstanceConfigFactory
from orgs.import_bkg import import_bkg_organizations
from orgs.models import Namespace, Organization, OrganizationClass, OrganizationIdentifier
from orgs.tests.factories import OrganizationFactory
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


def test_catalogue_and_import_reuse_existing_municipality(tmp_path: Path) -> None:
    InstanceConfigFactory.create(identifier='bisko', name='BISKO')
    framework = setup_bisko()
    assert set(OrganizationClass.objects.values_list('identifier', flat=True)) >= {
        identifier for identifier, _ in GERMAN_ORGANIZATION_CLASSES
    }
    assert {'ars', 'ags'} <= set(Namespace.objects.values_list('identifier', flat=True))
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


def test_provision_bisko_test_accounts_is_repeatable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    InstanceConfigFactory.create(identifier='bisko', name='BISKO')
    framework = setup_bisko()
    import_bkg_organizations(snapshot(tmp_path), framework=framework)
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

    monkeypatch.setenv('BISKO_TEST_ACCOUNT_PASSWORD', 'new-password-123')
    output = StringIO()
    call_command('provision_bisko_test_accounts', stdout=output)
    assert output.getvalue().count('existing:') == 12
    assert grants.count() == 12
    assert User.objects.get(email='bisko-03-state-admin.fake@kausal.tech').check_password('new-password-123')


def test_provision_bisko_test_accounts_rejects_changed_grant_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    InstanceConfigFactory.create(identifier='bisko', name='BISKO')
    framework = setup_bisko()
    import_bkg_organizations(snapshot(tmp_path), framework=framework)
    monkeypatch.setenv('BISKO_TEST_ACCOUNT_PASSWORD', 'test-password-123')
    call_command('provision_bisko_test_accounts', stdout=StringIO())
    grant = OrganizationAccessGrant.objects.get(framework=framework, user__email='bisko-07-state-admin.fake@kausal.tech')
    grant.role = ObjectRole.VIEWER
    grant.save(update_fields=['role'])

    monkeypatch.setenv('BISKO_TEST_ACCOUNT_PASSWORD', 'new-password-123')
    with pytest.raises(CommandError, match='changed or suspended grant'):
        call_command('provision_bisko_test_accounts', stdout=StringIO())
    assert User.objects.get(email='bisko-03-state-admin.fake@kausal.tech').check_password('test-password-123')


def test_framework_graphql_lists_only_granted_subtree(tmp_path: Path, client: Client) -> None:
    InstanceConfigFactory.create(identifier='bisko', name='BISKO')
    framework = setup_bisko()
    import_bkg_organizations(snapshot(tmp_path), framework=framework)
    district = OrganizationIdentifier.objects.get(namespace__identifier='ars', identifier='03001').organization
    user = UserFactory.create(username='district-reader')
    OrganizationAccessGrant.objects.create(framework=framework, organization=district, user=user, role=ObjectRole.VIEWER)
    client.force_login(user)
    gql = PathsTestClient(client)
    query = """
        query ($parent: ID, $search: String) {
          framework(identifier: "bisko") {
            organizations(parentId: $parent, search: $search) {
              __typename id name classificationIdentifier ars ags
            }
          }
        }
    """
    root = gql.query_data(query)['framework']['organizations']
    assert [entry['ars'] for entry in root] == ['03001']
    assert root[0]['__typename'] == 'FrameworkOrganization'
    children = gql.query_data(query, variables={'parent': str(district.uuid)})['framework']['organizations']
    assert {entry['ars'] for entry in children} == {'030010000001', '030010000002'}
    assert gql.query_data(query, variables={'search': 'State 07'})['framework']['organizations'] == []


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
