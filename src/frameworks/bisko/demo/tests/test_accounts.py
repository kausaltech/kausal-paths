"""CLI provisioning works without request language context and keeps district grants scoped."""

from io import StringIO
from typing import TYPE_CHECKING

from django.core.management import call_command, get_commands
from django.core.management.base import CommandError

import pytest

from kausal_common.i18n.pydantic import get_i18n_context, i18n_context
from kausal_common.people.models import ObjectRole

from frameworks.bisko.demo.inventories import DEMO_MUNICIPALITIES
from frameworks.bisko.provisioning import setup_bisko
from frameworks.models import FrameworkConfig, OrganizationAccessGrant
from frameworks.organization_access import user_can_access_organization
from frameworks.tests.test_organization_import import (
    activation_sources,  # noqa: F401 -- pytest discovers the imported fixture.
    publish_bisko_template,
    snapshot,
)
from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.tests.factories import InstanceConfigFactory
from orgs.import_bkg import import_bkg_organizations
from orgs.models import OrganizationClass, OrganizationIdentifier
from users.models import User

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from frameworks.models import Framework

pytestmark = pytest.mark.django_db


@pytest.fixture
def account_framework(tmp_path: Path) -> Iterator[Framework]:
    InstanceConfigFactory.create(
        identifier='bisko-base',
        name='BISKO',
        config_source='database',
        owner='Test Owner',
        spec=InstanceModelSpec(years=YearsSpec(reference=2020, min_historical=2010, max_historical=2023, target=2035)),
    )
    framework = setup_bisko()
    import_bkg_organizations(snapshot(tmp_path), framework=framework)
    publish_bisko_template(framework)
    for ars in ('03451', '12072'):
        state = OrganizationIdentifier.objects.get(namespace__identifier='ars', identifier=ars[:2]).organization
        district = state.add_child(name=f'District {ars}', classification=OrganizationClass.objects.get(identifier='de_district'))
        OrganizationIdentifier.objects.create(
            namespace=OrganizationIdentifier.objects.get(namespace__identifier='ars', identifier=ars[:2]).namespace,
            identifier=ars,
            organization=district,
        )
    # pytest has a process-wide i18n context; a fresh management-command process does not.
    token = i18n_context.set(None)
    try:
        yield framework
    finally:
        i18n_context.reset(token)


def test_accounts_work_without_language_context_and_cover_requested_districts(
    account_framework: Framework,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('BISKO_TEST_ACCOUNT_PASSWORD', 'district-demo-password')
    arguments = ['--district', '03451', '--district', '12072', '--district', '03451']
    output = StringIO()
    call_command('provision_bisko_test_accounts', *arguments, '--dry-run', stdout=output)
    assert not User.objects.filter(email__endswith='.fake@kausal.tech').exists()
    assert get_i18n_context() is None
    output = StringIO()
    call_command('provision_bisko_test_accounts', *arguments, stdout=output)
    assert output.getvalue().count('created:') == 16
    assert 'district-demo-password' not in output.getvalue()
    assert FrameworkConfig.objects.filter(framework=account_framework).count() == 3
    assert get_i18n_context() is None
    for ars in ('03451', '12072'):
        district = OrganizationIdentifier.objects.get(namespace__identifier='ars', identifier=ars).organization
        other = OrganizationIdentifier.objects.get(namespace__identifier='ars', identifier='07').organization
        for role in (ObjectRole.ADMIN, ObjectRole.EDITOR):
            user = User.objects.get(email=f'bisko-{ars}-district-{role}.fake@kausal.tech')
            assert user.check_password('district-demo-password')
            grant = OrganizationAccessGrant.objects.get(framework=account_framework, user=user)
            assert grant.organization == district
            assert grant.role == role
            assert user_can_access_organization(account_framework, user, district, action='change')
            assert not user_can_access_organization(account_framework, user, other)
    output = StringIO()
    call_command('provision_bisko_test_accounts', *arguments, stdout=output)
    assert output.getvalue().count('existing:') == 16
    assert User.objects.filter(email__endswith='.fake@kausal.tech').count() == 16


def test_invalid_district_is_rejected_before_creating_accounts(account_framework: Framework) -> None:
    with pytest.raises(CommandError, match='not a Landkreis'):
        call_command('provision_bisko_test_accounts', '--district', '03', stdout=StringIO())
    assert not User.objects.filter(email__endswith='.fake@kausal.tech').exists()
    assert get_i18n_context() is None


def test_earlier_demo_grants_survive_account_reprovisioning(
    account_framework: Framework, monkeypatch: pytest.MonkeyPatch
) -> None:
    for ars in DEMO_MUNICIPALITIES:
        state_identifier = OrganizationIdentifier.objects.get(namespace__identifier='ars', identifier=ars[:2])
        city = state_identifier.organization.add_child(
            name=f'Demo {ars}',
            classification=OrganizationClass.objects.get(identifier='de_municipality'),
        )
        OrganizationIdentifier.objects.create(namespace=state_identifier.namespace, identifier=ars, organization=city)
    monkeypatch.setenv('BISKO_TEST_ACCOUNT_PASSWORD', 'original-demo-password')
    call_command('provision_bisko_test_accounts', stdout=StringIO())
    assert OrganizationAccessGrant.objects.filter(framework=account_framework).count() == 18
    for ars in DEMO_MUNICIPALITIES:
        city = OrganizationIdentifier.objects.get(namespace__identifier='ars', identifier=ars).organization
        for level in ('district-editor', 'municipality-editor'):
            user = User.objects.get(email=f'bisko-{ars[:2]}-{level}.fake@kausal.tech')
            assert user_can_access_organization(account_framework, user, city, action='change')
    output = StringIO()
    call_command('provision_bisko_test_accounts', stdout=output)
    assert output.getvalue().count('existing:') == 12
    assert OrganizationAccessGrant.objects.filter(framework=account_framework).count() == 18
    # Only the expected extra city grants are tolerated.
    user = User.objects.get(email='bisko-03-district-editor.fake@kausal.tech')
    other = OrganizationIdentifier.objects.get(namespace__identifier='ars', identifier='07').organization
    OrganizationAccessGrant.objects.create(framework=account_framework, organization=other, user=user, role=ObjectRole.EDITOR)
    with pytest.raises(CommandError, match='another BISKO grant'):
        call_command('provision_bisko_test_accounts', stdout=StringIO())


def test_only_the_unified_seed_command_is_registered() -> None:
    commands = get_commands()
    assert 'seed_bisko_demo' in commands
    assert 'seed_bisko_aggregation_demo' not in commands
    assert 'load_bisko_demo' not in commands
