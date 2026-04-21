# ruff: noqa: E402  # noqa: INP001
"""
Idempotent bootstrap script for the CADS framework and landing instance.

Usage:
    python tools/setup_cads.py
"""

from __future__ import annotations

from kausal_common.development.django import init_django

from kausal_common.i18n.pydantic import set_i18n_context

init_django()

import json

from django.utils.translation import override as translation_override
from wagtail.models import Locale, Page

from frameworks.models import Framework
from nodes.defs.instance_defs import InstanceSpec, YearsSpec
from nodes.models import InstanceConfig
from orgs.models import Organization
from pages.models import InstanceRootPage

FRAMEWORK_IDENTIFIER = 'cads'
FRAMEWORK_NAME = 'CADS'
LANDING_INSTANCE_IDENTIFIER = 'cads-landing'
LANDING_INSTANCE_NAME = 'CADS Landing'
LANDING_ORG_NAME = 'CADS'
PRIMARY_LANGUAGE = 'en'


def get_or_create_framework() -> Framework:
    fw, created = Framework.objects.get_or_create(
        identifier=FRAMEWORK_IDENTIFIER,
        defaults=dict(
            name=FRAMEWORK_NAME,
            allow_user_registration=True,
            allow_instance_creation=True,
        ),
    )
    if created:
        print(f'Created framework: {fw}')
    else:
        # Ensure flags are up to date
        updated = False
        if not fw.allow_user_registration:
            fw.allow_user_registration = True
            updated = True
        if not fw.allow_instance_creation:
            fw.allow_instance_creation = True
            updated = True
        if fw.template_instance is None or fw.template_instance.identifier != 'climaville-c4c':
            fw.template_instance = InstanceConfig.objects.get(identifier='climaville-c4c')
            updated = True
        if fw.public_base_fqdn is None:
            fw.public_base_fqdn = 'cads.kausal.tech'
            updated = True
        if updated:
            fw.save(update_fields=['allow_user_registration', 'allow_instance_creation', 'template_instance', 'public_base_fqdn'])
            print(f'Updated framework flags: {fw}')
        else:
            print(f'Framework already exists: {fw}')
    return fw


def get_or_create_organization() -> Organization:
    org = Organization.objects.filter(name=LANDING_ORG_NAME).first()
    if org is not None:
        print(f'Organization already exists: {org}')
        return org
    org = Organization.add_root(name=LANDING_ORG_NAME)
    print(f'Created organization: {org}')
    return org


def get_or_create_landing_instance(org: Organization) -> InstanceConfig:
    spec = InstanceSpec(
        identifier=LANDING_INSTANCE_IDENTIFIER,
        name=LANDING_INSTANCE_NAME,
        owner=LANDING_ORG_NAME,
        primary_language=PRIMARY_LANGUAGE,
        theme_identifier='eu-climate-4-cast',
        years=YearsSpec(reference=2020, min_historical=2018, max_historical=2024, target=2030),
    )
    try:
        ic = InstanceConfig.objects.get(identifier=LANDING_INSTANCE_IDENTIFIER)
        needs_update = ic.spec is None or ic.spec.theme_identifier != spec.theme_identifier
        if needs_update:
            if ic.spec is not None:
                spec.uuid = ic.spec.uuid  # preserve existing UUID
            ic.spec = spec
            ic.save(update_fields=['spec'])
            print(f'Updated instance spec: {ic}')
        else:
            print(f'Instance already exists: {ic}')
    except InstanceConfig.DoesNotExist:
        pass
    else:
        return ic

    ic = InstanceConfig.objects.create(
        name=LANDING_INSTANCE_NAME,
        identifier=LANDING_INSTANCE_IDENTIFIER,
        primary_language=PRIMARY_LANGUAGE,
        other_languages=[],
        organization=org,
        config_source='database',
        spec=spec,
    )
    print(f'Created instance: {ic}')
    return ic


def _landing_body() -> str:
    return json.dumps([
        {
            'type': 'framework_landing',
            'value': {
                'heading': 'Welcome to CADS',
                'body': (
                    '<p>The Climate Action Decision Support Tool helps cities '
                    'assess the emission and economic impacts of climate actions, '
                    'build transparent emission scenarios, and integrate climate '
                    'considerations into decision-making.</p>'
                ),
                'cta_label': 'Get started',
                'cta_url': '/register',
                'framework_identifier': FRAMEWORK_IDENTIFIER,
            },
        }
    ])


def create_landing_root_page(ic: InstanceConfig) -> InstanceRootPage:
    root = Page.get_first_root_node()
    assert root is not None
    body = _landing_body()

    with translation_override(PRIMARY_LANGUAGE):
        locale, _ = Locale.objects.get_or_create(language_code=PRIMARY_LANGUAGE)

        existing = root.get_children().filter(slug=LANDING_INSTANCE_IDENTIFIER).first()
        if existing is not None:
            page = existing.specific
            assert isinstance(page, InstanceRootPage)
            # Update body if content has changed (ignore auto-generated block IDs)
            current = page.body.stream_block.get_prep_value(page.body)
            desired = json.loads(body)
            current_clean = [{k: v for k, v in b.items() if k != 'id'} for b in current]
            desired_clean = [{k: v for k, v in b.items() if k != 'id'} for b in desired]
            if current_clean != desired_clean:
                page.body = body
                page.title = LANDING_INSTANCE_NAME
                page.save()
                print(f'Updated root page body: {page}')
            else:
                print(f'Root page already up to date: {page}')
            return page

        page = root.add_child(
            instance=InstanceRootPage(
                locale=locale,
                title=LANDING_INSTANCE_NAME,
                slug=LANDING_INSTANCE_IDENTIFIER,
                url_path='',
                body=body,
            ),
        )
        print(f'Created root page: {page}')
        return page  # type: ignore[return-value]


def ensure_site(ic: InstanceConfig, root_page: Page) -> None:
    from wagtail.models import Site

    if ic.site is not None:
        print(f'Site already exists: {ic.site}')
        return

    site = Site.objects.create(
        site_name=LANDING_INSTANCE_NAME,
        hostname=f'{LANDING_INSTANCE_IDENTIFIER}.localhost',
        root_page=root_page,
    )
    ic.site = site
    ic.save(update_fields=['site'])
    print(f'Created site: {site}')


def setup_instance_groups(ic: InstanceConfig) -> None:
    if ic.admin_group is not None:
        print('Instance groups already exist')
        return
    ic.create_or_update_instance_groups()
    print('Created instance groups')


def main() -> None:
    get_or_create_framework()
    org = get_or_create_organization()
    with set_i18n_context(PRIMARY_LANGUAGE, []):
        ic = get_or_create_landing_instance(org)
        root_page = create_landing_root_page(ic)
        ensure_site(ic, root_page)
        setup_instance_groups(ic)
    print('CADS setup complete.')


if __name__ == '__main__':
    main()
