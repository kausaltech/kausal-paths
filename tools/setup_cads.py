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

from frameworks.models import Framework, FrameworkConfig
from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.models import InstanceConfig, InstanceHostname
from nodes.spec_export import sync_instance_to_db
from orgs.models import Organization
from pages.models import InstanceRootPage

FRAMEWORK_IDENTIFIER = 'cads'
FRAMEWORK_NAME = 'CADS'
SOURCE_TEMPLATE_IDENTIFIER = 'aarhus-c4c'
TEMPLATE_INSTANCE_IDENTIFIER = 'climaville-c4c'
LANDING_INSTANCE_IDENTIFIER = 'cads-landing'
LANDING_INSTANCE_NAME = 'CADS'
LANDING_ORG_NAME = 'CADS'
PRIMARY_LANGUAGE = 'en'

BASE_FQDN = 'cads.kausal.tech'
OLD_BASE_FQDNS = ('cads.paths.kausal.tech', 'cads.kausal.dev')


def get_or_create_framework() -> Framework:
    template_instance = InstanceConfig.objects.get(identifier=TEMPLATE_INSTANCE_IDENTIFIER)
    fw, created = Framework.objects.get_or_create(
        identifier=FRAMEWORK_IDENTIFIER,
        defaults=dict(
            name=FRAMEWORK_NAME,
            allow_user_registration=True,
            allow_instance_creation=True,
            enable_user_management=True,
            template_instance=template_instance,
            public_base_fqdn=BASE_FQDN,
            use_instance_subdomains=False,
        ),
    )
    if created:
        print(f'Created framework: {fw}')
    else:
        updated_fields: list[str] = []
        if not fw.allow_user_registration:
            fw.allow_user_registration = True
            updated_fields.append('allow_user_registration')
        if not fw.allow_instance_creation:
            fw.allow_instance_creation = True
            updated_fields.append('allow_instance_creation')
        if not fw.enable_user_management:
            fw.enable_user_management = True
            updated_fields.append('enable_user_management')
        if fw.template_instance is None or fw.template_instance.identifier != TEMPLATE_INSTANCE_IDENTIFIER:
            fw.template_instance = template_instance
            updated_fields.append('template_instance')
        if fw.public_base_fqdn != BASE_FQDN:
            fw.public_base_fqdn = BASE_FQDN
            updated_fields.append('public_base_fqdn')
        if fw.use_instance_subdomains:
            fw.use_instance_subdomains = False
            updated_fields.append('use_instance_subdomains')
        accept_invitation_url = 'https://{base_fqdn}/auth/register?framework={fwid}&invitation_code={code}'.format(
            base_fqdn=BASE_FQDN, code='{code}', fwid=fw.identifier
        )
        if fw.accept_invitation_url != accept_invitation_url:
            fw.accept_invitation_url = accept_invitation_url
            updated_fields.append('accept_invitation_url')
        if updated_fields:
            fw.save(update_fields=updated_fields)
            print(f'Updated framework  fields ({", ".join(updated_fields)}): {fw}')
        else:
            print(f'Framework already exists: {fw}')
    return fw


def set_root_instance(fw: Framework, root_ic: InstanceConfig) -> None:
    if fw.root_instance_id == root_ic.pk:
        return
    fw.root_instance = root_ic
    fw.save(update_fields=['root_instance'])
    print(f'Updated framework root instance: {root_ic}')


def enable_user_management_on_cads_instances(fw: Framework) -> None:
    instances = InstanceConfig.objects.filter(framework_config__framework=fw)
    flipped = 0
    for ic in instances:
        spec = ic.ensure_spec()
        if spec.features.enable_user_management:
            continue
        spec.features.enable_user_management = True
        ic.spec = spec
        ic.save(update_fields=['spec'])
        print(f'Enabled user_management on {ic}')
        flipped += 1
    if not flipped:
        print('All CADS instances already have user_management enabled')


def _desired_base_path(fw: Framework, ic: InstanceConfig) -> str:
    return '' if ic == fw.root_instance else f'/{ic.uuid}'


def set_instance_hostnames(fw: Framework):
    for ic in InstanceConfig.objects.filter(framework_config__framework=fw):
        desired_base_path = _desired_base_path(fw, ic)
        old_hostnames = list(ic.hostnames.filter(hostname__in=OLD_BASE_FQDNS))
        ich = ic.hostnames.filter(hostname=BASE_FQDN).first()
        if ich is None:
            if old_hostnames:
                ich = old_hostnames.pop(0)
                ich.hostname = BASE_FQDN
                ich.base_path = desired_base_path
                ich.save(update_fields=['hostname', 'base_path'])
                print(f'Migrated instance hostname: {ich}')
            else:
                ich = InstanceHostname.objects.create(instance=ic, hostname=BASE_FQDN, base_path=desired_base_path)
                print(f'Created instance hostname: {ich}')
        elif ich.base_path != desired_base_path:
            ich.base_path = desired_base_path
            ich.save(update_fields=['base_path'])
            print(f'Updated instance hostname base path: {ich}')
        for old_hostname in old_hostnames:
            print(f'Removed old instance hostname: {old_hostname}')
            old_hostname.delete()


def ensure_template_datasets() -> None:
    from django.contrib.contenttypes.models import ContentType

    from kausal_common.datasets.models import Dataset

    from nodes.instance_serialization import export_instance, import_instance_datasets

    source = InstanceConfig.objects.get(identifier=SOURCE_TEMPLATE_IDENTIFIER)
    target = InstanceConfig.objects.get(identifier=TEMPLATE_INSTANCE_IDENTIFIER)
    source_export = export_instance(source)
    source_datasets = [ds for ds in source_export.datasets if ds.identifier is not None and ds.data is not None]
    source_dataset_ids = {ds.identifier for ds in source_datasets if ds.identifier is not None}
    if not source_dataset_ids:
        print(f'No real datasets found in source template: {source}')
        return

    ic_ct = ContentType.objects.get_for_model(target)
    existing_dataset_ids = set(
        Dataset.objects
        .filter(
            scope_content_type=ic_ct,
            scope_id=target.pk,
            identifier__in=source_dataset_ids,
            is_external_placeholder=False,
            data_points__isnull=False,
        )
        .distinct()
        .values_list('identifier', flat=True)
    )
    missing_dataset_ids = source_dataset_ids - existing_dataset_ids
    if not missing_dataset_ids:
        print(f'Template datasets already copied: {target}')
        ensure_template_dataset_ports(source, target, source_dataset_ids)
        return

    copied = import_instance_datasets(
        target,
        [ds for ds in source_datasets if ds.identifier in missing_dataset_ids],
        rewire_dataset_ports=True,
        delete_superseded_placeholders=True,
        create_missing_dimensions=True,
    )
    print(f'Copied {len(copied)} dataset(s) from {source.identifier} to {target.identifier}')
    ensure_template_dataset_ports(source, target, source_dataset_ids)


def _dataset_port_key(port) -> tuple[str, str, str, str]:
    return (
        port.node.identifier,
        port.dataset.identifier,
        port.metric.name or str(port.metric.uuid),
        port.spec.model_dump_json(exclude_defaults=True, exclude_none=True),
    )


def ensure_template_dataset_ports(source: InstanceConfig, target: InstanceConfig, dataset_ids: set[str]) -> None:
    from django.contrib.contenttypes.models import ContentType

    from kausal_common.datasets.models import Dataset

    from nodes.models import DatasetPort, NodeConfig

    if not dataset_ids:
        return

    target_nodes = {
        node.identifier: node
        for node in NodeConfig.objects.filter(
            instance=target,
            identifier__in=DatasetPort.objects.filter(instance=source, dataset__identifier__in=dataset_ids).values_list(
                'node__identifier',
                flat=True,
            ),
        )
    }
    target_datasets = {
        dataset.identifier: dataset
        for dataset in Dataset.objects.filter(
            scope_content_type=ContentType.objects.get_for_model(target),
            scope_id=target.pk,
            identifier__in=dataset_ids,
            is_external_placeholder=False,
        ).select_related('schema')
        if dataset.identifier is not None
    }
    existing_port_keys = {
        _dataset_port_key(port)
        for port in DatasetPort.objects.filter(instance=target, dataset__identifier__in=dataset_ids).select_related(
            'node',
            'dataset',
            'metric',
        )
    }
    source_ports = DatasetPort.objects.filter(instance=source, dataset__identifier__in=dataset_ids).select_related(
        'node',
        'dataset',
        'metric',
    )

    missing_ports: list[DatasetPort] = []
    for source_port in source_ports:
        key = _dataset_port_key(source_port)
        if key in existing_port_keys:
            continue
        target_node = target_nodes.get(source_port.node.identifier)
        if not source_port.dataset.identifier:
            continue
        target_dataset = target_datasets.get(source_port.dataset.identifier)
        if target_node is None or target_dataset is None:
            continue
        assert target_dataset.schema is not None
        target_metric = target_dataset.schema.metrics.filter(name=source_port.metric.name).first()
        if target_metric is None:
            raise ValueError(
                f'Cannot copy dataset port for {source_port.dataset.identifier!r}; '
                + f'metric {source_port.metric.name!r} is missing in {target.identifier!r}'
            )
        missing_ports.append(
            DatasetPort(
                instance=target,
                node=target_node,
                port_id=source_port.port_id,
                dataset=target_dataset,
                metric=target_metric,
                spec=source_port.spec,
            )
        )

    if not missing_ports:
        print(f'Template dataset ports already copied: {target}')
        return
    DatasetPort.objects.bulk_create(missing_ports)
    print(f'Copied {len(missing_ports)} dataset port(s) from {source.identifier} to {target.identifier}')


def get_or_create_organization() -> Organization:
    org = Organization.objects.filter(name=LANDING_ORG_NAME).first()
    if org is not None:
        print(f'Organization already exists: {org}')
        return org
    org = Organization.add_root(name=LANDING_ORG_NAME)
    print(f'Created organization: {org}')
    return org


def get_or_create_landing_instance(org: Organization) -> InstanceConfig:
    spec = InstanceModelSpec(
        theme_identifier='eu-climate-4-cast',
        years=YearsSpec(reference=2020, min_historical=2018, max_historical=2024, target=2030),
    )
    try:
        ic = InstanceConfig.objects.get(identifier=LANDING_INSTANCE_IDENTIFIER)
        needs_update = ic.spec is None or ic.spec.theme_identifier != spec.theme_identifier
        if needs_update:
            # uuid lives on the InstanceConfig column, so it's preserved across spec updates.
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


def ensure_root_page(ic: InstanceConfig, root_page: Page) -> None:
    if ic.root_page_id == root_page.pk:
        print(f'Instance root page already set: {root_page}')
        return

    ic.root_page = root_page
    ic.save(update_fields=['root_page'])
    print(f'Updated instance root page: {root_page}')


def init_framework_instance(fw: Framework, ic: InstanceConfig) -> FrameworkConfig:
    ich = ic.hostnames.filter(hostname=BASE_FQDN).first()
    base_path = _desired_base_path(fw, ic)
    if ich is None:
        ich = InstanceHostname.objects.create(instance=ic, hostname=BASE_FQDN, base_path=base_path)
        print(f'Created instance hostname: {ich}')

    fwc = FrameworkConfig.objects.filter(instance_config=ic).first()
    if fwc is not None:
        print(f'Framework config already exists: {fwc}')
        return fwc
    fw = Framework.objects.get(identifier=FRAMEWORK_IDENTIFIER)
    spec = ic.ensure_spec()
    fwc = FrameworkConfig.objects.create(
        framework=fw,
        instance_config=ic,
        organization_name=LANDING_ORG_NAME,
        baseline_year=spec.years.reference or 2020,
        target_year=spec.years.target,
    )
    print(f'Created framework config: {fwc}')
    if ic.admin_group is None:
        ic.create_or_update_instance_groups()
        print('Created instance groups')

    return fwc


def main() -> None:
    fw = get_or_create_framework()
    # ensure_template_datasets()
    org = get_or_create_organization()
    with set_i18n_context(PRIMARY_LANGUAGE, []):
        landing_ic = get_or_create_landing_instance(org)
        set_root_instance(fw, landing_ic)
        root_page = create_landing_root_page(landing_ic)
        ensure_root_page(landing_ic, root_page)
        init_framework_instance(fw, landing_ic)
        template_ic = InstanceConfig.objects.get(identifier=TEMPLATE_INSTANCE_IDENTIFIER)
        if not template_ic.spec or not template_ic.spec.dimensions:
            sync_instance_to_db(template_ic.identifier)
        init_framework_instance(fw, template_ic)
    enable_user_management_on_cads_instances(fw)
    set_instance_hostnames(fw)
    print('CADS setup complete.')


if __name__ == '__main__':
    main()
