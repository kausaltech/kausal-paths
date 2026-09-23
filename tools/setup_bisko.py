# ruff: noqa: INP001
"""Provision BISKO and optionally publish its template or convert existing DB models."""

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from frameworks.models import Framework


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', default='bisko')
    parser.add_argument('--instance', action='append', default=[], help='Attach membership without converting the graph.')
    parser.add_argument(
        '--prepare-from', action='append', default=[], help='Prepare template declarations from verified DB models.'
    )
    parser.add_argument('--publish', action='store_true', help='Publish the template and advance dependent local drafts.')
    parser.add_argument('--reference-instance', help='Use this historical reference-data edition when publishing.')
    parser.add_argument('--convert', action='append', default=[], help='Convert a DB model to the published template.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.instance and args.convert:
        parser.error('--instance and --convert are alternative attachment modes')
    if args.reference_instance and not args.publish:
        parser.error('--reference-instance requires --publish')

    from kausal_common.development.django import init_django

    init_django()
    from django.db import transaction

    from frameworks.provisioning import setup_bisko

    with transaction.atomic():
        framework = setup_bisko(template_identifier=args.template, instance_identifiers=tuple(args.instance))
        print(f'Framework: {framework.identifier}; template: {args.template}; quality scheme: bisko v1 (A/B/C/D).')
        _setup_graphs(framework, args)
        if args.dry_run:
            transaction.set_rollback(True)
            print('Dry run: database changes rolled back.')


def _setup_graphs(framework: Framework, args: argparse.Namespace) -> None:
    from kausal_common.datasets.models import Dataset

    from frameworks.conversion import (
        convert_to_framework,
        declare_local_data_slots,
        prepare_template_inputs,
        share_template_catalogue,
    )
    from frameworks.identity import ensure_municipal_organization
    from nodes.models import InstanceConfig
    from nodes.template_graph import publish_template_instance

    template = framework.template_instance
    assert template is not None
    if args.prepare_from:
        examples = [InstanceConfig.objects.get(identifier=identifier) for identifier in args.prepare_from]
        for change in prepare_template_inputs(framework, examples):
            print(change)
        share_template_catalogue(framework)
    revision = template.live_revision
    if args.publish:
        for change in declare_local_data_slots(framework):
            print(change)
        share_template_catalogue(framework)
        reference_data = {}
        if args.reference_instance:
            source = InstanceConfig.objects.get(identifier=args.reference_instance)
            reference_data = {
                dataset.identifier: dataset
                for dataset in Dataset.objects.for_instance_config(source)
                if dataset.identifier and not dataset.identifier.startswith('kommune/')
            }
        revision = publish_template_instance(template, reference_data=reference_data)
        print(f'Published template revision {revision.pk}')
    if args.convert and revision is None:
        raise ValueError('Publish the template before converting framework instances')
    for identifier in args.convert:
        assert revision is not None
        instance = InstanceConfig.objects.get(identifier=identifier)
        print(f'{identifier}: {convert_to_framework(instance, framework, revision)}')
        instance.refresh_from_db()
        org = ensure_municipal_organization(instance)
        print(f'{identifier}: organization {org.name if org else "unchanged (no AGS)"}')


if __name__ == '__main__':
    main()
