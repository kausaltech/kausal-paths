from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

from django.core.management.base import BaseCommand, CommandError

import dvc_pandas

from frameworks.models import FrameworkConfig

if TYPE_CHECKING:
    from django.core.management.base import CommandParser


def _make_repo() -> dvc_pandas.Repository:
    creds = dvc_pandas.RepositoryCredentials(
        git_username=os.getenv('DVC_PANDAS_GIT_USERNAME'),
        git_token=os.getenv('DVC_PANDAS_GIT_TOKEN'),
        git_ssh_public_key_file=os.getenv('DVC_SSH_PUBLIC_KEY_FILE'),
        git_ssh_private_key_file=os.getenv('DVC_SSH_PRIVATE_KEY_FILE'),
    )
    return dvc_pandas.Repository(
        repo_url='https://github.com/kausaltech/dvctest.git',
        dvc_remote='kausal-s3',
        repo_credentials=creds,
    )


class Command(BaseCommand):
    help = 'Populate NZC MeasureDataPoint defaults from the nzc/placeholders_yearly DVC dataset'

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('--framework-config', type=int, metavar='PK', help='Target a single FrameworkConfig by primary key')
        parser.add_argument('--instance', metavar='IDENTIFIER', help='Target a single InstanceConfig by identifier')
        parser.add_argument(
            '--population', type=int, metavar='N', help='Override population (required when create_context is missing)'
        )
        parser.add_argument(
            '--renewmix',
            choices=['low', 'high'],
            help='Override renewable-mix category (required when create_context is missing)',
        )
        parser.add_argument(
            '--temperature',
            choices=['low', 'high'],
            help='Override temperature category (required when create_context is missing)',
        )

    def handle(self, *args, **options) -> None:  # noqa: C901
        fk_pk: int | None = options.get('framework_config')
        instance_id: str | None = options.get('instance')
        pop_override: int | None = options.get('population')
        mix_override: Literal['low', 'high'] | None = options.get('renewmix')
        tmp_override: Literal['low', 'high'] | None = options.get('temperature')

        if fk_pk:
            configs = list(FrameworkConfig.objects.filter(pk=fk_pk))
            if not configs:
                raise CommandError(f'FrameworkConfig with pk={fk_pk} not found')
        elif instance_id:
            from nodes.models import InstanceConfig

            try:
                ic = InstanceConfig.objects.get(identifier=instance_id)
            except InstanceConfig.DoesNotExist:
                raise CommandError(f'InstanceConfig with identifier={instance_id!r} not found')  # noqa: B904
            fc = FrameworkConfig.objects.filter(instance_config=ic).first()
            if fc is None:
                self.stdout.write(self.style.WARNING(f'Instance {instance_id!r} has no associated FrameworkConfig'))
                return
            configs = [fc]
        else:
            configs = list(FrameworkConfig.objects.filter(framework__identifier='nzc'))

        repo = _make_repo()

        for fc in configs:
            create_context = (fc.extra or {}).get('create_context') or {}

            # Apply CLI overrides so configs without create_context can still be populated
            if pop_override:
                create_context['population'] = pop_override
            if mix_override:
                create_context['renewable_mix'] = mix_override
            if tmp_override:
                create_context['temperature'] = tmp_override

            if not all(create_context.get(k) for k in ('population', 'renewable_mix', 'temperature')):
                self.stdout.write(
                    self.style.WARNING(
                        f'Skipping FC {fc.pk} ({fc}): create_context missing '
                        f'(use --population/--renewmix/--temperature to override)'
                    )
                )
                continue

            fc.extra = {**(fc.extra or {}), 'create_context': create_context}
            count = fc.populate_measure_defaults_from_nzc_yearly(repo)
            self.stdout.write(self.style.SUCCESS(f'FC {fc.pk} ({fc}): updated {count} data point(s)'))
