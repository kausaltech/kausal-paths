"""Remove rebuildable inputs no longer referenced by current model configurations."""

import os
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from datasets.manifests import MANIFEST_VERSION, persisted_manifest
from datasets.models import DVCSourceManifest, PreparedDataset
from nodes.models import InstanceConfig, PreferredInstanceSource

if TYPE_CHECKING:
    from django.core.management.base import CommandParser

# Include remote selection: the same Git tree can resolve to different storage.
type SourceKey = tuple[str, str, str, str, int]


class Command(BaseCommand):
    help = 'Find unreferenced dataset caches (dry run by default; scans both draft and published configurations).'

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('--delete', action='store_true', help='Delete the reported stale cache entries')
        parser.add_argument('--older-than-days', type=int, default=7, help='Minimum age of entries to remove (default: 7)')

    def collect_references(self) -> tuple[set[SourceKey], set[str]]:
        sources: set[SourceKey] = set()
        prepared: set[str] = set()
        for config in InstanceConfig.objects.order_by('pk').iterator():
            for source in PreferredInstanceSource:
                with config.enter_instance_context(source=source) as instance:
                    try:
                        context = instance.context
                        context.skip_cache = False
                        context.use_prepared_dataset_cache = True
                        spec = context.dataset_repo_spec
                        if spec is None:
                            continue
                        identifiers = context.get_all_dvc_dataset_ids()
                        remote = spec.dvc_remote or os.getenv('DVC_PANDAS_DVC_REMOTE') or ''
                        sources.update(
                            (spec.url, spec.commit or '', remote, identifier, MANIFEST_VERSION) for identifier in identifiers
                        )
                        # A dry run must not clone Git or create manifests. Missing
                        # rows simply mean there is no prepared key to retain yet.
                        context.dvc_source_manifest = persisted_manifest(
                            spec,
                            identifiers,
                            None,
                            resolve_missing=False,
                        )
                        for node in context.nodes.values():
                            for dataset in node.input_dataset_instances:
                                recipe = dataset.prepared_recipe()
                                if recipe is not None:
                                    prepared.add(recipe.key)
                    finally:
                        instance.clean()
        return sources, prepared

    def handle(self, *args: Any, **options: Any) -> None:
        days = options['older_than_days']
        if days < 1:
            raise CommandError('--older-than-days must be at least 1 (protects concurrent cache population).')
        cutoff = timezone.now() - timedelta(days=days)
        try:
            sources, prepared = self.collect_references()
        except Exception as exc:
            raise CommandError('Reference scan failed; no cache entries were deleted.') from exc
        stale_sources = [
            row.pk
            for row in DVCSourceManifest.objects.filter(updated_at__lt=cutoff).defer('content').iterator()
            if (row.repository_url, row.revision, row.remote_name, row.dataset_identifier, row.format_version) not in sources
        ]
        stale_prepared = PreparedDataset.objects.filter(created_at__lt=cutoff).exclude(key__in=prepared)
        self.stdout.write(f'Stale source manifests: {len(stale_sources)}; prepared datasets: {stale_prepared.count()}')
        if options['delete']:
            # Recheck age in case a concurrent resolver repaired a selected row.
            DVCSourceManifest.objects.filter(pk__in=stale_sources, updated_at__lt=cutoff).delete()
            stale_prepared.delete()
        else:
            self.stdout.write('Dry run; pass --delete to remove these rebuildable entries.')
