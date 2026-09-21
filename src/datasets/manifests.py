"""Shared, revision-pinned DVC source manifests."""

import os
import re
from typing import TYPE_CHECKING

from pydantic import ValidationError

from dvc_pandas import DatasetManifest, RepositoryManifest
from loguru import logger

from datasets.models import DVCSourceManifest

if TYPE_CHECKING:
    from collections.abc import Callable

    from dvc_pandas import Repository

    from nodes.defs.instance_defs import DatasetRepoSpec

MANIFEST_VERSION = 1


def persisted_manifest(
    spec: DatasetRepoSpec,
    identifiers: set[str],
    repository: Callable[[], Repository] | None,
    *,
    resolve_missing: bool = True,
) -> RepositoryManifest | None:
    """Batch-read pinned metadata, resolving only missing or invalid datasets."""
    if not identifiers or spec.commit is None or re.fullmatch(r'[0-9a-f]{40}', spec.commit) is None:
        return None
    key = {
        'repository_url': spec.url,
        'revision': spec.commit,
        'remote_name': spec.dvc_remote or os.getenv('DVC_PANDAS_DVC_REMOTE') or '',
        'format_version': MANIFEST_VERSION,
    }

    datasets: dict[str, DatasetManifest] = {}
    for row in DVCSourceManifest.objects.filter(**key, dataset_identifier__in=identifiers):
        try:
            manifest = DatasetManifest.model_validate(row.content)
        except ValidationError:
            continue
        if (
            manifest.repository_url == spec.url
            and manifest.revision == spec.commit
            and manifest.identifier == row.dataset_identifier
        ):
            datasets[manifest.identifier] = manifest
    missing = identifiers - datasets.keys()
    if missing and resolve_missing:
        if repository is None:
            raise ValueError('A repository factory is required to resolve missing manifests')
        repo = repository()
        try:
            resolved = repo.get_manifest(sorted(missing))
        except ValueError as error:
            # Manifest support is narrower than DVC's transport support (e.g.
            # HTTPS remotes). Metadata acceleration must not prevent loading
            # datasets through the repository's ordinary DVC path.
            logger.warning('Unable to resolve DVC manifests for {}; using repository loading: {}', spec.url, error)
            return None
        if resolved.repository_url != spec.url or resolved.revision != spec.commit:
            raise ValueError('Resolved DVC manifest does not match the pinned repository revision')
        if not missing <= resolved.datasets.keys():
            raise ValueError('Resolved DVC manifest is missing requested datasets')
        # One upsert per batch; disjoint workers cannot overwrite each other's datasets.
        DVCSourceManifest.objects.bulk_create(
            [
                DVCSourceManifest(**key, dataset_identifier=identifier, content=manifest.model_dump(mode='json'))
                for identifier, manifest in resolved.datasets.items()
            ],
            update_conflicts=True,
            unique_fields=[*key, 'dataset_identifier'],
            update_fields=['content', 'updated_at'],
        )
        datasets.update(resolved.datasets)
    return RepositoryManifest(repository_url=spec.url, revision=spec.commit, datasets=datasets)
