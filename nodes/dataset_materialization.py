"""Serialized calculation payloads for current and revisioned datasets."""

import hashlib
import json
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from django.db import transaction
from django.utils import timezone

from kausal_common.datasets.models import Dataset

from nodes.instance_serialization import DatasetSnapshot
from nodes.models import DatasetMaterialization

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from datasets.validation import RuleViolation
    from nodes.models import InstanceConfig
    from users.models import User


class StaleDatasetMaterializationError(RuntimeError):
    """The current serialized payload does not represent the live dataset."""


def _canonical_content(content: dict[str, Any]) -> bytes:
    return json.dumps(content, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode()


def hash_dataset_content(content: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_content(content)).hexdigest()


def serialize_dataset(dataset: Dataset) -> dict[str, Any]:
    return DatasetSnapshot.from_model(dataset).model_dump(mode='json')


def refresh_dataset_materialization(
    dataset: Dataset,
    *,
    user: User | None = None,
    touch: bool = True,
    enforce_edit_rules: bool = False,
) -> DatasetMaterialization:
    """
    Serialize the final state of one locked dataset and advance its generation.

    Callers must invoke this once at the end of a logical write operation,
    inside the transaction that changed the related rows.

    Validation-rule violations are re-evaluated and persisted alongside the
    payload. With ``enforce_edit_rules``, the refresh raises
    ``DatasetValidationError`` (rolling back the write) when the operation
    introduced violations of ``block_edit`` rules that the previous
    materialization did not have — user-facing edit boundaries pass this;
    imports and backfills do not.
    """
    if not transaction.get_connection().in_atomic_block:
        msg = 'Dataset materialization refresh requires an atomic write boundary'
        raise RuntimeError(msg)

    from datasets.validation import (
        DatasetValidationError,
        dump_violations,
        evaluate_dataset_rules,
        load_violations,
        new_blocking_violations,
    )

    dataset = Dataset.objects.select_for_update().get(pk=dataset.pk)
    if touch:
        dataset.last_modified_by = user
        dataset.last_modified_at = timezone.now()
        dataset.save(update_fields=['last_modified_by', 'last_modified_at'])

    from nodes.dataset_shape import build_observed_metric_shapes, dump_observed_metric_shapes

    content = serialize_dataset(dataset)
    shape_profiles = dump_observed_metric_shapes(build_observed_metric_shapes(dataset))
    content_hash = hash_dataset_content(content)
    violations = evaluate_dataset_rules(dataset)
    existing = DatasetMaterialization.objects.select_for_update().filter(dataset=dataset).first()
    if enforce_edit_rules:
        baseline = load_violations(existing.validation_violations) if existing is not None else []
        introduced = new_blocking_violations(baseline, violations)
        if introduced:
            raise DatasetValidationError(introduced)
    generation = existing.generation + 1 if existing is not None else 1
    materialization, _ = DatasetMaterialization.objects.update_or_create(
        dataset=dataset,
        defaults={
            'content': content,
            'content_hash': content_hash,
            'generation': generation,
            'shape_profiles': shape_profiles,
            'validation_violations': dump_violations(violations),
            'forecast_from': (dataset.spec or {}).get('forecast_from'),
            'source_modified_at': dataset.last_modified_at,
        },
    )
    dataset.clear_scope_instance_cache()
    return materialization


def materialize_dataset(dataset: Dataset, *, user: User | None = None) -> DatasetMaterialization:
    """Standalone atomic entry point used by backfills and non-editor writers."""
    with transaction.atomic():
        return refresh_dataset_materialization(dataset, user=user, touch=False)


def materialization_is_fresh(dataset: Dataset, materialization: DatasetMaterialization) -> bool:
    return materialization.source_modified_at == dataset.last_modified_at and materialization.shape_profiles is not None


def ensure_dataset_materializations(datasets: Iterable[Dataset]) -> dict[int, DatasetMaterialization]:
    """Return fresh materializations, repairing missing or stale derived state atomically."""
    datasets_by_pk = {dataset.pk: dataset for dataset in datasets if not dataset.is_external_placeholder}
    if not datasets_by_pk:
        return {}

    materializations = {
        materialization.dataset_id: materialization
        for materialization in DatasetMaterialization.objects.filter(dataset_id__in=datasets_by_pk)
    }
    stale_ids = {
        dataset_id
        for dataset_id, dataset in datasets_by_pk.items()
        if (materialization := materializations.get(dataset_id)) is None or not materialization_is_fresh(dataset, materialization)
    }
    if not stale_ids:
        return materializations

    with transaction.atomic():
        locked = list(Dataset.objects.select_for_update().filter(pk__in=stale_ids).order_by('pk'))
        locked_materializations = {
            materialization.dataset_id: materialization
            for materialization in DatasetMaterialization.objects.select_for_update().filter(dataset_id__in=stale_ids)
        }
        for dataset in locked:
            materialization = locked_materializations.get(dataset.pk)
            if materialization is None or not materialization_is_fresh(dataset, materialization):
                materialization = refresh_dataset_materialization(dataset, touch=False)
            materializations[dataset.pk] = materialization
    return materializations


@contextmanager
def datasets_change(datasets: Iterable[Dataset], *, user: User | None = None) -> Iterator[list[Dataset]]:
    """
    Atomic write boundary for one logical operation affecting multiple datasets.

    This is a user-facing edit boundary, so ``block_edit`` validation rules
    are enforced: an operation that introduces new violations raises
    ``DatasetValidationError`` and rolls back.
    """
    with transaction.atomic():
        dataset_pks = sorted({dataset.pk for dataset in datasets})
        locked = list(Dataset.objects.select_for_update().filter(pk__in=dataset_pks).order_by('pk'))
        yield locked
        for dataset in locked:
            refresh_dataset_materialization(dataset, user=user, enforce_edit_rules=True)


@contextmanager
def dataset_change(dataset: Dataset, *, user: User | None = None) -> Iterator[Dataset]:
    """Atomic write boundary for one logical operation affecting a dataset."""
    with datasets_change([dataset], user=user) as locked:
        yield locked[0]


def collect_instance_dataset_violations(instance_config: InstanceConfig) -> list[RuleViolation]:
    """
    Collect current validation-rule violations across the instance's bound datasets.

    Reads the persisted violation sets, repairing stale materializations
    first — the same dataset scope the publication gate enforces.
    """
    from datasets.validation import load_violations
    from nodes.models import DatasetPort

    dataset_ids = list(
        DatasetPort.objects.filter(instance=instance_config).order_by().values_list('dataset_id', flat=True).distinct(),
    )
    datasets = Dataset.objects.filter(pk__in=dataset_ids)
    materializations = ensure_dataset_materializations(datasets)
    return [
        violation
        for materialization in materializations.values()
        for violation in load_violations(materialization.validation_violations)
    ]


def require_valid_dataset_rules(materializations: Iterable[DatasetMaterialization]) -> None:
    """
    Publication gate for dataset validation rules.

    Raises ``InstanceDatasetValidationError`` when any of the (fresh)
    materializations carries validation-rule violations; both enforcement
    tiers keep a draft editable but block publication.
    """
    from datasets.validation import InstanceDatasetValidationError, load_violations

    violations = [
        violation for materialization in materializations for violation in load_violations(materialization.validation_violations)
    ]
    if violations:
        raise InstanceDatasetValidationError(violations)


def validate_materialization(dataset: Dataset, materialization: DatasetMaterialization) -> None:
    if not materialization_is_fresh(dataset, materialization):
        raise StaleDatasetMaterializationError(
            f'Dataset {dataset.uuid} materialization is stale: '
            f'{materialization.source_modified_at.isoformat()} != {dataset.last_modified_at.isoformat()}',
        )
