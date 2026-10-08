"""Serialized calculation payloads for current and revisioned datasets."""

import hashlib
import json
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypedDict, cast
from uuid import UUID

from django.db import transaction
from django.db.models import F
from django.db.models.fields.json import KeyTextTransform
from django.utils import timezone

from kausal_common.datasets.models import Dataset

from common.validation import blocks_operation
from datasets.shape_domain import SHAPE_SPEC_KEY, CategoryDomainResolver
from datasets.shapes import build_observed_metric_shapes, dump_observed_metric_shapes
from datasets.snapshot import DatasetSnapshot
from datasets.validation import (
    DatasetValidationError,
    InstanceDatasetValidationError,
    dataset_validation_hash,
    dump_violations,
    evaluate_dataset_rules,
    load_violations,
    new_blocking_violations,
    validation_context_hash,
    validation_rules_subquery,
)
from nodes.models import DatasetMaterialization

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from django.db.models import QuerySet
    from django_stubs_ext import WithAnnotations

    from kausal_common.datasets.category_domain import DatasetCategoryDomain

    from datasets.validation import RuleViolation, ValidationRuleFingerprint
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

    dataset = Dataset.objects.select_for_update(of=('self',)).select_related('schema', 'scope_content_type').get(pk=dataset.pk)
    if touch:
        dataset.last_modified_by = user
        dataset.last_modified_at = timezone.now()
        dataset.save(update_fields=['last_modified_by', 'last_modified_at'])

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
            'validation_payload_version': 1,
            'validation_rules_hash': dataset_validation_hash(dataset),
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


class _ValidationContext(TypedDict):
    current_validation_rules: list[ValidationRuleFingerprint]
    current_category_domain: DatasetCategoryDomain | None
    current_shape: str | None
    current_scope_type: int
    current_scope_id: int


def materializations_with_validation_hashes(
    queryset: QuerySet[DatasetMaterialization],
) -> Iterator[tuple[DatasetMaterialization, str]]:
    """Read current shared-schema state in the same query as the materializations."""
    annotated = queryset.annotate(
        current_validation_rules=validation_rules_subquery('dataset__schema_id'),
        current_category_domain=F('dataset__schema__category_domain'),
        current_shape=KeyTextTransform(SHAPE_SPEC_KEY, 'dataset__spec'),
        current_scope_type=F('dataset__scope_content_type_id'),
        current_scope_id=F('dataset__scope_id'),
    )
    # A dataset with a shape has its domain resolved in its instance; one resolver per call
    # resolves each instance once, so the query count does not grow with the datasets.
    domains = CategoryDomainResolver()
    for row in annotated:
        materialization = cast('WithAnnotations[DatasetMaterialization, _ValidationContext]', row)
        domain = materialization.current_category_domain
        if materialization.current_shape:
            instance = domains.instance(materialization.current_scope_type, materialization.current_scope_id)
            domain = domains.for_shape(instance, UUID(materialization.current_shape))
        yield (
            materialization,
            validation_context_hash(materialization.current_validation_rules, domain),
        )


def materialization_is_fresh(
    dataset: Dataset, materialization: DatasetMaterialization, *, validation_hash: str | None = None
) -> bool:
    return (
        materialization.source_modified_at == dataset.last_modified_at
        and materialization.shape_profiles is not None
        and materialization.validation_rules_hash
        == (validation_hash if validation_hash is not None else dataset_validation_hash(dataset))
        and (materialization.validation_payload_version == 1 or not materialization.validation_violations)
    )


def ensure_dataset_materializations(datasets: Iterable[Dataset]) -> dict[int, DatasetMaterialization]:
    """Return fresh materializations, repairing missing or stale derived state atomically."""
    datasets_by_pk = {dataset.pk: dataset for dataset in datasets if not dataset.is_external_placeholder}
    if not datasets_by_pk:
        return {}

    current = {
        materialization.dataset_id: (materialization, validation_hash)
        for materialization, validation_hash in materializations_with_validation_hashes(
            DatasetMaterialization.objects.filter(dataset_id__in=datasets_by_pk)
        )
    }
    materializations = {dataset_id: pair[0] for dataset_id, pair in current.items()}
    stale_ids = {
        dataset_id
        for dataset_id, dataset in datasets_by_pk.items()
        if (pair := current.get(dataset_id)) is None or not materialization_is_fresh(dataset, pair[0], validation_hash=pair[1])
    }
    if not stale_ids:
        return materializations

    with transaction.atomic():
        locked = list(Dataset.objects.select_for_update().filter(pk__in=stale_ids).order_by('pk'))
        locked_materializations = {
            materialization.dataset_id: (materialization, validation_hash)
            for materialization, validation_hash in materializations_with_validation_hashes(
                DatasetMaterialization.objects.select_for_update(of=('self',)).filter(dataset_id__in=stale_ids)
            )
        }
        for dataset in locked:
            pair = locked_materializations.get(dataset.pk)
            if pair is None or not materialization_is_fresh(dataset, pair[0], validation_hash=pair[1]):
                materialization = refresh_dataset_materialization(dataset, touch=False)
            else:
                materialization = pair[0]
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
    from nodes.instance_serialization import build_instance_snapshot

    snapshot = build_instance_snapshot(instance_config)
    datasets = Dataset.objects.filter(uuid__in=[dataset.id for dataset in snapshot.datasets]).exclude(
        uuid__in=[pin.dataset_uuid for pin in snapshot.dataset_revisions],
    )
    materializations = ensure_dataset_materializations(datasets)
    return [
        violation
        for materialization in materializations.values()
        for violation in load_violations(materialization.validation_violations)
    ]


def require_valid_dataset_rules(materializations: Iterable[DatasetMaterialization], *, require_submittable: bool = False) -> None:
    """
    Publication gate for dataset validation rules.

    Raises ``InstanceDatasetValidationError`` when any of the (fresh)
    materializations carries violations blocking the requested operation.
    Submission-only rules do not block ordinary publication.
    """
    violations = [
        violation
        for materialization in materializations
        for violation in load_violations(materialization.validation_violations)
        if blocks_operation(violation.enforcement, 'submit' if require_submittable else 'publish')
    ]
    if violations:
        raise InstanceDatasetValidationError(violations)


def validate_materialization(dataset: Dataset, materialization: DatasetMaterialization) -> None:
    if not materialization_is_fresh(dataset, materialization):
        raise StaleDatasetMaterializationError(
            f'Dataset {dataset.uuid} materialization is stale: '
            f'{materialization.source_modified_at.isoformat()} != {dataset.last_modified_at.isoformat()}',
        )
