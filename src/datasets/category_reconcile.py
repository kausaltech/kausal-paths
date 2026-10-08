"""
Bring an instance's stored dimension categories in line with what its model declares.

Sync creates and updates categories but never deletes one (``delete_stale`` is off on every
path that runs in practice, and on it a category holding data points raises ``ProtectedError``
partway through). So a category dropped from a module stays in every database that ever synced
it, and a renamed category leaves its data behind on the old row. This module is the explicit
step that resolves both, and refuses rather than guesses:

- A stored category the model no longer declares, whose identifier is an **alias** of a declared
  category, is **merged** into that category: its data points move over and the old row goes.
  That is how an id is renamed without losing data -- keep the old id as an alias.
- Any other undeclared category is **deleted**, but only when nothing holds it.

Categories are matched by identifier, never by UUID: UUIDs differ between databases.

What holds a category, and so refuses the change:

- data points, for a delete (a merge moves them, unless the target already has values in the
  same dataset -- combining two populated cells is not this module's business);
- a schema's category domain, a validation rule or a plausibility selection naming its UUID.
  Those are compiled from identifiers on sync, so re-syncing after the model change is the fix.

Published revisions are unaffected: they compute from their pinned dataset revisions, which are
frozen payloads, not from the live rows.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from django.db import models, transaction
from django.db.models.functions import Cast

from kausal_common.datasets.models import (
    DataPointDimensionCategory,
    Dataset,
    DatasetMetricValidationRule,
    DatasetSchema,
    DimensionCategory,
    DimensionScope,
)

from datasets.materialization import refresh_dataset_materialization
from datasets.models import DatasetMetricPlausibilityRange

if TYPE_CHECKING:
    from collections.abc import Iterable

    from kausal_common.datasets.models import Dimension as DimensionModel

    from nodes.dimensions import Dimension
    from nodes.models import InstanceConfig


@dataclass
class CategoryChange:
    dimension: DimensionModel
    dimension_identifier: str
    category: DimensionCategory
    action: Literal['merge', 'delete']
    target: DimensionCategory | None = None
    data_points: int = 0
    datasets: list[Dataset] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)

    def describe(self) -> str:
        what = f'{self.dimension_identifier}/{self.category.identifier}'
        if self.action == 'merge':
            assert self.target is not None
            what = f'merge {what} -> {self.target.identifier} ({self.data_points} data points)'
        else:
            what = f'delete {what}'
        if self.refusals:
            return f'{what}: REFUSED ({"; ".join(self.refusals)})'
        return what


def _uuid_references(category: DimensionCategory) -> list[str]:
    """Name the JSON payloads that refer to the category by UUID."""
    uuid = str(category.uuid)

    def naming(model: type[models.Model], column: str) -> list[str]:
        as_text = Cast(column, models.TextField())
        rows = model._default_manager.annotate(_text=as_text).filter(_text__contains=uuid)
        return [str(row_uuid) for row_uuid in rows.values_list('uuid', flat=True)]

    return [
        *(f'category domain of schema {u}' for u in naming(DatasetSchema, 'category_domain')),
        *(f'validation rule {u}' for u in naming(DatasetMetricValidationRule, 'rule')),
        *(f'plausibility range {u}' for u in naming(DatasetMetricPlausibilityRange, 'selection')),
    ]


def _datasets_with_points(category: DimensionCategory) -> dict[int, Dataset]:
    ids = (
        DataPointDimensionCategory.objects
        .filter(dimension_category=category)
        .values_list('data_point__dataset_id', flat=True)
        .distinct()
    )
    return {ds.pk: ds for ds in Dataset.objects.filter(pk__in=list(ids))}


def plan_dimension(dimension_identifier: str, stored: DimensionModel, declared: Dimension) -> list[CategoryChange]:
    """Plan the changes that make the stored categories of one dimension match `declared`."""
    declared_ids = declared.get_cat_ids()
    alias_target = {alias: cat.id for cat in declared.categories for alias in cat.aliases}
    by_identifier = {cat.identifier: cat for cat in stored.categories.all() if cat.identifier is not None}
    changes: list[CategoryChange] = []
    for identifier, category in sorted(by_identifier.items()):
        if identifier in declared_ids:
            continue
        sources = _datasets_with_points(category)
        target_id = alias_target.get(identifier)
        change = CategoryChange(
            dimension=stored,
            dimension_identifier=dimension_identifier,
            category=category,
            action='merge' if target_id is not None else 'delete',
            data_points=DataPointDimensionCategory.objects.filter(dimension_category=category).count(),
            datasets=list(sources.values()),
        )
        if target_id is not None:
            change.target = by_identifier.get(target_id)
            if change.target is None:
                change.refusals.append(f'target {target_id} has no row yet; sync the instance first')
            else:
                clash = set(sources) & set(_datasets_with_points(change.target))
                if clash:
                    names = ', '.join(sorted(str(sources[pk].identifier) for pk in clash))
                    change.refusals.append(f'both categories have values in {names}')
        elif change.data_points:
            names = ', '.join(sorted(str(ds.identifier) for ds in sources.values()))
            change.refusals.append(f'{change.data_points} data points in {names}')
        change.refusals.extend(_uuid_references(category))
        changes.append(change)
    return changes


def stored_dimensions(instance: InstanceConfig, identifiers: Iterable[str] | None = None) -> dict[str, DimensionModel]:
    """Return the dimension rows scoped to the instance, by the identifier the scope gives them."""
    scopes = DimensionScope.objects.for_instance_config(instance).select_related('dimension')
    if identifiers is not None:
        scopes = scopes.filter(identifier__in=list(identifiers))
    return {str(scope.identifier): scope.dimension for scope in scopes}


def plan_instance(instance: InstanceConfig, identifiers: Iterable[str] | None = None) -> list[CategoryChange]:
    """Plan for every dimension of the instance that its model declares."""
    declared = instance.get_instance().context.dimensions
    changes: list[CategoryChange] = []
    for identifier, stored in sorted(stored_dimensions(instance, identifiers).items()):
        if identifier in declared:
            changes.extend(plan_dimension(identifier, stored, declared[identifier]))
    return changes


@transaction.atomic
def apply_changes(changes: list[CategoryChange]) -> list[Dataset]:
    """Apply a fully actionable plan; return the datasets whose data points moved."""
    refused = [change for change in changes if change.refusals]
    if refused:
        raise ValueError('Refusing category reconcile:\n' + '\n'.join(change.describe() for change in refused))
    touched: dict[int, Dataset] = {}
    for change in changes:
        if change.action == 'merge':
            assert change.target is not None
            DataPointDimensionCategory.objects.filter(dimension_category=change.category).update(dimension_category=change.target)
            touched.update((ds.pk, ds) for ds in change.datasets)
        change.category.delete()
    for dataset in touched.values():
        refresh_dataset_materialization(dataset, touch=False)
    return list(touched.values())
