"""
A dataset as one edition of an instance reads it.

An instance source -- its draft, its published revision, its YAML -- reads each dataset
from one place: the pinned dataset revision when the graph pins one, the live row
otherwise. A `DatasetEdition` is that choice made once, so that everything describing a
dataset (the GraphQL `Dataset`, the data-entry views, rule and plausibility evaluation)
reads the same structure and content for the same source.

The row stays with the edition in every source, because it carries the dataset's identity
and permissions. It supplies content only when the edition is live: the draft's, unpinned.
"""

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

from django.db.models import prefetch_related_objects
from wagtail.models import Revision

import polars as pl
from pint.errors import UndefinedUnitError

from kausal_common.datasets.models import Dataset
from kausal_common.i18n.pydantic import set_i18n_context

from datasets.catalogue import dataset_meta_from_model
from datasets.coordinates import DatasetCoordinate, DatasetCoordinateIndex
from datasets.materialization import materialization_is_fresh, materializations_with_validation_hashes, serialize_dataset
from datasets.plausibility import (
    DatasetPlausibilityCells,
    PlausibilityFinding,
    _log_unreadable,
    _observed_population,
    applicable_plausibility_ranges,
    evaluate_plausibility_cells,
)
from datasets.plausibility_history import derive_history_ranges, eligible
from datasets.runtime import DBDataset
from datasets.shape_domain import CategoryDomainResolver
from datasets.snapshot import DatasetSnapshot
from datasets.validation import RuleViolation, evaluate_closed_domain, evaluate_rule
from frameworks.models import Framework
from frameworks.qualifiers import attach_evidence_qualifiers
from nodes.defs.graph import DimensionCategoryMeta, DimensionMeta
from nodes.instance_serialization import _dimension_catalog_for
from nodes.models import DatasetMaterialization, InstanceConfig, PreferredInstanceSource
from nodes.snapshot_base import translated_string_from_model

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from uuid import UUID

    from kausal_common.users import UserOrAnon

    from common.qualifiers import QualifierCatalog
    from datasets.models import DatasetMetricPlausibilityRange
    from nodes.defs.graph import DatasetMeta
    from nodes.instance_graph import InstanceGraph


@dataclass
class EditionFrame:
    """An edition's values as a plain frame, with its dimension columns as category labels."""

    frame: pl.DataFrame
    dimensions: dict[str, UUID]
    """Dimension uuid by frame column."""
    categories: dict[tuple[str, str], UUID]
    """Category uuid by (frame column, category label)."""

    @property
    def dim_cols(self) -> list[str]:
        return list(self.dimensions)

    def coordinates(self, row: Mapping[str, object]) -> tuple[tuple[UUID, UUID], ...]:
        return tuple(
            sorted(
                (dim, self.categories[column, str(row[column])])
                for column, dim in self.dimensions.items()
                if row.get(column) is not None
            )
        )

    def dataset_coordinates(self, categories: Mapping[str, str]) -> list[DatasetCoordinate]:
        return [
            DatasetCoordinate(
                dimension=column,
                category=label,
                dimension_uuid=self.dimensions[column],
                category_uuid=self.categories[column, label],
            )
            for column, label in categories.items()
        ]

    def domain(self, meta: DatasetMeta) -> dict[UUID, dict[str, str]]:
        """Return the labels of each combination of the dataset's category domain, by frame column."""
        labels = {(dim, uuid): label for (column, label), uuid in self.categories.items() for dim in [self.dimensions[column]]}
        return {
            item.id: {
                column: labels[dim, item.categories[dim]]
                for column, dim in self.dimensions.items()
                if dim in item.categories and (dim, item.categories[dim]) in labels
            }
            for item in meta.category_domain.combinations
        }

    def with_frame(self, frame: pl.DataFrame) -> EditionFrame:
        return EditionFrame(frame, self.dimensions, self.categories)


def dimension_catalog(
    meta: DatasetMeta,
    catalog: Mapping[UUID, DimensionMeta],
    config: InstanceConfig,
    row: Dataset | None,
    instance_dimensions: Mapping[UUID, DimensionMeta] | None = None,
) -> Mapping[UUID, DimensionMeta]:
    """
    Name a dataset's dimensions: from the instance's catalog, else from the dataset's own schema.

    A dimension the instance does not scope still names a column of a live dataset, as
    `DBDataset.dimension_columns` names it, so the dataset reads as it did from its rows.
    """
    if all(id in catalog for id in meta.declared_dimension_ids):
        return catalog
    if instance_dimensions is None:
        instance_dimensions = {dimension.id: dimension for dimension in _dimension_catalog_for(config)}
    merged = dict(instance_dimensions)
    if row is not None and not all(id in merged for id in meta.declared_dimension_ids):
        for dimension, column in DBDataset.dimension_columns(row):
            merged.setdefault(
                dimension.uuid,
                DimensionMeta(
                    id=dimension.uuid,
                    identifier=column,
                    label=translated_string_from_model(dimension, 'name', config.primary_language),
                    categories=tuple(
                        DimensionCategoryMeta(
                            id=category.uuid,
                            identifier=category.identifier,
                            label=translated_string_from_model(category, 'label', config.primary_language),
                            order=category.order,
                        )
                        for category in dimension.categories.all()
                    ),
                ),
            )
    return {**merged, **catalog}


@dataclass(eq=False)
class DatasetEdition:
    """One dataset as one instance source reads it."""

    meta: DatasetMeta
    row: Dataset | None
    """The dataset's row: its identity and permissions in every edition, its content only when `is_live`."""
    is_live: bool
    """Whether the content is the row's current state: the draft's dataset, unpinned."""
    revision: Revision[Dataset] | None
    """The pinned dataset revision this edition reads, if any."""
    dimensions: Mapping[UUID, DimensionMeta]
    """The catalog the dataset's dimensions are named from."""
    config: InstanceConfig
    source: PreferredInstanceSource | None
    graph: InstanceGraph | None = None
    """The instance edition this dataset belongs to; None for a dataset read alone, as a mutation returns it."""
    unavailable: str | None = None
    """Why this edition's content cannot be read, if it cannot: its structure is still known."""
    owner: DatasetEditions | None = field(default=None, repr=False)
    """The editions this one was resolved with, which batch what its siblings also need."""
    _content: dict[str, Any] | None = field(default=None, repr=False)
    _snapshot: DatasetSnapshot | None = field(default=None, repr=False)
    _frames: dict[bool, EditionFrame] = field(default_factory=dict, repr=False)

    @property
    def id(self) -> UUID:
        return self.meta.id

    @classmethod
    def from_row(cls, row: Dataset) -> DatasetEdition:
        """Build the draft edition of one dataset from its row alone."""
        config = row.scope_instance
        meta = dataset_meta_from_model(row, primary_language=config.primary_language)
        return cls(
            meta=meta,
            row=row,
            is_live=True,
            revision=None,
            dimensions=dimension_catalog(meta, {}, config, row),
            config=config,
            source=PreferredInstanceSource.DRAFT,
        )

    @property
    def snapshot(self) -> DatasetSnapshot:
        if self.unavailable is not None:
            raise ValueError(self.unavailable)
        if self._snapshot is None:
            load_contents([self])
            if self._content is None:
                # Nothing records content for this dataset in this edition.
                self._snapshot = DatasetSnapshot(meta=self.meta)
            else:
                with set_i18n_context(self.config.primary_language, self.config.other_languages):
                    self._snapshot = DatasetSnapshot.model_validate(self._content)
        return self._snapshot

    def frame(self, qualifiers: QualifierCatalog | None = None) -> EditionFrame:
        """
        Return the edition's values, one column per metric and declared dimension.

        With ``qualifiers``, each graded cell also carries its assessment.
        """
        key = qualifiers is not None
        if key not in self._frames:
            snapshot = self.snapshot
            ppdf = snapshot.to_frame(self.dimensions)
            if qualifiers is not None:
                ppdf = attach_evidence_qualifiers(ppdf, snapshot.cell_grades(self.dimensions), qualifiers)
            frame = pl.DataFrame(ppdf)
            dimensions: dict[str, UUID] = {}
            categories: dict[tuple[str, str], UUID] = {}
            for dim_id in self.meta.declared_dimension_ids:
                dim = self.dimensions[dim_id]
                column = snapshot.dimension_columns.get(dim.id, dim.identifier)
                dimensions[column] = dim.id
                categories.update({(column, cat.identifier or str(cat.id)): cat.id for cat in dim.categories})
                if column not in frame.columns:
                    frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias(column))
                else:
                    frame = frame.with_columns(pl.col(column).cast(pl.String))
            self._frames[key] = EditionFrame(frame, dimensions, categories)
        return self._frames[key]


def load_contents(editions: Iterable[DatasetEdition]) -> None:
    """Load the content of each edition that has none yet, in as few queries as the sources allow."""
    pending = [
        edition for edition in editions if edition._content is None and edition._snapshot is None and edition.unavailable is None
    ]
    live = [edition for edition in pending if edition.is_live and edition.row is not None]
    if live:
        rows = {edition.row.pk: edition.row for edition in live if edition.row is not None}
        # A read must not write: a materialization older than its row is passed over, and
        # the row serialized instead, rather than refreshed here.
        fresh = {
            materialization.dataset_id: materialization.content
            for materialization, validation_hash in materializations_with_validation_hashes(
                DatasetMaterialization.objects.filter(dataset_id__in=rows)
            )
            if materialization_is_fresh(rows[materialization.dataset_id], materialization, validation_hash=validation_hash)
        }
        for edition in live:
            assert edition.row is not None
            content = fresh.get(edition.row.pk)
            edition._content = content if content is not None else serialize_dataset(edition.row)
    for edition in pending:
        if edition.revision is not None:
            edition._content = edition.revision.content


def evaluate_edition_rules(edition: DatasetEdition, data: EditionFrame | None = None) -> list[RuleViolation]:
    """
    Evaluate the metric rules and the closed category domain of one edition.

    The same violations `evaluate_dataset_rules` finds in the live rows, found in the
    edition's content instead. The rules are the ones the content was recorded with, and
    the domain is the one the instance resolves. ``data`` lets a caller evaluate a frame
    it has extended, as the data-entry views do with the years nobody has entered yet.
    """
    meta = edition.meta
    if meta.is_external_placeholder:
        return []
    closed = meta.category_domain.mode == 'closed'
    metric_by_name = {metric.identifier: metric for metric in meta.metrics}
    rules = [
        (metric_by_name[recorded.identifier], rule)
        for recorded in edition.snapshot.meta.metrics
        if recorded.identifier in metric_by_name
        for rule in recorded.validation_rules
        if rule.id is not None  # a rule row always has one; only YAML-declared rules lack it
    ]
    if not closed and not rules:
        return []
    data = data or edition.frame()
    frame = data.frame
    for metric in meta.metrics:
        column = metric.identifier or str(metric.id)
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
    violations: list[RuleViolation] = []
    for metric, rule in rules:
        assert rule.id is not None
        column = metric.identifier or str(metric.id)
        violations.extend(evaluate_rule(rule.rule, rule.id, metric.id, column, frame, data.dim_cols))
    if closed:
        allowed = list(data.domain(meta).values())
        domain_id = meta.shape_id or meta.schema_id
        for metric in meta.metrics:
            column = metric.identifier or str(metric.id)
            violations.extend(evaluate_closed_domain(domain_id, metric.id, column, frame, data.dim_cols, allowed))
    for violation in violations:
        violation.dataset_uuid = meta.id
        violation.dataset_identifier = meta.identifier
        violation.coordinates = data.dataset_coordinates(violation.categories)
    return violations


def edition_plausibility_ranges(edition: DatasetEdition) -> list[DatasetMetricPlausibilityRange]:
    """
    Return the curated ranges that apply to an edition, then those derived from its own history.

    Curated ranges are reference data and come from the database; the history is the edition's.
    """
    row = edition.row
    if row is None:
        return []
    curated = [rule for rule in applicable_plausibility_ranges(row) if rule.metric.uuid in edition.meta.metric_by_id]
    if not eligible(row):
        return curated
    try:
        data = edition.frame()
    except UndefinedUnitError as exc:
        _log_unreadable(row, exc)
        return curated
    return [*curated, *(derived.rule for derived in derive_history_ranges(row, data.frame, data.dim_cols, curated))]


def evaluate_edition_plausibility(
    edition: DatasetEdition,
    data: EditionFrame | None = None,
    violations: list[RuleViolation] | None = None,
) -> list[PlausibilityFinding]:
    """Advisory plausibility findings for the cells of one edition."""
    row = edition.row
    if row is None:
        return []
    ranges = [rule for rule in applicable_plausibility_ranges(row) if rule.metric.uuid in edition.meta.metric_by_id]
    if not ranges and not eligible(row):
        return []
    try:
        data = data or edition.frame()
    except UndefinedUnitError as exc:
        # A curated range is a configured check, and its failure should show; the history
        # fallback is advisory and must not turn an unreadable dataset into an editor error.
        if ranges:
            raise
        _log_unreadable(row, exc)
        return []
    cells = DatasetPlausibilityCells(
        dataset=row,
        frame=data.frame,
        dim_cols=data.dim_cols,
        coordinate_index=DatasetCoordinateIndex(row),
        violations=violations if violations is not None else evaluate_edition_rules(edition, data),
        combinations=data.domain(edition.meta),
        population=_observed_population(row, ranges),
    )
    return evaluate_plausibility_cells(cells, ranges)


class DatasetEditions:
    """
    The datasets of one instance source, as one user may read them.

    Request-scoped, like the graph it is built on: the rows, pins and contents are
    loaded once and shared by every field that reads a dataset of this source.
    """

    def __init__(
        self,
        graph: InstanceGraph,
        user: UserOrAnon,
        *,
        published: bool = False,
        config: InstanceConfig | None = None,
        source: PreferredInstanceSource | None = None,
    ):
        self.graph = graph
        self.user = user
        self.published = published
        self.source = source
        if config is not None:
            self.config = config
        self._editions: dict[UUID, DatasetEdition | None] = {}
        self._domains = CategoryDomainResolver()
        self._schemas_prefetched = False

    @cached_property
    def config(self) -> InstanceConfig:
        return InstanceConfig.objects.get(uuid=self.graph.instance_id)

    @cached_property
    def rows(self) -> dict[UUID, Dataset]:
        """Load every row of the graph's datasets, readable or not."""
        rows = (
            Dataset.objects
            .with_schema_editability(Framework)
            .filter(uuid__in=self.graph.dataset_by_id)
            # Structure comes from the graph's catalog, so the schema's metrics and dimensions are not needed.
            .select_related('schema', 'created_by', 'last_modified_by')
        )
        return {row.uuid: row for row in rows}

    @cached_property
    def viewable(self) -> dict[UUID, Dataset]:
        """Select the rows of the graph's datasets the user may read."""
        ids = set(Dataset.objects.filter(uuid__in=self.rows).viewable_by(self.user).values_list('uuid', flat=True))
        return {id: row for id, row in self.rows.items() if id in ids}

    @cached_property
    def _revisions(self) -> dict[int, Revision[Dataset]]:
        pins = self.graph.pinned_revisions
        # Each is checked to be a dataset's revision before it is read, in `_pinned_revision`.
        return {
            revision.pk: cast('Revision[Dataset]', revision)
            for revision in Revision.objects.filter(pk__in=pins.values()).select_related('content_type')
        }

    def _pinned_revision(self, id: UUID, row: Dataset | None) -> tuple[Revision[Dataset] | None, str | None]:
        """Return the revision a dataset is pinned to, or why the pin cannot be read."""
        revision_id = self.graph.pinned_revisions.get(id)
        if revision_id is None:
            return None, f'Published dataset {id} has no revision pin' if self.published else None
        revision = self._revisions.get(revision_id)
        if (
            revision is None
            or (row is not None and revision.object_id != str(row.pk))
            or revision.content_type.model_class() is not Dataset
        ):
            return None, f'Dataset revision pin is unavailable for {id}'
        return revision, None

    @cached_property
    def _instance_dimensions(self) -> dict[UUID, DimensionMeta]:
        return {dimension.id: dimension for dimension in _dimension_catalog_for(self.config)}

    def _dimensions(self, meta: DatasetMeta, row: Dataset | None) -> Mapping[UUID, DimensionMeta]:
        catalog = self.graph.dimension_by_id
        if all(id in catalog for id in meta.declared_dimension_ids):
            return catalog
        return dimension_catalog(meta, catalog, self.config, row, self._instance_dimensions)

    def prefetch_schemas(self) -> None:
        """Load the schema structure of every row of this source at once, for the deprecated `Dataset.schema`."""
        if self._schemas_prefetched:
            return
        self._schemas_prefetched = True
        resolved = [edition.row for edition in self._editions.values() if edition is not None and edition.row is not None]
        rows = {row.pk: row for row in [*self.rows.values(), *resolved]}.values()
        prefetch_related_objects(
            [row.schema for row in rows if row.schema is not None],
            'metrics__validation_rules',
            'dimensions__dimension__categories',
        )

    def get(self, id: UUID) -> DatasetEdition | None:
        """
        Return the edition of a dataset of the graph; None when it is not the graph's.

        A dataset reached through the graph -- a binding, a data-entry table -- is read
        with the instance source, which the caller has authorized: a municipality's
        editor reads the template datasets its bindings name, though the dataset policy
        would not let it list them. Listings filter by `viewable` instead.
        """
        if id not in self._editions:
            meta = self.graph.dataset_by_id.get(id)
            row = self.rows.get(id)
            if meta is None:
                self._editions[id] = None
            else:
                revision, unavailable = self._pinned_revision(id, row)
                self._editions[id] = DatasetEdition(
                    meta=meta,
                    row=row,
                    is_live=revision is None and unavailable is None and row is not None,
                    revision=revision,
                    unavailable=unavailable,
                    dimensions=self._dimensions(meta, row),
                    config=self.config,
                    source=self.source,
                    graph=self.graph,
                    owner=self,
                )
        return self._editions[id]

    def require(self, id: UUID) -> DatasetEdition:
        edition = self.get(id)
        if edition is None:
            raise ValueError(f'Dataset {id} is not part of instance {self.graph.instance_id}')
        return edition

    def for_row(self, row: Dataset) -> DatasetEdition | None:
        """
        Return the edition of a dataset found by its row, which the caller has checked the user may read.

        The draft also holds datasets its graph does not catalog yet -- one nothing binds --
        and reads them from the row. Any other source holds only its graph's datasets.
        """
        if row.uuid in self.graph.dataset_by_id:
            return self.get(row.uuid)
        if self.published or row.schema is None:
            return None
        if row.uuid not in self._editions:
            meta = dataset_meta_from_model(row, primary_language=self.config.primary_language, domains=self._domains)
            self._editions[row.uuid] = DatasetEdition(
                meta=meta,
                row=row,
                is_live=True,
                revision=None,
                dimensions=self._dimensions(meta, row),
                config=self.config,
                source=self.source,
                graph=self.graph,
                owner=self,
            )
        return self._editions[row.uuid]

    def for_rows(self, rows: Iterable[Dataset]) -> list[DatasetEdition]:
        return [edition for row in rows if (edition := self.for_row(row)) is not None]

    def all(self) -> list[DatasetEdition]:
        """Return every dataset of the graph the user may read, in catalog order."""
        return [
            edition
            for meta in self.graph.datasets
            if (meta.id in self.viewable or meta.id not in self.rows) and (edition := self.get(meta.id)) is not None
        ]

    def load(self, ids: Iterable[UUID]) -> list[DatasetEdition]:
        """Return the editions of ``ids``, with their contents loaded together."""
        editions = [edition for id in ids if (edition := self.get(id)) is not None]
        load_contents(editions)
        return editions
