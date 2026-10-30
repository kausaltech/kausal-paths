"""
The shapes of DVC datasets, so that a graph can know them before it loads them.

A DVC dataset's dimensions and metrics are what the solver needs to infer anything downstream
of it, and the parquet is the only complete record of them. The `.dvc` file records units and,
for datasets pushed since dvc-pandas started writing them, index columns. Everything else is
learnt the first time the runtime loads the dataset and recorded under the parquet's content
hash, so that a graph built later reads it without fetching anything.
"""

from typing import TYPE_CHECKING

from django.db import DatabaseError
from pydantic import BaseModel, ConfigDict, Field

import polars as pl
from loguru import logger

from common import polars as ppl
from datasets.manifests import persisted_manifest
from datasets.models import DVCSourceShape
from datasets.runtime.dvc import DVCDataset
from nodes.constants import FORECAST_COLUMN, YEAR_COLUMN

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import dvc_pandas

    from nodes.defs.instance_defs import DatasetRepoSpec


DVC_SOURCE_SHAPE_VERSION = 3
"""2: records which dimension columns each metric has values in. 3: and which it fills only in part."""

_recorded_hashes: set[str] = set()
"""Content hashes this process has already recorded; the table is keyed by them, so once is enough."""


class SourceShape(BaseModel):
    """What a DVC dataset holds before any binding reshapes it."""

    model_config = ConfigDict(frozen=True)

    dimension_columns: tuple[str, ...] | None
    """
    The columns the source is indexed by, other than `Year`.

    `None` when the source has no index but carries other columns: which of those become
    dimensions is then decided by the binding's operations, not by the source.
    """
    metric_units: dict[str, str]
    metric_dimension_columns: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    """
    For each metric, the dimension columns it has values in.

    A dataset with several metrics marks a dimension that does not apply to one of them with
    nulls, and the runtime drops a column that is null throughout what a binding selected, so
    a metric's shape can be narrower than the source's index.
    """
    metric_partial_columns: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    """
    For each metric, those of its dimension columns that are null in some of its rows.

    A binding that filters such a metric may leave one of them null throughout, and the
    runtime then drops it, so the metric's shape depends on the categories selected.
    """


def shape_of_dvc_dataset(dataset: dvc_pandas.Dataset) -> SourceShape:
    """Read a loaded dataset's shape as the runtime sees it, provenance columns dropped."""
    frame = DVCDataset.drop_reserved_columns(ppl.from_dvc_dataset(dataset))
    metric_units = {column: unit for column, unit in (dataset.units or {}).items() if column in frame.metric_cols}
    if frame.primary_keys:
        dimensions: tuple[str, ...] | None = tuple(key for key in frame.primary_keys if key != YEAR_COLUMN)
    else:
        payload = {*metric_units, *frame.qualifier_cols.values(), YEAR_COLUMN, FORECAST_COLUMN}
        dimensions = () if all(column in payload for column in frame.columns) else None
    present, partial = _metric_dimension_columns(frame, list(metric_units), dimensions or ())
    return SourceShape(
        dimension_columns=dimensions,
        metric_units=metric_units,
        metric_dimension_columns=present,
        metric_partial_columns={metric: columns for metric, columns in partial.items() if columns},
    )


type _ColumnsByMetric = dict[str, tuple[str, ...]]


def _metric_dimension_columns(
    frame: pl.DataFrame, metrics: list[str], dimensions: Sequence[str]
) -> tuple[_ColumnsByMetric, _ColumnsByMetric]:
    """Return each metric's dimension columns with a value, and those of them that also have nulls."""
    if not metrics or not dimensions:
        return {metric: tuple(dimensions) for metric in metrics}, {}
    exprs: list[pl.Expr] = []
    for index, metric in enumerate(metrics):
        for dimension in dimensions:
            selected = pl.col(dimension).filter(pl.col(metric).is_not_null())
            exprs.append(selected.is_not_null().any().alias(f'{index}:{dimension}:value'))
            exprs.append(selected.is_null().any().alias(f'{index}:{dimension}:null'))
    found = frame.select(exprs).row(0, named=True)
    present = {
        metric: tuple(dimension for dimension in dimensions if found[f'{index}:{dimension}:value'])
        for index, metric in enumerate(metrics)
    }
    partial = {
        metric: tuple(dimension for dimension in present[metric] if found[f'{index}:{dimension}:null'])
        for index, metric in enumerate(metrics)
    }
    return present, partial


def shape_from_manifest(manifest: dvc_pandas.DatasetManifest) -> SourceShape | None:
    """
    Read the shape the `.dvc` file records, when it records enough to know one.

    An empty index says only that the source is not indexed; its dimensions may still come
    from columns the `.dvc` file does not list, so it is not a shape. Nor is the index of a
    source with several metrics, since the file cannot say which dimensions each one has.
    """
    names = _index_column_names(manifest.index_columns)
    metric_units = {column: unit for column, unit in (manifest.units or {}).items() if column not in names}
    if not names or len(metric_units) != 1:
        return None
    dimensions = tuple(name for name in names if name != YEAR_COLUMN)
    return SourceShape(
        dimension_columns=dimensions,
        metric_units=metric_units,
        metric_dimension_columns=dict.fromkeys(metric_units, dimensions),
    )


def _index_column_names(index_columns: Sequence[object] | None) -> list[str]:
    # Older `.dvc` files may carry Arrow's descriptor for an unnamed RangeIndex.
    names: list[str] = []
    for column in index_columns or ():
        if isinstance(column, str):
            names.append(column)
        elif isinstance(column, dict) and isinstance(name := column.get('name'), str):
            names.append(name)
    return names


def dvc_source_shape_is_recorded(content_hash: str) -> bool:
    """Whether the current shape version is recorded for this content hash; asked of the database once per process."""
    if content_hash in _recorded_hashes:
        return True
    try:
        recorded = DVCSourceShape.objects.filter(content_hash=content_hash, format_version=DVC_SOURCE_SHAPE_VERSION).exists()
    except DatabaseError as error:
        logger.warning('Unable to look up the shape of DVC content {}: {}', content_hash, error)
        return True
    if recorded:
        _recorded_hashes.add(content_hash)
    return recorded


def record_dvc_source_shape(dataset: dvc_pandas.Dataset) -> None:
    """
    Record a loaded dataset's shape under its content hash, once per process.

    The table is a rebuildable cache, and a computation must not fail because a cache could
    not be written, so a database error is logged rather than raised.
    """
    content_hash = dataset.hash
    if not isinstance(content_hash, str) or content_hash in _recorded_hashes:
        return
    shape = shape_of_dvc_dataset(dataset)
    try:
        DVCSourceShape.objects.bulk_create(
            [
                DVCSourceShape(
                    content_hash=content_hash,
                    format_version=DVC_SOURCE_SHAPE_VERSION,
                    shape=shape.model_dump(mode='json'),
                )
            ],
            ignore_conflicts=True,
        )
    except DatabaseError as error:
        logger.warning('Unable to record the shape of DVC dataset {}: {}', dataset.identifier, error)
        return
    _recorded_hashes.add(content_hash)


def known_dvc_source_shapes(spec: DatasetRepoSpec | None, identifiers: Iterable[str]) -> dict[str, SourceShape]:
    """
    Look up the shapes known for these datasets at the pinned revision, without reading git or DVC.

    Uses only what earlier loads persisted: the revision's manifests give each dataset's
    content hash, and the hash gives a recorded shape, or failing that the shape its `.dvc`
    file records. A dataset no load has seen yet is left out.
    """
    if spec is None:
        return {}
    manifest = persisted_manifest(spec, set(identifiers), None, resolve_missing=False)
    if manifest is None or not manifest.datasets:
        return {}
    hashes = {identifier: source.content_hash for identifier, source in manifest.datasets.items()}
    recorded = {
        row.content_hash: SourceShape.model_validate(row.shape)
        for row in DVCSourceShape.objects.filter(content_hash__in=set(hashes.values()), format_version=DVC_SOURCE_SHAPE_VERSION)
    }
    shapes: dict[str, SourceShape] = {}
    for identifier, source in manifest.datasets.items():
        shape = recorded.get(hashes[identifier]) or shape_from_manifest(source)
        if shape is not None:
            shapes[identifier] = shape
    return shapes
