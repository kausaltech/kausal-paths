"""Durable, content-addressed cache for the deterministic prefix of a DVC binding."""

import hashlib
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, TypedDict

import orjson
import polars as pl
from loguru import logger

from common import polars as ppl
from datasets.models import PreparedDataset
from nodes.units import unit_registry

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pydantic import JsonValue

# Bump for changes to source conversion, unit definitions, or IPC interpretation.
# Individual transformations also contribute their own implementation versions.
PREPARATION_VERSION = 1


class FrameMetadata(TypedDict):
    units: dict[str, str]
    primary_keys: list[str]
    explanations: list[str]


@dataclass(frozen=True)
class PreparedRecipe:
    content: dict[str, JsonValue]

    @property
    def key(self) -> str:
        return hashlib.sha256(orjson.dumps(self.content, option=orjson.OPT_SORT_KEYS)).hexdigest()


def serialize_frame(frame: ppl.PathsDataFrame) -> tuple[bytes, FrameMetadata]:
    buffer = BytesIO()
    frame.write_ipc(buffer, compression='zstd')
    meta = frame.get_meta()
    return buffer.getvalue(), FrameMetadata(
        units={column: str(unit) for column, unit in meta.units.items()},
        primary_keys=meta.primary_keys,
        explanations=frame._explanation.copy(),
    )


def deserialize_frame(payload: bytes, metadata: FrameMetadata) -> ppl.PathsDataFrame:
    frame = pl.read_ipc(BytesIO(payload), memory_map=False)
    meta = ppl.DataFrameMeta(
        units={column: unit_registry.parse_units(unit) for column, unit in metadata['units'].items()},
        primary_keys=list(metadata['primary_keys']),
    )
    result = ppl.to_ppdf(frame, meta=meta)
    result._explanation = list(metadata['explanations'])
    return result


class PreparedDatasetStore:
    """Batch reads for a context; immutable bytes prevent mutation across bindings."""

    def __init__(self) -> None:
        self.prefetched = False
        self.rows: dict[str, PreparedDataset | None] = {}

    def prefetch(self, recipes: Iterable[PreparedRecipe]) -> None:
        keys = {recipe.key for recipe in recipes} - self.rows.keys()
        if not keys:
            return
        self.rows.update(dict.fromkeys(keys))
        self.rows.update(PreparedDataset.objects.filter(key__in=keys).in_bulk())

    def get(self, recipe: PreparedRecipe) -> ppl.PathsDataFrame | None:
        self.prefetch([recipe])
        row = self.rows[recipe.key]
        if row is None:
            return None
        try:
            return deserialize_frame(bytes(row.payload), row.frame_metadata)
        except ValueError, TypeError, KeyError, OSError, pl.exceptions.PolarsError:
            logger.warning('Ignoring invalid prepared dataset {}', recipe.key)
            self.rows[recipe.key] = None
            return None

    def put(self, recipe: PreparedRecipe, frame: ppl.PathsDataFrame) -> None:
        payload, metadata = serialize_frame(frame)
        row = PreparedDataset(key=recipe.key, recipe=recipe.content, payload=payload, frame_metadata=metadata)
        # Concurrent workers may compute the same immutable prefix. An upsert also
        # repairs a corrupt payload without deleting a row another worker has repaired.
        PreparedDataset.objects.bulk_create(
            [row],
            update_conflicts=True,
            unique_fields=['key'],
            update_fields=['payload', 'frame_metadata', 'recipe'],
        )
        self.rows[recipe.key] = row
