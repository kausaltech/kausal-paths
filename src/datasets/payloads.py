"""Lazy bulk loaders for current and revisioned dataset payloads."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping
    from uuid import UUID

    from common import polars as ppl
    from datasets.snapshot import CellLabels, DatasetSnapshot, QualityLevelRef
    from nodes.defs.graph import DimensionMeta


@dataclass(frozen=True)
class DatasetPayloadRef:
    """Lightweight pointer to a serialized dataset payload."""

    payload_id: int
    dataset_pk: int
    dataset_uuid: str
    identifier: str
    content_hash: str
    generation: int | None
    forecast_from: int | None


class DatasetPayloadStore(ABC):
    """
    Lazy bulk loader shared by current and revision-backed datasets.

    ``dimensions`` is the catalog of the graph reading the payloads, which names
    their frames' columns and category labels.
    """

    def __init__(self, refs: list[DatasetPayloadRef], dimensions: Mapping[UUID, DimensionMeta]) -> None:
        self.refs = refs
        self.dimensions = dimensions
        self._contents: dict[tuple[bool, int], dict[str, Any]] | None = None
        self._snapshots: dict[tuple[bool, int], DatasetSnapshot] = {}
        self._dataframes: dict[tuple[bool, int], ppl.PathsDataFrame] = {}

    @abstractmethod
    def _load_contents(self) -> dict[tuple[bool, int], dict[str, Any]]:
        raise NotImplementedError

    @staticmethod
    def _key(ref: DatasetPayloadRef) -> tuple[bool, int]:
        return (ref.generation is None, ref.payload_id)

    def get_content(self, ref: DatasetPayloadRef) -> dict[str, Any]:
        if self._contents is None:
            self._contents = self._load_contents()
        try:
            return self._contents[self._key(ref)]
        except KeyError as exc:
            raise RuntimeError(f'Missing serialized payload {ref.payload_id} for dataset {ref.identifier}') from exc

    def get_snapshot(self, ref: DatasetPayloadRef) -> DatasetSnapshot:
        from datasets.snapshot import DatasetSnapshot

        snapshot = self._snapshots.get(self._key(ref))
        if snapshot is None:
            snapshot = DatasetSnapshot.model_validate(self.get_content(ref))
            self._snapshots[self._key(ref)] = snapshot
        return snapshot

    def get_dataframe(self, ref: DatasetPayloadRef) -> ppl.PathsDataFrame:
        cached = self._dataframes.get(self._key(ref))
        if cached is not None:
            return cached
        snapshot = self.get_snapshot(ref)
        if not snapshot.meta.metrics:
            raise RuntimeError(f'Dataset {ref.identifier} has no metrics, so it has no dataframe')
        df = snapshot.to_frame(self.dimensions)
        self._dataframes[self._key(ref)] = df
        return df

    def get_cell_grades(self, ref: DatasetPayloadRef) -> dict[CellLabels, QualityLevelRef]:
        return self.get_snapshot(ref).cell_grades(self.dimensions)


class CurrentDatasetPayloadStore(DatasetPayloadStore):
    def _load_contents(self) -> dict[tuple[bool, int], dict[str, Any]]:
        from nodes.models import DatasetMaterialization

        payload_ids = {ref.payload_id for ref in self.refs}
        rows = DatasetMaterialization.objects.filter(pk__in=payload_ids).values_list('pk', 'content')
        contents = dict(rows)
        missing = payload_ids - contents.keys()
        if missing:
            raise RuntimeError(f'Missing current dataset materializations: {sorted(missing)}')
        return {(False, pk): content for pk, content in contents.items()}


class RevisionDatasetPayloadStore(DatasetPayloadStore):
    def _load_contents(self) -> dict[tuple[bool, int], dict[str, Any]]:
        from wagtail.models import Revision

        payload_ids = {ref.payload_id for ref in self.refs}
        rows = Revision.objects.filter(pk__in=payload_ids).values_list('pk', 'content')
        contents = dict(rows)
        missing = payload_ids - contents.keys()
        if missing:
            raise RuntimeError(f'Missing published dataset revisions: {sorted(missing)}')
        return {(True, pk): content for pk, content in contents.items()}


class MixedDatasetPayloadStore(DatasetPayloadStore):
    """Draft local data together with immutable framework reference data."""

    def _load_contents(self) -> dict[tuple[bool, int], dict[str, Any]]:
        current = CurrentDatasetPayloadStore([ref for ref in self.refs if ref.generation is not None], self.dimensions)
        revisions = RevisionDatasetPayloadStore([ref for ref in self.refs if ref.generation is None], self.dimensions)
        return current._load_contents() | revisions._load_contents()
