"""Lazy bulk loaders for current and revisioned dataset payloads."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from common import polars as ppl


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
    """Lazy bulk loader shared by current and revision-backed datasets."""

    def __init__(self, refs: list[DatasetPayloadRef]) -> None:
        self.refs = refs
        self._contents: dict[tuple[bool, int], dict[str, Any]] | None = None
        self._dataframes: dict[tuple[bool, int], ppl.PathsDataFrame] = {}

    @abstractmethod
    def _load_contents(self) -> dict[tuple[bool, int], dict[str, Any]]:
        raise NotImplementedError

    def get_content(self, ref: DatasetPayloadRef) -> dict[str, Any]:
        if self._contents is None:
            self._contents = self._load_contents()
        try:
            return self._contents[(ref.generation is None, ref.payload_id)]
        except KeyError as exc:
            raise RuntimeError(f'Missing serialized payload {ref.payload_id} for dataset {ref.identifier}') from exc

    def get_dataframe(self, ref: DatasetPayloadRef) -> ppl.PathsDataFrame:
        cached = self._dataframes.get((ref.generation is None, ref.payload_id))
        if cached is not None:
            return cached
        content = self.get_content(ref)
        data = content.get('data')
        if data is None:
            raise RuntimeError(f'Dataset {ref.identifier} has no serialized dataframe payload')
        from datasets.runtime import JSONDataset

        df = JSONDataset.deserialize_df(data)
        self._dataframes[(ref.generation is None, ref.payload_id)] = df
        return df


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
        current = CurrentDatasetPayloadStore([ref for ref in self.refs if ref.generation is not None])
        revisions = RevisionDatasetPayloadStore([ref for ref in self.refs if ref.generation is None])
        return current._load_contents() | revisions._load_contents()
