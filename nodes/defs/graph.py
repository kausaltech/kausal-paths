from functools import cached_property
from typing import TYPE_CHECKING, Any, Self
from uuid import UUID

from pydantic import ConfigDict, Field, PrivateAttr

from kausal_common.i18n.pydantic import I18nBaseModel, I18nString

from datasets.validation_rules import ValidationRule

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nodes.instance_graph import InstanceGraph


class InstanceGraphBoundModel(I18nBaseModel):
    """Immutable public data that gains graph navigation after hydration."""

    model_config = ConfigDict(frozen=True)

    _graph: Any = PrivateAttr(default=None)

    @property
    def graph(self) -> InstanceGraph:
        if self._graph is None:
            raise RuntimeError(f'{type(self).__name__} is not bound to an InstanceGraph')
        return self._graph

    def _bind_graph(self, graph: InstanceGraph) -> None:
        if self._graph is not None and self._graph is not graph:
            raise RuntimeError(f'{type(self).__name__} is already bound to another graph')
        self._graph = graph

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        if self._graph is not None:
            raise RuntimeError(f'{type(self).__name__} is graph-bound and cannot be copied')
        return super().model_copy(update=update, deep=deep)


class FrozenGraphModel(I18nBaseModel):
    """Serializable graph catalog value with no graph back-reference."""

    model_config = ConfigDict(frozen=True)


class DimensionCategoryMeta(FrozenGraphModel):
    id: UUID
    identifier: str | None = None
    label: I18nString | None = None
    order: int | None = None
    spec: dict[str, Any] = Field(default_factory=dict)


class DimensionMeta(FrozenGraphModel):
    id: UUID
    identifier: str
    label: I18nString | None = None
    order: int | None = None
    spec: dict[str, Any] = Field(default_factory=dict)
    categories: tuple[DimensionCategoryMeta, ...] = ()


class DatasetMetricMeta(FrozenGraphModel):
    id: UUID
    identifier: str | None = None
    label: I18nString | None = None
    unit: str = ''
    order: int | None = None
    validation_rules: tuple[ValidationRule, ...] = ()


class DatasetMeta(FrozenGraphModel):
    id: UUID
    identifier: str | None = None
    schema_id: UUID
    is_editable: bool | None = None
    metrics: tuple[DatasetMetricMeta, ...] = ()
    declared_dimension_ids: tuple[UUID, ...] = ()
    is_external_placeholder: bool = False
    external_ref: dict[str, Any] | None = None
    revision_id: int | None = None

    @cached_property
    def metric_by_id(self) -> dict[UUID, DatasetMetricMeta]:
        return {metric.id: metric for metric in self.metrics}
