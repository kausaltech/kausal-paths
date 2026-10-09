from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import ConfigDict, Field, PrivateAttr

from kausal_common.datasets.category_domain import DatasetCategoryCombination, DatasetCategoryDomain
from kausal_common.i18n.pydantic import I18nBaseModel, I18nString

from paths.identifiers import DatasetId, DatasetMetricId, DatasetSchemaId, DimensionCategoryId, DimensionId, ValidationRuleId
from paths.refs import DatasetMetricRef, DimensionRef, ShapeRef
from paths.uuid_kinds import DictOf, Ref, UuidLeaf, register_uuid_kinds

from datasets.validation_rules import ValidationRule

if TYPE_CHECKING:
    from collections.abc import Mapping
    from uuid import UUID

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


# A combination of a shape-derived domain carries the shape combination's uuid; one of a
# schema's stored domain has an id nothing else refers to, which a copy keeps.
register_uuid_kinds(
    DatasetCategoryCombination,
    id=UuidLeaf(Ref('shape_combination')),
    categories=DictOf(UuidLeaf(Ref('dimension')), UuidLeaf(Ref('category'))),
)


class FrozenGraphModel(I18nBaseModel):
    """Serializable graph catalog value with no graph back-reference."""

    model_config = ConfigDict(frozen=True)


class DatasetExternalRef(FrozenGraphModel):
    """Where a dataset's data comes from outside the database: a path in a DVC repository."""

    repo_url: str
    commit: str | None = None
    """The commit the data was read from; None when the repository was not pinned."""
    dataset_id: str
    """The dataset's path in the repository, without extension."""


class DimensionCategoryMeta(FrozenGraphModel):
    id: DimensionCategoryId
    identifier: str | None = None
    label: I18nString | None = None
    # Left out when absent, so revisions frozen before short labels keep their content hash.
    short_label: I18nString | None = Field(default=None, exclude_if=lambda value: value is None)
    order: int | None = None
    spec: dict[str, Any] = Field(default_factory=dict)


type CatalogScope = Literal['instance', 'framework']
"""Who owns a catalog entry: the instance itself, or the framework the instance belongs to."""


class DimensionMeta(FrozenGraphModel):
    id: DimensionId
    identifier: str
    label: I18nString | None = None
    order: int | None = None
    spec: dict[str, Any] = Field(default_factory=dict)
    categories: tuple[DimensionCategoryMeta, ...] = ()
    # Left out when the instance owns it, so content hashes from before scopes were recorded hold.
    scope: CatalogScope = Field(default='instance', exclude_if=lambda value: value == 'instance')
    """A framework's dimension is shared by every instance of it; a copy refers to it rather than copying it."""


class ValidationRuleMeta(FrozenGraphModel):
    """A validation rule on a metric, with the identity of its row where it has one."""

    id: ValidationRuleId | None = None
    """The rule row's uuid; None for a rule declared in YAML that has no row yet."""
    rule: ValidationRule


class DatasetMetricMeta(FrozenGraphModel):
    id: DatasetMetricId
    identifier: str | None = None
    label: I18nString | None = None
    unit: str = ''
    quantity: str | None = None
    """Quantity-kind id of what the metric measures; None means any quantity."""
    order: int | None = None
    validation_rules: tuple[ValidationRuleMeta, ...] = ()
    quality_of: DatasetMetricRef | None = None
    """The metric whose grades this metric holds, as scores; see `frameworks.evidence.QUALITY_OF_SPEC_KEY`."""


class QualityLevelKey(FrozenGraphModel):
    """A framework quality level named by authored identity, so a module can declare it before any database exists."""

    scheme: str
    level: str


class DatasetMeta(FrozenGraphModel):
    """
    A dataset's structure: what it is, not what it holds.

    The graph catalog entry, and the structural half of a `DatasetSnapshot`. Which
    revision of the dataset a graph reads is binding state, not structure, and lives
    in the instance's `dataset_revisions` pins.
    """

    id: DatasetId
    identifier: str | None = None
    name: I18nString | None = None
    schema_id: DatasetSchemaId
    schema_scope: CatalogScope = Field(default='instance', exclude_if=lambda value: value == 'instance')
    """A framework's schema may be shared by datasets in several instances; a copy refers to it."""
    is_editable: bool | None = None
    metrics: tuple[DatasetMetricMeta, ...] = ()
    declared_dimension_ids: tuple[DimensionRef, ...] = ()
    time_resolution: str = 'yearly'
    forecast_from: int | None = None
    """The dataset's own first forecast year, which bindings inherit unless they set one."""
    is_external_placeholder: bool = False
    external_ref: DatasetExternalRef | None = None
    category_domain: DatasetCategoryDomain = Field(default_factory=DatasetCategoryDomain)
    # Left out when absent, so revisions frozen before shapes keep their content hash.
    shape_id: ShapeRef | None = Field(default=None, exclude_if=lambda value: value is None)
    """The shape this dataset's entry form follows, resolved in the dataset's own instance."""
    default_quality: QualityLevelKey | None = None
    """
    The grade of every value in the dataset that has no grade of its own.

    A grade that belongs to the kind of data rather than to one figure: BISKO grades the
    gemeindefeine ifeu defaults B as a class. A value's own evidence overrides it.
    """

    @cached_property
    def metric_by_id(self) -> dict[UUID, DatasetMetricMeta]:
        return {metric.id: metric for metric in self.metrics}
