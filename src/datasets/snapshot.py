"""Dataset revision and portable export snapshot shapes."""

from typing import TYPE_CHECKING, Any, Self, cast
from uuid import UUID  # noqa: TC003 - Pydantic evaluates snapshot fields

from pydantic import BaseModel, Field

from kausal_common.datasets.category_domain import DatasetCategoryDomain
from kausal_common.i18n.pydantic import TranslatedString  # noqa: TC002 - Pydantic field

from datasets.validation_rules import ValidationRule  # noqa: TC001 - Pydantic field
from nodes.snapshot_base import ModelSnapshot

if TYPE_CHECKING:
    from kausal_common.datasets.models import DatasetMetric

    from nodes.models import InstanceConfig


class MetricValidationRuleSnapshot(ModelSnapshot):
    """
    One validation rule bound to a metric.

    ``rule`` parses the stored blob strictly against the schema in
    ``datasets.validation_rules``; ``uuid`` records the source row's identity.
    """

    uuid: UUID
    rule: ValidationRule

    @classmethod
    def from_model(cls, obj: Any) -> Self:
        from datasets.transfer import metric_validation_rule_snapshot_from_model

        return cast('Self', metric_validation_rule_snapshot_from_model(obj))


def metric_column_id(metric: DatasetMetric) -> str:
    """
    Resolve the dataframe column a metric maps to.

    ``DatasetMetric.name`` *is* the column name and ``label`` is display text, so the
    two agree for imported metrics: ``load_dvc_dataset`` and ``dataset_placeholders``
    both create them as ``name=label=<column>``. A metric authored in the editor has
    no ``name`` at all, and its column is built from the label — see the
    ``Coalesce(name, label, uuid)`` in ``DBDataset.deserialize_df``
    (``nodes/datasets.py``), which is the writer this function has to agree with.

    **Falling through to the uuid is never a working answer.** No dataframe column is
    ever named after a metric uuid, so a selector built from one cannot match: it
    fails in ``select_metric`` as "Column '<uuid>' not found". It is kept only so a
    metric carrying neither name nor label still yields a stable string instead of
    ``None``.
    """
    return metric.name or metric.label or str(metric.uuid)


class DatasetMetricSnapshot(ModelSnapshot):
    identifier: str
    label: TranslatedString | None = None
    unit: str
    quantity: str | None = None
    validation_rules: list[MetricValidationRuleSnapshot] = Field(default_factory=list)

    @classmethod
    def from_model(cls, obj: Any) -> Self:
        from datasets.transfer import dataset_metric_snapshot_from_model

        return cast('Self', dataset_metric_snapshot_from_model(obj))

    @classmethod
    def from_model_with_language(cls, obj: Any, primary_language: str) -> Self:
        from datasets.transfer import dataset_metric_snapshot_from_model_with_language

        return cast('Self', dataset_metric_snapshot_from_model_with_language(obj, primary_language))

    @staticmethod
    def _rules_from_model(obj: Any) -> list[MetricValidationRuleSnapshot]:
        from datasets.transfer import metric_rules_from_model

        return metric_rules_from_model(obj)


class DataPointKey(BaseModel):
    """Natural key locating a DataPoint within its dataset (id-free, restore-stable)."""

    year: int
    metric: str  # metric identifier (name or uuid)
    categories: list[str] = Field(default_factory=list)  # sorted dimension-category ids


class DataSourceSnapshot(BaseModel):
    """A published data source referenced by a dataset or its data points."""

    uuid: str  # source DataSource uuid; the join key for references within the snapshot
    name: str
    edition: str | None = None
    authority: str | None = None
    description: str | None = None
    url: str | None = None


class SourceReferenceSnapshot(BaseModel):
    """Links a data source to the dataset (``point`` is None) or to one data point."""

    data_source: str  # DataSourceSnapshot.uuid
    point: DataPointKey | None = None


class DataPointCommentSnapshot(BaseModel):
    """A (non-soft-deleted) comment on a data point. Users are referenced by uuid."""

    point: DataPointKey
    text: str
    is_sticky: bool = False
    is_review: bool = False
    review_state: str | None = None
    resolved_at: str | None = None  # ISO 8601
    created_by: str | None = None  # user uuid
    last_modified_by: str | None = None  # user uuid
    resolved_by: str | None = None  # user uuid


class QualityLevelRef(BaseModel):
    """
    A framework quality grade, by UUID and by authored identity.

    The UUID is exact within one deployment; the identifiers let a snapshot
    resolve against a framework provisioned elsewhere.
    """

    uuid: str
    scheme: str
    scheme_version: str
    level: str


class DataPointEvidenceSnapshot(BaseModel):
    """What is asserted about one data point's value. Users are referenced by uuid."""

    point: DataPointKey
    kind: str | None = None
    quality_level: QualityLevelRef | None = None
    created_by: str | None = None  # user uuid
    last_modified_by: str | None = None  # user uuid


class DatasetSnapshot(ModelSnapshot):
    """
    Pydantic representation of a ``Dataset`` ORM row.

    Includes its DataPoints. Used both as the Wagtail revision payload for Dataset
    (via ``Dataset.serializable_data`` bridged in Paths) and as the
    dataset-body carrier inside ``InstanceExport``.
    """

    schema_version: int = 1
    identifier: str | None = None
    name: TranslatedString | None = None
    forecast_from: int | None = None
    is_external_placeholder: bool = False
    external_ref: dict[str, Any] | None = None
    time_resolution: str = 'yearly'
    is_editable: bool = True
    dimensions: list[str] = Field(default_factory=list)
    dimension_columns: dict[str, str] = Field(default_factory=dict)
    metrics: list[DatasetMetricSnapshot] = Field(default_factory=list)
    category_domain: DatasetCategoryDomain = Field(default_factory=DatasetCategoryDomain)
    data: dict[str, Any] | None = None
    data_sources: list[DataSourceSnapshot] = Field(default_factory=list)
    source_references: list[SourceReferenceSnapshot] = Field(default_factory=list)
    comments: list[DataPointCommentSnapshot] = Field(default_factory=list)
    evidence: list[DataPointEvidenceSnapshot] = Field(default_factory=list)

    @classmethod
    def from_model(cls, obj: Any) -> Self:
        return cls.from_model_for_instance(obj, None)

    @classmethod
    def from_model_for_instance(cls, obj: Any, instance_config: InstanceConfig | None) -> Self:
        from datasets.transfer import snapshot_from_model_for_instance

        return cast('Self', snapshot_from_model_for_instance(obj, instance_config))


def _primary_language_for_dataset(obj: Any) -> str:
    """Resolve the primary language for a Dataset via its scope's InstanceConfig."""
    scope = getattr(obj, 'scope', None)
    if scope is not None:
        lang = getattr(scope, 'primary_language', None)
        if lang:
            return lang
    return 'en'


def _label_from_identifier(identifier: str) -> str:
    return identifier.replace('_', ' ').replace('-', ' ').title()
