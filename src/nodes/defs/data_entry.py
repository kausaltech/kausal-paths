"""Authored data-entry layout, independent of datasets and observations."""

from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from kausal_common.i18n.pydantic import I18nBaseModel, I18nString

from paths.identifiers import DataEntrySectionId, DataEntryTableId
from paths.refs import DataEntrySectionRef, DatasetMetricRef, DatasetRef, DimensionCategoryRef, DimensionRef, NodeRef, PortRef

if TYPE_CHECKING:
    from uuid import UUID


class DataEntrySliceSpec(BaseModel):
    categories: dict[DimensionRef, list[DimensionCategoryRef]] = Field(default_factory=dict)


class DataEntryAnchorSpec(BaseModel):
    node_id: NodeRef
    output_port_id: PortRef
    slices: list[DataEntrySliceSpec] = Field(default_factory=list)


class DataEntryTableBase(BaseModel):
    model_config = ConfigDict(extra='forbid')

    id: DataEntryTableId


class DataEntryPlacementSpec(DataEntryTableBase):
    kind: Literal['input_port'] = 'input_port'
    node_id: NodeRef
    port_id: PortRef
    slices: list[DataEntrySliceSpec] = Field(default_factory=list)


class DataEntryDatasetSpec(DataEntryTableBase):
    kind: Literal['dataset'] = 'dataset'
    dataset_id: DatasetRef
    metric_ids: list[DatasetMetricRef] | None = None
    slices: list[DataEntrySliceSpec] = Field(default_factory=list)


class DataEntryAutodiscoverSpec(DataEntryTableBase):
    kind: Literal['autodiscover'] = 'autodiscover'
    anchors: list[DataEntryAnchorSpec]


type DataEntryTableSpec = Annotated[
    DataEntryPlacementSpec | DataEntryDatasetSpec | DataEntryAutodiscoverSpec, Field(discriminator='kind')
]


class DataEntrySectionSpec(I18nBaseModel):
    model_config = ConfigDict(extra='forbid')

    id: DataEntrySectionId
    identifier: str | None = None
    name: I18nString
    description: I18nString | None = None
    tables: list[DataEntryTableSpec] = Field(default_factory=list)


class DataEntrySectionAmendmentSpec(I18nBaseModel):
    model_config = ConfigDict(extra='forbid')

    section_id: DataEntrySectionRef
    name: I18nString | None = None
    description: I18nString | None = None
    tables: list[DataEntryTableSpec] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_name(self) -> DataEntrySectionAmendmentSpec:
        if 'name' in self.model_fields_set and self.name is None:
            raise ValueError('An amended section name cannot be null')
        return self

    @model_serializer(mode='wrap')
    def serialize_sparse(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        for field in ('name', 'description'):
            if field not in self.model_fields_set:
                data.pop(field, None)
        return data


class DataEntrySpec(BaseModel):
    kind: Literal['authored'] = 'authored'
    sections: list[DataEntrySectionSpec] = Field(default_factory=list)
    amendments: list[DataEntrySectionAmendmentSpec] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_unique(self) -> DataEntrySpec:
        ids = [section.id for section in self.sections]
        if len(ids) != len(set(ids)):
            raise ValueError('Data-entry section UUIDs must be unique')
        targets = [amendment.section_id for amendment in self.amendments]
        if len(targets) != len(set(targets)):
            raise ValueError('Only one amendment per inherited section is allowed')
        table_ids = [table.id for entry in _layout_entries(self) for table in entry.tables]
        if len(table_ids) != len(set(table_ids)):
            raise ValueError('Data-entry table UUIDs must be unique')
        return self


class ComposedDataEntrySpec(BaseModel):
    """Frozen-edition composition inputs; never persisted as a local authored layout."""

    kind: Literal['composed'] = 'composed'
    template: DataEntrySpec
    local: DataEntrySpec


type DataEntryDefinition = Annotated[DataEntrySpec | ComposedDataEntrySpec, Field(discriminator='kind')]


def remap_data_entry(  # noqa: C901, PLR0912
    spec: DataEntrySpec,
    identities: dict[UUID, UUID],
) -> DataEntrySpec:
    """Copy authored declarations while remapping only identities owned by the copy."""
    result = spec.model_copy(deep=True)
    for section in result.sections:
        section.id = identities.get(section.id, section.id)
    for amendment in result.amendments:
        amendment.section_id = identities.get(amendment.section_id, amendment.section_id)
    entries: list[DataEntrySectionSpec | DataEntrySectionAmendmentSpec] = [*result.sections, *result.amendments]
    for entry in entries:
        for table in entry.tables:
            table.id = identities.get(table.id, table.id)
            if isinstance(table, DataEntryAutodiscoverSpec):
                selections: list[DataEntryAnchorSpec | DataEntryPlacementSpec | DataEntryDatasetSpec] = list(table.anchors)
            else:
                selections = [table]
            for item in selections:
                if isinstance(item, DataEntryDatasetSpec):
                    item.dataset_id = identities.get(item.dataset_id, item.dataset_id)
                    if item.metric_ids is not None:
                        item.metric_ids = [identities.get(metric, metric) for metric in item.metric_ids]
                else:
                    item.node_id = identities.get(item.node_id, item.node_id)
                    if isinstance(item, DataEntryAnchorSpec):
                        item.output_port_id = identities.get(item.output_port_id, item.output_port_id)
                    else:
                        item.port_id = identities.get(item.port_id, item.port_id)
                for selection in item.slices:
                    selection.categories = {
                        identities.get(dim, dim): [identities.get(cat, cat) for cat in cats]
                        for dim, cats in selection.categories.items()
                    }
    return result


def data_entry_dataset_ids(definition: DataEntryDefinition | None) -> set[UUID]:
    """Datasets explicitly referenced by a layout, including unbound inputs."""
    if definition is None:
        return set()
    specs = (definition.template, definition.local) if isinstance(definition, ComposedDataEntrySpec) else (definition,)
    return {
        table.dataset_id
        for spec in specs
        for section in _layout_entries(spec)
        for table in section.tables
        if isinstance(table, DataEntryDatasetSpec)
    }


def _layout_entries(spec: DataEntrySpec) -> list[DataEntrySectionSpec | DataEntrySectionAmendmentSpec]:
    return [*spec.sections, *spec.amendments]
