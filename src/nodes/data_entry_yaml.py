"""Explicit YAML adapter; stored layouts contain only UUID references."""

from typing import TYPE_CHECKING, Annotated, Literal
from uuid import UUID, uuid3

from pydantic import BaseModel, ConfigDict, Field

from kausal_common.i18n.pydantic import I18nBaseModel, I18nString

from nodes.defs.data_entry import (
    DataEntryAnchorSpec,
    DataEntryAutodiscoverSpec,
    DataEntryDatasetSpec,
    DataEntryPlacementSpec,
    DataEntrySectionAmendmentSpec,
    DataEntrySectionSpec,
    DataEntrySliceSpec,
    DataEntrySpec,
    DataEntryTableSpec,
)

if TYPE_CHECKING:
    from nodes.defs.graph import DatasetMeta, DimensionMeta
    from nodes.instance_serialization import NodeSnapshot


class YAMLAnchor(BaseModel):
    node: str
    output_port: str | None = None
    slices: list[dict[str, list[str]]] = Field(default_factory=list)


class YAMLTableBase(BaseModel):
    model_config = ConfigDict(extra='forbid')

    id: str | None = None
    uuid: UUID | None = None


class YAMLPlacement(YAMLTableBase):
    kind: Literal['input_port']

    input: str
    port: str | None = None
    slices: list[dict[str, list[str]]] = Field(default_factory=list)


class YAMLDataset(YAMLTableBase):
    kind: Literal['dataset']
    dataset: str
    metrics: list[str] | None = None
    slices: list[dict[str, list[str]]] = Field(default_factory=list)


class YAMLAutodiscover(YAMLTableBase):
    kind: Literal['autodiscover']
    anchors: list[YAMLAnchor]


type YAMLTable = Annotated[YAMLPlacement | YAMLDataset | YAMLAutodiscover, Field(discriminator='kind')]


class YAMLSection(I18nBaseModel):
    model_config = ConfigDict(extra='forbid')

    id: str | None = None
    uuid: UUID | None = None
    name: I18nString
    description: I18nString | None = None
    tables: list[YAMLTable] = Field(default_factory=list)


class YAMLAmendment(I18nBaseModel):
    model_config = ConfigDict(extra='forbid')

    section_id: UUID
    name: I18nString | None = None
    description: I18nString | None = None
    tables: list[YAMLTable] = Field(default_factory=list)


class YAMLDataEntry(BaseModel):
    namespace: UUID | None = None
    sections: list[YAMLSection] = Field(default_factory=list)
    amendments: list[YAMLAmendment] = Field(default_factory=list)


def resolve_yaml_data_entry(  # noqa: C901, PLR0915
    config: YAMLDataEntry,
    namespace: UUID,
    nodes: list[NodeSnapshot],
    dimensions: list[DimensionMeta],
    datasets: list[DatasetMeta],
) -> DataEntrySpec:
    namespace = config.namespace or namespace
    by_name = {node.identifier: node for node in nodes}
    dims = {dim.identifier: dim for dim in dimensions}
    datasets_by_name = {dataset.identifier: dataset for dataset in datasets if dataset.identifier is not None}
    if len(datasets_by_name) != sum(dataset.identifier is not None for dataset in datasets):
        raise ValueError('Ambiguous dataset identifiers in YAML data-entry catalog')

    def slices(items: list[dict[str, list[str]]]) -> list[DataEntrySliceSpec]:
        result = []
        for item in items:
            coordinates: dict[UUID, list[UUID]] = {}
            for name, categories in item.items():
                dimension = dims[name]
                lookup = {category.identifier: category.id for category in dimension.categories}
                coordinates[dimension.id] = [lookup[category] for category in categories]
            result.append(DataEntrySliceSpec(categories=coordinates))
        return result

    def anchors(items: list[YAMLAnchor]) -> list[DataEntryAnchorSpec]:
        result = []
        for item in items:
            node = by_name[item.node]
            assert node.spec is not None
            ports = [
                p for p in node.spec.output_ports if item.output_port is None or item.output_port in (p.identifier, str(p.id))
            ]
            if len(ports) != 1:
                raise ValueError(f'Anchor {item.node} requires an unambiguous output_port')
            result.append(DataEntryAnchorSpec(node_id=node.uuid, output_port_id=ports[0].id, slices=slices(item.slices)))
        return result

    def tables(items: list[YAMLTable], section_id: UUID) -> list[DataEntryTableSpec]:
        result: list[DataEntryTableSpec] = []
        for item in items:
            if item.uuid is None and item.id is None:
                raise ValueError('YAML table entries require uuid or id')
            table_id = item.uuid or uuid3(section_id, f'data-entry-table:{item.id}')
            if isinstance(item, YAMLAutodiscover):
                result.append(DataEntryAutodiscoverSpec(id=table_id, anchors=anchors(item.anchors)))
            elif isinstance(item, YAMLDataset):
                dataset = datasets_by_name.get(item.dataset)
                if dataset is None:
                    raise ValueError(f'Unknown data-entry dataset identifier: {item.dataset}')
                metrics = {metric.identifier: metric.id for metric in dataset.metrics}
                if item.metrics is not None and any(metric not in metrics for metric in item.metrics):
                    raise ValueError(f'Unknown metric identifier for data-entry dataset: {item.dataset}')
                result.append(
                    DataEntryDatasetSpec(
                        id=table_id,
                        dataset_id=dataset.id,
                        metric_ids=[metrics[metric] for metric in item.metrics] if item.metrics is not None else None,
                        slices=slices(item.slices),
                    )
                )
            else:
                node = by_name[item.input]
                assert node.spec is not None
                ports = [
                    p
                    for p in node.spec.input_ports
                    if p.binding_owner == 'instance' and (item.port is None or item.port in (p.identifier, str(p.id)))
                ]
                if len(ports) != 1:
                    raise ValueError(f'Placement {item.input} requires an unambiguous instance-owned port')
                result.append(
                    DataEntryPlacementSpec(
                        id=table_id,
                        node_id=node.uuid,
                        port_id=ports[0].id,
                        slices=slices(item.slices),
                    )
                )
        return result

    sections = []
    for item in config.sections:
        if item.uuid is None and item.id is None:
            raise ValueError('YAML sections require uuid or id')
        section_id = item.uuid or uuid3(namespace, f'data-entry-section:{item.id}')
        sections.append(
            DataEntrySectionSpec(
                id=section_id,
                identifier=item.id,
                name=item.name,
                description=item.description,
                tables=tables(item.tables, section_id),
            )
        )
    amendments = []
    for item in config.amendments:
        amendment = DataEntrySectionAmendmentSpec(
            section_id=item.section_id,
            tables=tables(item.tables, item.section_id),
        )
        for name in ('name', 'description'):
            if name in item.model_fields_set:
                setattr(amendment, name, getattr(item, name))
        amendments.append(DataEntrySectionAmendmentSpec.model_validate(amendment.model_dump()))
    return DataEntrySpec(sections=sections, amendments=amendments)
