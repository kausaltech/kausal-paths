"""Pure, observation-free ownership of dataset coordinates by data-entry sections."""

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid3

from kausal_common.i18n.pydantic import TranslatedString

from nodes.constraints.solver import AssignStep, FilterStep, OpaqueStep, resolve_binding_steps
from nodes.defs.binding_def import DatasetBindingDef, EdgeBindingDef
from nodes.defs.data_entry import (
    ComposedDataEntrySpec,
    DataEntryDatasetSpec,
    DataEntryPlacementSpec,
    DataEntrySectionSpec,
    DataEntrySliceSpec,
    DataEntrySpec,
    data_entry_dataset_ids,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kausal_common.i18n.pydantic import I18nString

    from nodes.defs.binding_def import AnyPortBindingDef
    from nodes.defs.data_entry import DataEntryAnchorSpec, DataEntrySectionAmendmentSpec
    from nodes.instance_graph import InstanceGraph


type Rectangle = tuple[tuple[UUID, frozenset[UUID]], ...]
type Placement = Literal['template_explicit', 'instance_explicit', 'derived', 'unplaced']


def rectangle(categories: Mapping[UUID, frozenset[UUID]]) -> Rectangle:
    return tuple(sorted(categories.items()))


def intersect(left: Rectangle, right: Rectangle) -> Rectangle | None:
    result = dict(left)
    for dimension, values in right:
        result[dimension] = result.get(dimension, values) & values
    if any(not values for values in result.values()):
        return None
    return rectangle(result)


def subtract(left: Rectangle, right: Rectangle) -> tuple[Rectangle, ...]:
    """Disjoint rectangles for left minus right; left has a finite category universe."""
    overlap = intersect(left, right)
    if overlap is None:
        return (left,)
    remainder = dict(left)
    pieces: list[Rectangle] = []
    for dimension, selected in overlap:
        outside = remainder[dimension] - selected
        if outside:
            pieces.append(rectangle({**remainder, dimension: outside}))
        remainder[dimension] = selected
    return tuple(pieces)


@dataclass(frozen=True)
class EntryProblem:
    code: str
    message: str
    section_id: UUID | None = None
    node_id: UUID | None = None
    port_id: UUID | None = None
    dataset_id: UUID | None = None
    blocks_publication: bool = True


@dataclass(frozen=True)
class EntrySelection:
    dataset_id: UUID
    metric_id: UUID
    rectangles: tuple[Rectangle, ...]
    placement: Placement
    approximate: bool = False
    path: tuple[UUID, ...] = ()
    reasons: tuple[str, ...] = ()
    node_id: UUID | None = None
    port_id: UUID | None = None
    table_id: UUID | None = None


@dataclass(frozen=True)
class EntrySection:
    id: UUID
    identifier: str | None
    name: I18nString
    description: I18nString | None = None
    selections: tuple[EntrySelection, ...] = ()
    problems: tuple[EntryProblem, ...] = ()
    kind: Literal['authored', 'unplaced'] = 'authored'


@dataclass(frozen=True)
class EntryLayout:
    sections: tuple[EntrySection, ...]
    problems: tuple[EntryProblem, ...]


@dataclass
class _Section:
    spec: DataEntrySectionSpec
    problems: list[EntryProblem] = field(default_factory=list)


class EntryResolver:
    def __init__(self, graph: InstanceGraph):
        self.graph = graph
        self.sections: dict[UUID, _Section] = {}
        self.claims: list[tuple[int, UUID, EntrySelection]] = []
        self.problems: list[EntryProblem] = []
        self.active_table: UUID | None = None
        self.claim_order: dict[tuple[UUID, UUID], int] = {}
        self.unplaced_id = uuid3(graph.instance_id, 'data-entry:unplaced')
        self.domains = {dim.id: frozenset(cat.id for cat in dim.categories) for dim in graph.dimensions}
        self.anchors: dict[tuple[UUID, UUID], list[tuple[UUID, UUID, DataEntryAnchorSpec]]] = {}

    def problem(
        self,
        code: str,
        message: str,
        section_id: UUID | None = None,
        *,
        blocks_publication: bool = True,
        **refs: UUID | None,
    ) -> None:
        problem = EntryProblem(code, message, section_id, blocks_publication=blocks_publication, **refs)
        self.problems.append(problem)
        if section_id in self.sections:
            self.sections[section_id].problems.append(problem)

    def unbound(self, section: UUID, node: UUID, port: UUID, *, explicit: bool = False) -> None:
        disconnected = (node, port) in self.graph.disconnected_inputs
        self.problem(
            'disconnected_input' if disconnected else 'unbound_input',
            'Input is explicitly disconnected' if disconnected else 'Input has no effective binding',
            section,
            blocks_publication=explicit,
            node_id=node,
            port_id=port,
        )

    def slices(self, specs: list[DataEntrySliceSpec]) -> tuple[Rectangle, ...]:
        return tuple(rectangle({dim: frozenset(cats) for dim, cats in spec.categories.items()}) for spec in specs) or ((),)

    def validate_slices(self, specs: list[DataEntrySliceSpec], section: UUID) -> bool:
        for spec in specs:
            for dim, categories in spec.categories.items():
                if dim not in self.domains or not set(categories) <= self.domains[dim]:
                    self.problem('invalid_slice', 'Unknown dimension or category in section selection', section)
                    return False
        return True

    def source_universe(self, binding: DatasetBindingDef) -> Rectangle:
        return rectangle({dim: self.domains.get(dim, frozenset()) for dim in binding.dataset.declared_dimension_ids})

    def translate(self, binding: AnyPortBindingDef, selection: Rectangle) -> tuple[Rectangle | None, tuple[str, ...]]:  # noqa: C901
        dims = frozenset(binding.dataset.declared_dimension_ids) if isinstance(binding, DatasetBindingDef) else None
        steps = resolve_binding_steps(self.graph, binding, [], dims)
        current = dict(selection)
        reasons: list[str] = []
        for step in reversed(steps):
            match step:
                case OpaqueStep():
                    current.clear()
                    reasons.append(step.reason)
                case AssignStep():
                    if step.category_id is None:
                        current.pop(step.dimension_id, None)
                        reasons.append('Unknown assigned category')
                    elif step.dimension_id in current and step.category_id not in current[step.dimension_id]:
                        return None, tuple(reasons)
                    else:
                        current.pop(step.dimension_id, None)
                case FilterStep():
                    if step.selection is None:
                        current.pop(step.dimension_id, None)
                        reasons.append('Unresolved category filter')
                        continue
                    selected = step.selection
                    if step.exclude:
                        selected = self.domains[step.dimension_id] - selected
                    if not step.flatten and step.dimension_id in current:
                        selected &= current[step.dimension_id]
                    if not selected:
                        return None, tuple(reasons)
                    current[step.dimension_id] = selected
        return rectangle(current), tuple(reasons)

    def visit_binding(  # noqa: C901, PLR0912
        self,
        section: UUID,
        binding: AnyPortBindingDef,
        selected: Rectangle,
        *,
        tier: int,
        path: tuple[UUID, ...] = (),
        reasons: tuple[str, ...] = (),
    ) -> None:
        if binding.id in path:
            self.problem('cycle', 'Cycle while resolving data-entry input', section)
            return
        if isinstance(binding, DatasetBindingDef) and (
            binding.dataset_uuid not in self.graph.dataset_by_id
            or binding.metric_uuid not in self.graph.dataset_by_id[binding.dataset_uuid].metric_by_id
        ):
            self.problem(
                'unresolved_source',
                'Dataset or metric is unavailable',
                section,
                node_id=binding.target_node.id,
                port_id=binding.target_port.id,
            )
            return
        translated, widened = self.translate(binding, selected)
        if translated is None:
            return
        if tier < 2 and widened:
            self.problem(
                'inexact_placement',
                'Explicit placement cannot be translated exactly',
                section,
                node_id=binding.target_node.id,
                port_id=binding.target_port.id,
            )
            return
        reasons = tuple(dict.fromkeys((*reasons, *widened)))
        path = (*path, binding.id)
        if isinstance(binding, DatasetBindingDef):
            if binding.target_port.binding_owner != 'instance':
                return
            if binding.dataset.is_external_placeholder and not self.graph.is_template:
                self.problem(
                    'external_placeholder',
                    'Input data has not been imported',
                    section,
                    blocks_publication=tier < 2,
                    node_id=binding.target_node.id,
                    port_id=binding.target_port.id,
                    dataset_id=binding.dataset.id,
                )
                return
            if tier < 2 and any(dim not in binding.dataset.declared_dimension_ids for dim, _ in translated):
                self.problem(
                    'invalid_slice',
                    'Placement restricts a dimension absent from its input',
                    section,
                    node_id=binding.target_node.id,
                    port_id=binding.target_port.id,
                )
                return
            translated = rectangle({dim: cats for dim, cats in translated if dim in binding.dataset.declared_dimension_ids})
            selected_source = intersect(self.source_universe(binding), translated)
            if selected_source is None:
                return
            assert binding.metric_uuid is not None
            claim = EntrySelection(
                binding.dataset.id,
                binding.metric_uuid,
                (selected_source,),
                'instance_explicit' if tier == 0 else 'template_explicit' if tier == 1 else 'derived',
                bool(reasons),
                path,
                reasons,
                binding.target_node.id,
                binding.target_port.id,
                self.active_table,
            )
            self.claims.append((tier if tier < 2 else 3 if reasons else 2, section, claim))
        else:
            remaining = [translated]
            # Finite expansion is only needed for subtracting a sliced boundary.
            for other, _table, anchor in self.anchors.get((binding.source_node.id, binding.source_port.id), []):
                if other == section:
                    continue
                for stop in self.slices(anchor.slices):
                    dimensions = {dim for dim, _ in stop}
                    remaining = [
                        rectangle({**{dim: self.domains[dim] for dim in dimensions}, **dict(item)}) for item in remaining
                    ]
                    remaining = [piece for item in remaining for piece in subtract(item, stop)]
            for item in remaining:
                self.visit_node(section, binding.source_node.id, binding.source_port.id, item, path, reasons)

    def visit_node(
        self,
        section: UUID,
        node_id: UUID,
        output_id: UUID,
        selected: Rectangle,
        path: tuple[UUID, ...] = (),
        reasons: tuple[str, ...] = (),
    ) -> None:
        node = self.graph.node_by_id[node_id]
        # Nodes own the declaration of category preservation. Unknown behavior cannot
        # safely pass an output restriction backwards to input coordinates.
        inputs = node.node_class.data_entry_dependencies(node, output_id)
        if inputs is None:
            inputs = tuple(port.id for port in node.spec.input_ports)
            selected = ()
            reasons = (*reasons, f'Unknown category mapping through {node.identifier or node.id}')
        for port in node.spec.input_ports:
            if port.id not in inputs:
                continue
            bindings = node.bindings_for_port(port.id)
            if not bindings and port.binding_owner == 'instance':
                self.unbound(section, node.id, port.id)
            for binding in bindings:
                self.visit_binding(section, binding, selected, tier=2, path=path, reasons=reasons)

    def explicit(self, section: UUID, placement: DataEntryPlacementSpec, tier: int) -> None:
        node = self.graph.node_by_id.get(placement.node_id)
        port = next((port for port in node.spec.input_ports if port.id == placement.port_id), None) if node else None
        if port is None or port.binding_owner != 'instance':
            self.problem(
                'invalid_placement',
                'Placement requires an existing instance-owned input port',
                section,
                node_id=placement.node_id,
                port_id=placement.port_id,
            )
            return
        if not self.validate_slices(placement.slices, section):
            return
        assert node is not None
        bindings = node.bindings_for_port(port.id)
        if not bindings:
            self.unbound(section, node.id, port.id, explicit=True)
        for binding in bindings:
            if isinstance(binding, EdgeBindingDef):
                self.problem(
                    'unsupported_source', 'Submodel input forms are not available', section, node_id=node.id, port_id=port.id
                )
                continue
            for selected in self.slices(placement.slices):
                self.visit_binding(section, binding, selected, tier=tier)

    def direct(self, section: UUID, table: DataEntryDatasetSpec, tier: int) -> None:
        dataset = self.graph.dataset_by_id.get(table.dataset_id)
        if dataset is None or (dataset.is_external_placeholder and not self.graph.is_template):
            self.problem(
                'unresolved_source', 'Dataset is unavailable or has not been imported', section, dataset_id=table.dataset_id
            )
            return
        if not self.validate_slices(table.slices, section):
            return
        if any(dim not in dataset.declared_dimension_ids for item in table.slices for dim in item.categories):
            self.problem(
                'invalid_slice', 'Placement restricts a dimension absent from its dataset', section, dataset_id=table.dataset_id
            )
            return
        metrics = (
            [metric.id for metric in dataset.metrics if metric.quality_of is None]
            if table.metric_ids is None
            else table.metric_ids
        )
        if any(metric not in dataset.metric_by_id or dataset.metric_by_id[metric].quality_of is not None for metric in metrics):
            self.problem(
                'invalid_metric', 'Dataset selection requires existing value metrics', section, dataset_id=table.dataset_id
            )
            return
        universe = rectangle({dim: self.domains.get(dim, frozenset()) for dim in dataset.declared_dimension_ids})
        for metric in metrics:
            for selected in self.slices(table.slices):
                source = intersect(universe, selected)
                if source is not None:
                    self.claims.append((
                        tier,
                        section,
                        EntrySelection(
                            dataset.id,
                            metric,
                            (source,),
                            'instance_explicit' if tier == 0 else 'template_explicit',
                            table_id=table.id,
                        ),
                    ))

    def resolve(self) -> EntryLayout:  # noqa: C901, PLR0912
        definition = self.graph.spec.data_entry
        if isinstance(definition, ComposedDataEntrySpec):
            base, local = definition.template, definition.local
        else:
            base, local = DataEntrySpec(), definition or DataEntrySpec()
        tiers: dict[tuple[UUID, UUID], int] = {}
        for tier, specs in ((1, base.sections), (0, local.sections)):
            for spec in specs:
                if spec.id in self.sections:
                    self.problem('duplicate_section', 'Local section shadows an inherited UUID', spec.id)
                    continue
                self.sections[spec.id] = _Section(spec.model_copy(deep=True))
                tiers.update({(spec.id, table.id): tier for table in spec.tables})
        for amendment in (*base.amendments, *local.amendments):
            target = self.sections.get(amendment.section_id)
            if target is None or amendment.section_id not in {section.id for section in base.sections}:
                self.problem('missing_section', 'Amendment target is not an inherited section', amendment.section_id)
                continue
            for name in ('name', 'description'):
                if name in amendment.model_fields_set:
                    setattr(target.spec, name, getattr(amendment, name))
            tables = {table.id: table for table in target.spec.tables}
            tables.update({table.id: table for table in amendment.tables})
            target.spec.tables = list(tables.values())
            tiers.update({(amendment.section_id, table.id): 0 for table in amendment.tables})
        table_sections: dict[UUID, UUID] = {}
        for section_id, section in self.sections.items():
            unique_tables = []
            for table in section.spec.tables:
                if table.id in table_sections:
                    self.problem('duplicate_table', f'Table UUID {table.id} belongs to another entry', section_id)
                    continue
                table_sections[table.id] = section_id
                unique_tables.append(table)
            section.spec.tables = unique_tables
        authored_entries: list[DataEntrySectionSpec | DataEntrySectionAmendmentSpec] = [
            *base.sections,
            *local.amendments,
            *local.sections,
        ]
        self.claim_order = {
            (entry.id if isinstance(entry, DataEntrySectionSpec) else entry.section_id, table.id): index
            for index, (entry, table) in enumerate((entry, table) for entry in authored_entries for table in entry.tables)
        }
        for section_id, section in self.sections.items():
            for table in section.spec.tables:
                self.active_table = table.id
                tier = tiers[section_id, table.id]
                if isinstance(table, DataEntryPlacementSpec):
                    self.explicit(section_id, table, tier)
                elif isinstance(table, DataEntryDatasetSpec):
                    self.direct(section_id, table, tier)
                else:
                    for anchor in table.anchors:
                        node = self.graph.node_by_id.get(anchor.node_id)
                        if node is None or not any(port.id == anchor.output_port_id for port in node.spec.output_ports):
                            self.problem(
                                'invalid_anchor', 'Anchor output port is unavailable', section_id, node_id=anchor.node_id
                            )
                            continue
                        if self.validate_slices(anchor.slices, section_id):
                            self.anchors.setdefault((anchor.node_id, anchor.output_port_id), []).append((
                                section_id,
                                table.id,
                                anchor,
                            ))
        for (node, output), anchors in self.anchors.items():
            for section, table_id, anchor in anchors:
                self.active_table = table_id
                for selected in self.slices(anchor.slices):
                    self.visit_node(section, node, output, selected)
        return self.partition()

    def partition(self) -> EntryLayout:  # noqa: C901, PLR0912, PLR0915
        owned: dict[tuple[UUID, UUID], list[Rectangle]] = {}
        owners: dict[tuple[UUID, UUID], list[tuple[UUID, Rectangle]]] = {}
        assigned: dict[UUID, list[EntrySelection]] = {section: [] for section in self.sections}
        explicit_owners: dict[tuple[UUID, UUID], list[tuple[int, UUID | None, Rectangle]]] = {}
        order = {section: index for index, section in enumerate(self.sections)}
        table_order = {
            (section, table.id): index
            for section, entry in self.sections.items()
            for index, table in enumerate(entry.spec.tables)
        }

        def claim_priority(item: tuple[int, UUID, EntrySelection]) -> tuple[int, int, int]:
            tier, section, claim = item
            key = (section, claim.table_id) if claim.table_id is not None else None
            if tier < 2:
                return tier, self.claim_order.get(key, 0) if key is not None else 0, 0
            return tier, order[section], table_order.get(key, 0) if key is not None else 0

        for tier, section_id, claim in sorted(self.claims, key=claim_priority):
            key = (claim.dataset_id, claim.metric_id)
            if tier < 2:
                if any(
                    previous_tier == tier and table_id != claim.table_id and intersect(selected, taken) is not None
                    for previous_tier, table_id, taken in explicit_owners.get(key, [])
                    for selected in claim.rectangles
                ):
                    self.problem(
                        'overlapping_tables',
                        'Explicit table entries select overlapping cells',
                        section_id,
                        dataset_id=claim.dataset_id,
                    )
                explicit_owners.setdefault(key, []).extend((tier, claim.table_id, item) for item in claim.rectangles)
            remaining = list(claim.rectangles)
            for taken in owned.get(key, []):
                remaining = [piece for item in remaining for piece in subtract(item, taken)]
            if tier >= 2 and any(
                owner != section_id and intersect(selected, taken) is not None
                for owner, taken in owners.get(key, [])
                for selected in claim.rectangles
            ):
                self.problem(
                    'competing_claim',
                    'Selection also reaches cells already assigned elsewhere',
                    section_id,
                    dataset_id=claim.dataset_id,
                )
            if not remaining:
                continue
            owned.setdefault(key, []).extend(remaining)
            if tier >= 2:
                owners.setdefault(key, []).extend((section_id, item) for item in remaining)
            if claim.approximate:
                self.problem(
                    'approximate_placement',
                    '; '.join(claim.reasons),
                    section_id,
                    node_id=claim.node_id,
                    port_id=claim.port_id,
                    dataset_id=claim.dataset_id,
                )
            assigned[section_id].append(replace(claim, rectangles=tuple(remaining)))
        for section_id, selections in assigned.items():
            entry_order = {table.id: index for index, table in enumerate(self.sections[section_id].spec.tables)}
            selections.sort(
                key=lambda item: (entry_order[item.table_id] if item.table_id is not None else -1, str(item.dataset_id))
            )
        leftovers: list[EntrySelection] = []
        for binding in self.graph.bindings:
            if not isinstance(binding, DatasetBindingDef) or binding.target_port.binding_owner != 'instance':
                continue
            if binding.dataset_uuid not in self.graph.dataset_by_id or binding.metric_uuid is None:
                continue
            if binding.dataset.is_external_placeholder and not self.graph.is_template:
                continue
            key = (binding.dataset.id, binding.metric_uuid)
            remaining = [self.source_universe(binding)]
            for taken in owned.get(key, []):
                remaining = [piece for item in remaining for piece in subtract(item, taken)]
            owned.setdefault(key, []).extend(remaining)
            if remaining:
                leftovers.append(EntrySelection(*key, tuple(remaining), 'unplaced', reasons=('No section claims these cells',)))
        for dataset_id in sorted(data_entry_dataset_ids(self.graph.spec.data_entry)):
            dataset = self.graph.dataset_by_id.get(dataset_id)
            if dataset is None or (dataset.is_external_placeholder and not self.graph.is_template):
                continue
            universe = rectangle({dim: self.domains.get(dim, frozenset()) for dim in dataset.declared_dimension_ids})
            for metric in dataset.metrics:
                if metric.quality_of is not None:
                    continue
                key = (dataset_id, metric.id)
                remaining = [universe]
                for taken in owned.get(key, []):
                    remaining = [piece for item in remaining for piece in subtract(item, taken)]
                if remaining:
                    leftovers.append(
                        EntrySelection(*key, tuple(remaining), 'unplaced', reasons=('No section claims these cells',))
                    )
        unplaced_problems = self.unplaced_problems()
        result = [
            EntrySection(
                section_id,
                section.spec.identifier,
                section.spec.name,
                section.spec.description,
                tuple(assigned[section_id]),
                tuple(section.problems),
            )
            for section_id, section in self.sections.items()
        ]
        if leftovers or unplaced_problems:
            result.append(
                EntrySection(
                    self.unplaced_id,
                    'other',
                    TranslatedString(en='Other data', de='Weitere Daten', fi='Muut tiedot'),
                    selections=tuple(leftovers),
                    problems=tuple(unplaced_problems),
                    kind='unplaced',
                )
            )
        return EntryLayout(tuple(result), tuple(self.problems))

    def unplaced_problems(self) -> list[EntryProblem]:
        associated = {(p.node_id, p.port_id) for p in self.problems}
        unplaced_problems: list[EntryProblem] = []
        for node in self.graph.nodes:
            for port in node.spec.input_ports:
                if port.binding_owner != 'instance' or (node.id, port.id) in associated:
                    continue
                bindings = node.bindings_for_port(port.id)
                code: str | None = None
                message = ''
                if not bindings:
                    disconnected = (node.id, port.id) in self.graph.disconnected_inputs
                    code = 'disconnected_input' if disconnected else 'unbound_input'
                    message = 'Input is explicitly disconnected' if disconnected else 'Input has no effective binding'
                elif any(
                    isinstance(binding, DatasetBindingDef)
                    and (
                        binding.dataset_uuid not in self.graph.dataset_by_id
                        or binding.metric_uuid not in self.graph.dataset_by_id[binding.dataset_uuid].metric_by_id
                    )
                    for binding in bindings
                ):
                    code, message = 'unresolved_source', 'Dataset or metric is unavailable'
                elif not self.graph.is_template and any(
                    isinstance(binding, DatasetBindingDef) and binding.dataset.is_external_placeholder for binding in bindings
                ):
                    code, message = 'external_placeholder', 'Input data has not been imported'
                elif any(isinstance(binding, EdgeBindingDef) for binding in bindings):
                    code, message = 'unsupported_source', 'Submodel input forms are not available'
                if code:
                    problem = EntryProblem(code, message, self.unplaced_id, node.id, port.id, blocks_publication=False)
                    unplaced_problems.append(problem)
                    self.problems.append(problem)
        return unplaced_problems


def resolve_data_entry(graph: InstanceGraph) -> EntryLayout:
    return EntryResolver(graph).resolve()
