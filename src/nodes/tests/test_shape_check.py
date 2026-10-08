"""The static shape check: a dataset's entry form projected onto the port it feeds."""

from typing import Any
from uuid import uuid4

import pytest

from nodes.constraints.solver import compile_constraint_program
from nodes.defs.graph import DatasetMeta, DatasetMetricMeta, DimensionCategoryMeta, DimensionMeta
from nodes.defs.instance_defs import InstanceMetadata, InstanceModelSpec
from nodes.defs.node_defs import DatasetPortSpec, NodeSpec, SimpleConfig
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.defs.shape_defs import ShapeCombinationSpec, ShapeRequiredGroupSpec, ShapeSpec
from nodes.defs.transform_def import FilterDimensionOp, TagOperationOp
from nodes.graphql.types.problems import ProblemEnforcement
from nodes.graphql.types.spec import InputPortType
from nodes.instance_graph import build_instance_graph
from nodes.instance_serialization import DatasetPortSnapshot, InstanceSnapshot, NodeSnapshot, unified_binding_snapshots
from nodes.units import unit_registry
from nodes.value_validation import ValueContract

pytestmark = pytest.mark.django_db

SHAPE_CODES = {'unknown_shape', 'shape_dimension_missing', 'outside_shape', 'shape_requirement_unreachable'}


def _dimension(identifier: str, *categories: str) -> DimensionMeta:
    return DimensionMeta(
        id=uuid4(),
        identifier=identifier,
        categories=tuple(DimensionCategoryMeta(id=uuid4(), identifier=category) for category in categories),
    )


def _shape(
    identifier: str,
    pairs: list[tuple[str, str]],
    *,
    required: list[list[tuple[str, str]]] | None = None,
    closed: bool = False,
) -> ShapeSpec:
    combinations = {
        pair: ShapeCombinationSpec(uuid=uuid4(), identifier='_'.join(pair), categories={'sector': pair[0], 'carrier': pair[1]})
        for pair in pairs
    }
    return ShapeSpec(
        uuid=uuid4(),
        identifier=identifier,
        dimensions=['sector', 'carrier'],
        closed=closed,
        combinations=list(combinations.values()),
        required=[
            ShapeRequiredGroupSpec(uuid=uuid4(), identifier=str(index), combinations=[combinations[pair].uuid for pair in group])
            for index, group in enumerate(required or [])
        ],
    )


def _findings(entry: ShapeSpec, needed: ShapeSpec, transformations: list[Any] | None = None) -> tuple[list[str], list[str]]:
    """Bind a dataset shaped by `entry` to a port whose contract needs `needed`; return the codes and notices."""
    sector = _dimension('sector', 'homes', 'industry')
    carrier = _dimension('carrier', 'gas', 'power')
    metric = DatasetMetricMeta(id=uuid4(), identifier='energy', unit='kWh')
    dataset = DatasetMeta(
        id=uuid4(),
        identifier='test/energy',
        schema_id=uuid4(),
        metrics=(metric,),
        declared_dimension_ids=(sector.id, carrier.id),
        shape_id=entry.uuid,
    )
    unit = unit_registry.parse_units('kWh')
    port = InputPortDef(id=uuid4(), unit=unit, validation=ValueContract(shape=needed.uuid))
    node = NodeSnapshot(
        uuid=uuid4(),
        identifier='consumer',
        spec=NodeSpec(
            type_config=SimpleConfig(node_class='simple.AdditiveNode'),
            input_ports=[port],
            output_ports=[OutputPortDef(id=uuid4(), identifier='default', unit=unit)],
        ),
    )
    binding = DatasetPortSnapshot(
        uuid=uuid4(),
        node=node.uuid,
        dataset='test/energy',
        dataset_uuid=dataset.id,
        port_id=port.id,
        metric='energy',
        metric_uuid=metric.id,
        spec=DatasetPortSpec(transformations=transformations or []),
    )
    graph = build_instance_graph(
        InstanceSnapshot(
            metadata=InstanceMetadata(uuid=uuid4(), identifier='shape-check', name='Shape check'),
            spec=InstanceModelSpec(shapes=[entry, needed]),
            nodes=[node],
            bindings=unified_binding_snapshots([], [binding]),
            dimensions=[sector, carrier],
            datasets=[dataset],
        )
    )
    program = compile_constraint_program(graph)
    codes = [conflict.code for conflict in program.static_conflicts if conflict.code in SHAPE_CODES]
    return codes, [notice.message for notice in program.static_notices]


def test_an_entry_form_that_covers_the_requirements_passes() -> None:
    entry = _shape('entry', [('homes', 'gas'), ('homes', 'power')])
    needed = _shape(
        'needed', [('homes', 'gas'), ('homes', 'power')], required=[[('homes', 'gas'), ('homes', 'power')]], closed=True
    )
    assert _findings(entry, needed) == ([], [])


def test_a_requirement_the_entry_form_cannot_hold_is_reported() -> None:
    entry = _shape('entry', [('homes', 'gas')])
    needed = _shape('needed', [('homes', 'gas'), ('industry', 'power')], required=[[('industry', 'power')]])
    assert _findings(entry, needed) == (['shape_requirement_unreachable'], [])


def test_the_binding_filter_is_projected_before_comparing() -> None:
    entry = _shape('entry', [('homes', 'gas'), ('industry', 'gas')])
    needed = _shape('needed', [('homes', 'gas'), ('industry', 'gas')], required=[[('industry', 'gas')]])
    assert _findings(entry, needed) == ([], [])
    excluded = [FilterDimensionOp(dimension='sector', categories=['industry'], exclude=True)]
    assert _findings(entry, needed, excluded) == (['shape_requirement_unreachable'], [])


def test_a_closed_port_shape_rejects_entry_rows_outside_it() -> None:
    entry = _shape('entry', [('homes', 'gas'), ('industry', 'power')])
    needed = _shape('needed', [('homes', 'gas')], closed=True)
    assert _findings(entry, needed) == (['outside_shape'], [])


def test_a_flattened_dimension_the_port_shape_constrains_is_missing() -> None:
    entry = _shape('entry', [('homes', 'gas')])
    needed = _shape('needed', [('homes', 'gas')])
    assert _findings(entry, needed, [FilterDimensionOp(dimension='carrier', flatten=True)]) == (['shape_dimension_missing'], [])


def test_a_binding_that_cannot_be_projected_is_a_notice_not_a_pass() -> None:
    entry = _shape('entry', [('homes', 'gas')])
    needed = _shape('needed', [('industry', 'power')], required=[[('industry', 'power')]], closed=True)
    codes, notices = _findings(entry, needed, [TagOperationOp(tag='complement')])
    assert codes == []
    (notice,) = notices
    assert 'cannot be checked' in notice


def test_an_input_port_exposes_what_its_contract_blocks() -> None:
    shape_id = uuid4()
    port = InputPortDef(
        id=uuid4(),
        unit=unit_registry.parse_units('kWh'),
        validation=ValueContract(shape=shape_id, enforcement='block_publish'),
    )
    assert InputPortType.contract_enforcement(InputPortType.from_def(port, [])) == ProblemEnforcement.BLOCK_PUBLISH
    bare = InputPortType.from_def(InputPortDef(id=uuid4(), unit=unit_registry.parse_units('kWh')), [])
    assert InputPortType.contract_enforcement(bare) is None
