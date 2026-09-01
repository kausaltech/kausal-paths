"""
Tests for the port binding vocabulary: transform ops, port identifiers, binding identity.

See `docs/architecture/dimension-constraints.md` for the model these encode.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from nodes.defs.binding_def import DatasetBindingDef, EdgeBindingDef, NodePortRef, PortBindingDef
from nodes.defs.node_defs import (
    ColumnDatasetFilterDef,
    DatasetPortSpec,
    DimensionDatasetFilterDef,
    InputDatasetDef,
    NodeSpec,
    RenameColumnDatasetFilterDef,
    RenameItemDatasetFilterDef,
    input_dataset_filter_to_ops,
)
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.defs.transform_def import (
    AssignDimensionOp,
    FilterColumnOp,
    FilterDimensionOp,
    InterpolateOp,
    PortTransformOp,
    RenameColumnOp,
    RenameItemOp,
    SelectMetricOp,
    SetForecastFromOp,
    unsupported_transformations_for_binding,
)
from nodes.instance_from_db import _serialize_dataset_groups
from nodes.instance_serialization import DatasetMetricSource, InputBindingSnapshot, group_unified_dataset_bindings
from nodes.spec_export import _drop_ambiguous_port_identifiers, _port_identifier_for_column
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


# ---------------------------------------------------------------------------
# Transform op vocabulary
# ---------------------------------------------------------------------------


def test_transform_pipeline_reproduces_runtime_order():
    """
    The pipeline order is the contract: executing it literally must equal the old behaviour.

    The old loading sequence was hardcoded — renames before anything looked at
    columns, then metric selection, temporal indexing and year remapping, then
    forecast synthesis *before* the other filters, then tag transformations, then the
    output shaping. Every one of those stages is an op now, so this list is what
    guarantees the executor reproduces it.
    """
    ds_def = InputDatasetDef(
        id='some/dataset',
        column='Trucks and lorries',
        forecast_from=2025,
        min_year=2010,
        dropna=True,
        unit=unit_registry.parse_units('kt/a'),
        tags=['prepare_gpc_dataset'],
        filters=[
            RenameColumnDatasetFilterDef(rename_col='Old', value='New'),
            DimensionDatasetFilterDef(dimension='sector', categories=['transport']),
            ColumnDatasetFilterDef(column='action', value='some_action'),
        ],
    )

    kinds = [op.kind for op in ds_def.to_transformations()]

    assert kinds == [
        'rename_column',
        'select_metric',
        'index_temporal',
        'remap_legacy_years',
        'set_forecast_from',
        'filter_dimension',
        'filter_column',
        'tag_operation',
        'filter_temporal',
        'drop_nulls',
        'ensure_unit',
    ]


def test_select_metric_carries_no_column():
    """
    Which metric is selected is the binding's source reference, not an op parameter.

    The op says only *where* the selection happens, so there is exactly one
    place to edit the metric.
    """
    ds_def = InputDatasetDef(id='some/dataset', column='Value')

    ops = ds_def.to_transformations()

    assert [op.kind for op in ops] == ['select_metric', 'index_temporal', 'remap_legacy_years']
    assert ops[0].model_dump() == {'kind': 'select_metric'}


def test_interpolate_is_an_operation():
    """The YAML boolean is converted into the binding's ordered recipe."""
    ds_def = InputDatasetDef(id='some/dataset', interpolate=True)

    assert isinstance(ds_def.to_transformations()[-1], InterpolateOp)
    assert isinstance(DatasetPortSpec.from_input_dataset(ds_def).transformations[-1], InterpolateOp)


def test_legacy_temporal_flags_are_normalized_into_the_stored_pipeline():
    spec = DatasetPortSpec.model_validate({
        'transformations': [{'kind': 'index_temporal'}],
        'interpolate': True,
        'backfill': True,
        'extend': True,
        # Retired field; rows written before removal still carry it.
        'output_dimensions': ['sector'],
    })

    assert [op.kind for op in spec.transformations] == ['index_temporal', 'interpolate', 'backfill', 'extend']
    assert set(spec.model_dump()) == {'transformations', 'column', 'tags', 'input_dataset'}


def test_forecast_from_and_unit_are_derived_from_the_pipeline():
    """They are stored once, as ops; the accessors read them back out."""
    ds_def = InputDatasetDef(id='some/dataset', forecast_from=2025, unit=unit_registry.parse_units('kt/a'))

    spec = DatasetPortSpec.from_input_dataset(ds_def)

    assert spec.forecast_from == 2025
    assert spec.unit == unit_registry.parse_units('kt/a')
    assert spec.without_forecast_from().forecast_from is None


def test_dimension_filter_with_assign_category_splits_into_two_ops():
    filter_def = DimensionDatasetFilterDef(dimension='sector', assign_category='transport', flatten=True)

    ops = input_dataset_filter_to_ops(filter_def)

    assert ops == [
        AssignDimensionOp(dimension='sector', category='transport'),
        FilterDimensionOp(dimension='sector', flatten=True),
    ]


def test_dimension_filter_that_both_selects_and_assigns_is_rejected():
    """The runtime silently ignores the assignment, so the intent is unrecoverable."""
    filter_def = DimensionDatasetFilterDef(dimension='sector', categories=['transport'], assign_category='buildings')

    with pytest.raises(ValueError, match='both selects and assigns'):
        input_dataset_filter_to_ops(filter_def)


def test_legacy_filters_map_onto_ops():
    assert input_dataset_filter_to_ops(ColumnDatasetFilterDef(column='action', values=['a', 'b'])) == [
        FilterColumnOp(column='action', values=['a', 'b'])
    ]
    assert input_dataset_filter_to_ops(RenameColumnDatasetFilterDef(rename_col='Old', value='New')) == [
        RenameColumnOp(column='Old', new_name='New')
    ]
    assert input_dataset_filter_to_ops(RenameItemDatasetFilterDef(rename_item='sector|old', value='new')) == [
        RenameItemOp(column='sector', old_item='old', new_item='new')
    ]


def test_dataset_only_ops_are_rejected_for_edge_bindings():
    ops: list[PortTransformOp] = [FilterDimensionOp(dimension='sector'), SetForecastFromOp(year=2025), SelectMetricOp()]

    assert unsupported_transformations_for_binding(ops, 'dataset') == []
    assert [op.kind for op in unsupported_transformations_for_binding(ops, 'edge')] == ['set_forecast_from', 'select_metric']


def test_operations_keep_their_discriminator_when_defaults_are_excluded():
    """
    Parameterless ops must not serialize to `{}`.

    `kind` always equals its default, so `exclude_defaults` drops it first — and
    without it the union cannot be deserialized. Config dicts for the loader are
    dumped exactly that way.
    """
    ds_def = InputDatasetDef(id='some/dataset', column='C', forecast_from=2025)

    dumped = (
        DatasetPortSpec
        .from_input_dataset(ds_def)
        .to_input_dataset(id='some/dataset')
        .model_dump(mode='json', exclude_defaults=True, exclude_none=True)
    )

    assert [op['kind'] for op in dumped['transformations']] == [
        'select_metric',
        'index_temporal',
        'remap_legacy_years',
        'set_forecast_from',
    ]
    assert [op.kind for op in (InputDatasetDef.model_validate(dumped).transformations or [])] == [
        'select_metric',
        'index_temporal',
        'remap_legacy_years',
        'set_forecast_from',
    ]


# ---------------------------------------------------------------------------
# Port identifiers
# ---------------------------------------------------------------------------


def _input_port(identifier: str | None) -> InputPortDef:
    return InputPortDef(id=uuid4(), identifier=identifier)


def test_port_identifiers_are_optional():
    spec = NodeSpec(input_ports=[_input_port(None), _input_port(None)])

    assert spec.input_port_by_identifier == {}


def test_port_identifiers_index_the_ports_that_have_them():
    named = _input_port('electricity')
    spec = NodeSpec(input_ports=[named, _input_port(None)])

    assert spec.input_port_by_identifier == {'electricity': named}


def test_duplicate_input_port_identifiers_are_rejected():
    with pytest.raises(ValueError, match="Duplicate input port identifier 'fuel'"):
        NodeSpec(input_ports=[_input_port('fuel'), _input_port('fuel')])


def test_input_and_output_ports_have_separate_identifier_namespaces():
    spec = NodeSpec(
        input_ports=[_input_port('energy')],
        output_ports=[OutputPortDef(id=uuid4(), identifier='energy', unit=unit_registry.parse_units('TJ/a'))],
    )

    assert set(spec.input_port_by_identifier) == {'energy'}
    assert set(spec.output_port_by_identifier) == {'energy'}


def test_port_identifier_is_derived_from_the_bound_column_when_usable():
    assert _port_identifier_for_column('population') == 'population'
    assert _port_identifier_for_column('Electricity') == 'Electricity'
    # The generic column name is not a usable port name.
    assert _port_identifier_for_column('Value') is None
    # Legacy wide-DVC labels are not identifier-shaped.
    assert _port_identifier_for_column('Trucks and lorries') is None


def test_identifiers_that_would_collide_on_export_are_dropped():
    """
    Derived names are not unique: two datasets can expose the same column.

    An unnamed port is better than two ports sharing a name, and the editor can
    always assign one afterwards.
    """
    ports = [_input_port('emissions'), _input_port('emissions'), _input_port('energy')]

    _drop_ambiguous_port_identifiers('some_node', ports)

    assert [port.identifier for port in ports] == [None, None, 'energy']


# ---------------------------------------------------------------------------
# Binding defs
# ---------------------------------------------------------------------------


def test_bindings_share_a_port_ref_through_the_base_class():
    port_ref = NodePortRef(node_id='target', port_id=uuid4())
    edge = EdgeBindingDef(id=uuid4(), port_ref=port_ref, from_ref=NodePortRef(node_id='source', port_id=uuid4()))
    dataset = DatasetBindingDef(id=uuid4(), port_ref=port_ref, external_dataset_id='some/dataset')

    for binding in (edge, dataset):
        assert isinstance(binding, PortBindingDef)
        assert binding.port_ref.node_id == 'target'
    assert (edge.kind, dataset.kind) == ('edge', 'dataset')


# ---------------------------------------------------------------------------
# DatasetPortSpec round-trip
# ---------------------------------------------------------------------------


def test_spec_reads_the_shape_that_stored_the_pipeline_as_operations():
    """
    Rows written while the field was called `operations` still read correctly.

    `operations` now means a node's own computation, so the binding's list was
    renamed — but stored rows keep the old key until bindings move to their own
    table.
    """
    spec = DatasetPortSpec.model_validate({
        'operations': [{'kind': 'select_metric'}, {'kind': 'set_forecast_from', 'year': 2024}],
        'column': 'Cars',
    })

    assert [op.kind for op in spec.transformations] == ['select_metric', 'set_forecast_from']
    assert spec.forecast_from == 2024
    assert spec.column == 'Cars'


def test_interpolate_survives_the_dataset_port_spec_round_trip():
    """
    `interpolate` used to be dropped for DB-sourced instances.

    `DatasetPortSpec` had no such field, and the node-level
    `input_dataset_processors` fallback only reaches datasets declared as bare
    strings — which the DB path never emits.
    """
    ds_def = InputDatasetDef(id='potsdam/pro_potsdam_renovation', forecast_from=2025, interpolate=True)

    spec = DatasetPortSpec.from_input_dataset(ds_def)
    assert any(isinstance(op, InterpolateOp) for op in spec.transformations)

    round_tripped = spec.to_input_dataset(id=ds_def.id)
    assert any(isinstance(op, InterpolateOp) for op in round_tripped.to_transformations())
    assert round_tripped.forecast_from is None, 'the flat field is gone; the pipeline carries it'
    assert [op.kind for op in (round_tripped.transformations or [])] == [
        'index_temporal',
        'remap_legacy_years',
        'set_forecast_from',
        'interpolate',
    ]


# ---------------------------------------------------------------------------
# Binding identity: how unified dataset rows group back into input datasets
# ---------------------------------------------------------------------------


def _dataset_row(node: UUID, dataset: str, metric: str, port_id: UUID, transformations: list[PortTransformOp]):
    return InputBindingSnapshot(
        uuid=uuid4(),
        node_id=node,
        port_id=port_id,
        source=DatasetMetricSource(dataset=dataset, metric=metric),
        transformations=transformations,
    )


def _serialize_node_rows(
    node: UUID,
    rows: list[InputBindingSnapshot],
    node_spec: NodeSpec | None = None,
) -> list[dict[str, Any]]:
    groups = group_unified_dataset_bindings([(row, 0) for row in rows], {node: node_spec})
    return _serialize_dataset_groups(groups.get(node, []))


def test_two_bindings_of_one_dataset_stay_separate_when_specs_match():
    """
    A metric-selecting row is a binding of its own, even when another looks identical.

    Grouping by serialized spec merged distinct bindings that happened to look
    alike, which changes the length of the node's `input_dataset_instances` and
    so breaks nodes that index into it.
    """
    node = uuid4()
    rows = [_dataset_row(node, 'shared', 'Value', uuid4(), [SelectMetricOp()]) for _ in range(2)]

    input_datasets = _serialize_node_rows(node, rows)

    assert len(input_datasets) == 2
    assert [ds['id'] for ds in input_datasets] == ['shared', 'shared']


def test_ports_of_one_binding_collapse_to_a_single_input_dataset():
    """A column-less binding expands to one row per metric but stays one input dataset."""
    node = uuid4()
    transformations = DatasetPortSpec.from_input_dataset(InputDatasetDef(id='ds', forecast_from=2025)).transformations

    rows = [_dataset_row(node, 'multi_metric', name, uuid4(), list(transformations)) for name in ('emissions', 'energy')]

    input_datasets = _serialize_node_rows(node, rows)

    assert len(input_datasets) == 1
    assert input_datasets[0]['id'] == 'multi_metric'
    assert [op['kind'] for op in input_datasets[0]['transformations']] == [
        'index_temporal',
        'remap_legacy_years',
        'set_forecast_from',
    ]


def test_diverging_pipelines_within_one_group_keep_the_first():
    """
    The group-level spec is recovered from the first row in port-declaration order.

    The retired per-row `dataset_spec` mirror could diverge within one binding
    (a stale mirror) and warned about it; with the pipeline in native columns,
    the first row of the fan-out is authoritative for the group.
    """
    node = uuid4()
    port_a, port_b = uuid4(), uuid4()
    node_spec = NodeSpec(input_ports=[InputPortDef(id=port_a), InputPortDef(id=port_b)])
    rows = [
        _dataset_row(
            node,
            'diverged',
            metric,
            port_id,
            DatasetPortSpec.from_input_dataset(InputDatasetDef(id='diverged', forecast_from=year)).transformations,
        )
        for metric, port_id, year in (('emissions', port_a, 2025), ('energy', port_b, 2030))
    ]

    input_datasets = _serialize_node_rows(node, rows, node_spec)

    assert len(input_datasets) == 1
    forecast_ops = [op for op in input_datasets[0]['transformations'] if op['kind'] == 'set_forecast_from']
    assert forecast_ops == [{'kind': 'set_forecast_from', 'year': 2025}], 'the first row wins'
