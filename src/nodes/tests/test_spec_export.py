from __future__ import annotations

from typing import Any, cast

import pytest

from kausal_common.i18n.pydantic import TranslatedString

from nodes.defs.transform_def import FilterColumnOp, PortTransformOp, RenameColumnOp, resolve_metric_columns
from nodes.node import Node
from nodes.simple import AdditiveNode, MultiplicativeNode, SectorEmissions
from nodes.spec_export import _export_input_ports, pair_metrics_to_columns
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from nodes.units import unit_registry


def _make_context(identifier: str = 'multi-port-export-test'):
    instance = InstanceFactory.create(id=identifier, name=identifier)
    InstanceConfigFactory.create(identifier=instance.id, instance=instance, name=identifier)
    return instance.context


def _make_node(context, cls=Node, identifier: str = 'node', unit: str = 'kt/a', quantity: str = 'emissions'):
    return cls(
        id=identifier,
        context=context,
        name=TranslatedString(identifier, default_language='en'),
        unit=unit_registry.parse_units(unit),
        quantity=quantity,
    )


@pytest.mark.django_db
def test_generic_node_port_roles_have_one_shared_typed_namespace():
    assert AdditiveNode.input_port_declarations == (AdditiveNode.additive_port, AdditiveNode.impute_port)
    assert AdditiveNode.additive_port.role == 'additive'
    assert AdditiveNode.output_port.role == 'output'
    assert AdditiveNode.output_port.identifier == 'default'

    assert MultiplicativeNode.input_port_declarations == (
        MultiplicativeNode.factors_port,
        MultiplicativeNode.additive_port,
        MultiplicativeNode.impute_port,
    )
    assert MultiplicativeNode.factors_port.repeatable is True
    assert MultiplicativeNode.factors_port.min_count == 1
    assert MultiplicativeNode.factors_port.effective_default_count == 2
    assert MultiplicativeNode.additive_port.multi is True
    assert MultiplicativeNode.additive_port.min_count == 0
    assert MultiplicativeNode.additive_port.effective_default_count == 1
    assert MultiplicativeNode.impute_port.effective_default_count == 0


@pytest.mark.django_db
def test_export_marks_plain_additive_input_ports_as_multi():
    context = _make_context()
    source_a = _make_node(context, identifier='source_a')
    source_b = _make_node(context, identifier='source_b')
    target = _make_node(context, AdditiveNode, identifier='target')
    target.input_dimensions['sector'] = cast('Any', object())
    target.add_input_node(source_a)
    target.add_input_node(source_b)

    ports = _export_input_ports(target)

    assert len(ports) == 1
    assert ports[0].multi is True
    assert ports[0].required_dimensions == ['sector']
    assert target.edges[0]._to_port_ids == target.edges[1]._to_port_ids == [str(ports[0].id)]


@pytest.mark.django_db
def test_export_marks_sector_emissions_input_ports_as_multi():
    context = _make_context()
    source = _make_node(context, identifier='source')
    target = _make_node(context, SectorEmissions, identifier='target')
    target.add_input_node(source)

    ports = _export_input_ports(target)

    assert len(ports) == 1
    assert ports[0].multi is True
    assert target.edges[0]._to_port_ids == [str(ports[0].id)]


@pytest.mark.django_db
def test_export_does_not_mark_non_additive_inputs_as_multi():
    context = _make_context()
    additive_source = _make_node(context, identifier='additive_source')
    non_additive_source = _make_node(context, identifier='non_additive_source')
    target = _make_node(context, AdditiveNode, identifier='target')
    target.add_input_node(additive_source)
    target.add_input_node(non_additive_source, tags=['non_additive'])

    ports = _export_input_ports(target)

    assert len(ports) == 2
    assert [port.multi for port in ports] == [True, False]


@pytest.mark.django_db
def test_export_keeps_additive_input_ports_single_when_units_are_incompatible():
    context = _make_context()
    emissions_source = _make_node(context, identifier='emissions_source')
    energy_source = _make_node(context, identifier='energy_source', unit='MWh/a', quantity='energy')
    target = _make_node(context, AdditiveNode, identifier='target')
    target.add_input_node(emissions_source)
    target.add_input_node(energy_source)

    ports = _export_input_ports(target)

    assert [port.multi for port in ports] == [False, False]


@pytest.mark.django_db
def test_export_keeps_additive_input_ports_single_when_effective_dimensions_differ():
    context = _make_context()
    source_a = _make_node(context, identifier='source_a')
    source_b = _make_node(context, identifier='source_b')
    target = _make_node(context, AdditiveNode, identifier='target')
    target.add_input_node(source_a)
    target.add_input_node(source_b)
    target.edges[0].to_dimensions = cast('Any', {'sector': object()})
    target.edges[1].to_dimensions = cast('Any', {'building': object()})

    ports = _export_input_ports(target)

    assert [port.multi for port in ports] == [False, False]


@pytest.mark.django_db
def test_resolve_metric_columns_traces_renames_and_records_drops():
    """A metric is known by its delivered column, and a dropped one is reported separately."""
    ops: list[PortTransformOp] = [
        RenameColumnOp(column='Suorite', new_name='mileage'),
        RenameColumnOp(column='Toteutuskustannus', new_name='currency'),
        FilterColumnOp(column='Päästökerroin', drop_col=True),
    ]
    delivered, dropped = resolve_metric_columns(['Päästökerroin', 'Suorite', 'Toteutuskustannus'], ops)
    assert delivered == {'Suorite': 'mileage', 'Toteutuskustannus': 'currency'}
    assert dropped == ['Päästökerroin']


@pytest.mark.django_db
def test_resolve_metric_columns_follows_a_chained_rename():
    """A later op names the column as an earlier one left it, so renames chain."""
    ops: list[PortTransformOp] = [RenameColumnOp(column='A', new_name='B'), RenameColumnOp(column='B', new_name='C')]
    delivered, dropped = resolve_metric_columns(['A'], ops)
    assert delivered == {'A': 'C'}
    assert dropped == []


@pytest.mark.django_db
def test_pair_metrics_to_columns_pairs_through_the_bindings_renames():
    """
    The espoo-2026 regression: renamed metrics must still pair with the node's columns.

    Pairing on the raw schema names found no match and no lone leftover (three
    metrics, two columns), so every binding fell back to schema-metric-keyed
    port ids that no exported port carried — and the DB load then refused them.
    """
    ops: list[PortTransformOp] = [
        RenameColumnOp(column='Suorite', new_name='mileage'),
        RenameColumnOp(column='Toteutuskustannus', new_name='currency'),
        FilterColumnOp(column='Päästökerroin', drop_col=True),
    ]
    pairs = pair_metrics_to_columns(
        ['currency', 'mileage'],
        ['Päästökerroin', 'Suorite', 'Toteutuskustannus'],
        log_ctx='test',
        transformations=ops,
    )
    assert sorted(pairs) == [('currency', 'Toteutuskustannus'), ('mileage', 'Suorite')]


@pytest.mark.django_db
def test_pair_metrics_to_columns_without_transformations_is_unchanged():
    """Name matching and the lone-leftover rule still hold when no pipeline is given."""
    assert pair_metrics_to_columns(['Fuel'], ['fuel'], log_ctx='test') == [('Fuel', 'fuel')]
    assert pair_metrics_to_columns(['Value'], ['share'], log_ctx='test') == [('Value', 'share')]
