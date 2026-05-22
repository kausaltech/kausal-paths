from __future__ import annotations

from typing import Any, cast

import pytest

from kausal_common.i18n.pydantic import TranslatedString

from nodes.node import Node
from nodes.simple import AdditiveNode, SectorEmissions
from nodes.spec_export import _export_input_ports
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
def test_export_marks_plain_additive_input_ports_as_multi():
    context = _make_context()
    source_a = _make_node(context, identifier='source_a')
    source_b = _make_node(context, identifier='source_b')
    target = _make_node(context, AdditiveNode, identifier='target')
    target.add_input_node(source_a)
    target.add_input_node(source_b)

    ports = _export_input_ports(target)

    assert len(ports) == 1
    assert ports[0].multi is True
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
