"""
``interpolate`` and ``extend``: how a dataset binding says what shape its data should take.

Both are properties of the *binding*, not of the consuming node class, so that a dataset
means the same thing wherever it is bound. The one class-dependent part is the default for
``interpolate``: the rebuilt classes interpolate unless told otherwise, the historical ones
only when asked. See ``docs/plans/additive-multiplicative-modernization.md``.
"""

from uuid import uuid4

import pytest

from nodes.instance_parser import parse_instance_snapshot

pytestmark = pytest.mark.django_db


def _parse(nodes: list[dict]):
    return parse_instance_snapshot(
        {
            'id': 'flags',
            'default_language': 'en',
            'name': 'Flags',
            'owner': 'Owner',
            'target_year': 2030,
            'reference_year': 2020,
            'minimum_historical_year': 2010,
            'nodes': nodes,
        },
        instance_uuid=uuid4(),
    )


def _snapshot(node: dict) -> dict:
    """Parse a one-node instance and return its dataset port snapshots by dataset id."""
    snapshot = _parse([{'name': 'Node', 'unit': 'kg/a', 'quantity': 'mass', **node}])
    return {port.dataset: port for port in snapshot.dataset_bindings}


def test_historical_classes_do_not_interpolate_unless_asked():
    ports = _snapshot({'id': 'n', 'type': 'simple.AdditiveNode', 'input_datasets': ['some/data']})
    assert ports['some/data'].spec.interpolate is False


def test_historical_classes_interpolate_with_the_processor_entry():
    ports = _snapshot(
        {
            'id': 'n',
            'type': 'simple.AdditiveNode',
            'input_datasets': ['some/data'],
            'input_dataset_processors': ['LinearInterpolation'],
        },
    )
    assert ports['some/data'].spec.interpolate is True


def test_rebuilt_classes_interpolate_by_default():
    for node_type in ('simple.AdditiveNode2', 'simple.MultiplicativeNode2'):
        ports = _snapshot({'id': 'n', 'type': node_type, 'input_datasets': ['some/data']})
        assert ports['some/data'].spec.interpolate is True, node_type


def test_a_binding_can_opt_out_of_the_class_default():
    ports = _snapshot(
        {
            'id': 'n',
            'type': 'simple.AdditiveNode2',
            'input_datasets': [{'id': 'some/data', 'interpolate': False}],
        },
    )
    assert ports['some/data'].spec.interpolate is False


def test_the_processor_entry_still_forces_interpolation_on_every_binding():
    """Unlike a class default, the legacy processor is not a default — it overrides."""
    ports = _snapshot(
        {
            'id': 'n',
            'type': 'simple.AdditiveNode',
            'input_datasets': [{'id': 'some/data', 'interpolate': False}],
            'input_dataset_processors': ['LinearInterpolation'],
        },
    )
    assert ports['some/data'].spec.interpolate is True


def test_a_class_default_does_not_manufacture_a_processor_entry():
    """
    The processor list echoes what was authored; a class default was not authored.

    The export path derives the same list from the runtime, so if a class default showed up
    here the two representations of one model would disagree.
    """
    snapshot = _parse([
        {
            'id': 'n',
            'type': 'simple.AdditiveNode2',
            'name': 'Node',
            'unit': 'kg/a',
            'quantity': 'mass',
            'input_datasets': ['some/data'],
        },
    ])
    parsed = next(n for n in snapshot.nodes if n.identifier == 'n')
    assert parsed.spec.extra.input_dataset_processors == []


def test_extend_defaults_off_and_is_carried_through_the_binding():
    off = _snapshot({'id': 'n', 'type': 'simple.AdditiveNode2', 'input_datasets': ['some/data']})
    assert off['some/data'].spec.extend is False

    on = _snapshot(
        {
            'id': 'n',
            'type': 'simple.AdditiveNode2',
            'input_datasets': [{'id': 'some/data', 'extend': True}],
        },
    )
    assert on['some/data'].spec.extend is True
