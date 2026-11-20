"""
Tests for how a conversion maps an instance's copies of the template's shapes to the template's own.

A YAML instance mints shape UUIDs from its own identity, so the shapes it took from a module have
other UUIDs than the template's; see `_shape_identities`.
"""

from uuid import uuid4

import pytest

from frameworks.conversion import _shape_identities
from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.defs.shape_defs import ShapeCombinationSpec, ShapeRequiredGroupSpec, ShapeSpec

pytestmark = pytest.mark.django_db


def _spec(*shapes: ShapeSpec) -> InstanceModelSpec:
    return InstanceModelSpec(
        years=YearsSpec(reference=2020, min_historical=2020, max_historical=2023, target=2030), shapes=list(shapes)
    )


def _shape(identifier: str | None, carriers: list[str], group_identifier: str | None = 'electricity') -> ShapeSpec:
    combinations = [ShapeCombinationSpec(uuid=uuid4(), categories={'energy_carrier': carrier}) for carrier in carriers]
    required = [ShapeRequiredGroupSpec(uuid=uuid4(), identifier=group_identifier, combinations=[combinations[0].uuid])]
    return ShapeSpec(
        uuid=uuid4(), identifier=identifier, dimensions=['energy_carrier'], combinations=combinations, required=required
    )


def test_a_copied_shape_and_its_parts_map_to_the_template_shape_of_the_same_identifier():
    local = _shape('end_energy', ['electricity', 'natural_gas'])
    template = _shape('end_energy', ['natural_gas', 'electricity'])

    identities = _shape_identities(_spec(local), _spec(template))

    by_carrier = {c.categories['energy_carrier']: c.uuid for c in template.combinations}
    assert identities == {
        str(local.uuid): str(template.uuid),
        str(local.combinations[0].uuid): str(by_carrier['electricity']),
        str(local.combinations[1].uuid): str(by_carrier['natural_gas']),
        str(local.required[0].uuid): str(template.required[0].uuid),
    }


def test_a_required_group_without_identifier_matches_by_its_combinations():
    local = _shape('end_energy', ['electricity'], group_identifier=None)
    template = _shape('end_energy', ['electricity'], group_identifier=None)

    identities = _shape_identities(_spec(local), _spec(template))

    assert identities[str(local.required[0].uuid)] == str(template.required[0].uuid)


def test_a_shape_the_template_does_not_have_is_left_alone():
    local = _shape('own_survey', ['electricity'])

    assert _shape_identities(_spec(local), _spec(_shape('end_energy', ['electricity']))) == {}


def test_a_combination_the_template_lacks_keeps_its_identity():
    local = _shape('end_energy', ['electricity', 'hydrogen'])
    template = _shape('end_energy', ['electricity'])

    identities = _shape_identities(_spec(local), _spec(template))

    assert str(local.combinations[1].uuid) not in identities
