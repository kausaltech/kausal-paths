from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest

from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.instance_loader import InstanceYAMLConfig
from nodes.instance_parser import InstanceParseError, parse_instance_snapshot
from nodes.shapes import resolve_shapes
from nodes.template_spec import compose_instance_spec, compose_shapes, ensure_instance_shapes

if TYPE_CHECKING:
    from nodes.defs.shape_defs import ShapeSpec

pytestmark = pytest.mark.django_db

DIMENSIONS = [
    {
        'id': 'sector',
        'label': 'Sector',
        'categories': [{'id': 'households', 'label': 'Households'}, {'id': 'industry', 'label': 'Industry'}],
    },
    {
        'id': 'energy_carrier',
        'label': 'Energy carrier',
        'categories': [
            {'id': 'electricity', 'label': 'Electricity'},
            {'id': 'natural_gas', 'label': 'Natural gas'},
            {'id': 'hydrogen', 'label': 'Hydrogen'},
        ],
    },
]

STANDARD: dict[str, Any] = {
    'id': 'std/end_energy',
    'name': 'End energy',
    'dimensions': ['sector', 'energy_carrier'],
    'combinations': [
        {'id': 'households_electricity', 'categories': {'sector': 'households', 'energy_carrier': 'electricity'}},
        {'id': 'households_gas', 'categories': {'sector': 'households', 'energy_carrier': 'natural_gas'}},
    ],
    'required': [
        {
            'id': 'households_any',
            'combinations': ['households_electricity', 'households_gas'],
            'qualifiers': {'quality.coverage': {'min': 1}},
        }
    ],
}
EXTENSION_POINT: dict[str, Any] = {'id': 'end_energy', 'inherits': ['std/end_energy'], 'closed': True, 'owner': 'instance'}


def _spec(shapes: list[dict[str, Any]]) -> InstanceModelSpec:
    snapshot = parse_instance_snapshot(
        {
            'id': 'test',
            'default_language': 'en',
            'name': 'Test',
            'owner': 'Owner',
            'target_year': 2030,
            'reference_year': 2020,
            'minimum_historical_year': 2010,
            'dimensions': DIMENSIONS,
            'shapes': shapes,
        },
        instance_uuid=uuid4(),
    )
    return snapshot.spec


def test_inheritance_unions_combinations_and_keeps_their_origin() -> None:
    child = {
        'id': 'local',
        'inherits': ['std/end_energy'],
        'combinations': [
            {'id': 'industry_hydrogen', 'categories': {'sector': 'industry', 'energy_carrier': 'hydrogen'}},
            # Already in the standard: the standard's entry wins, the local one is reported redundant.
            {'id': 'also_households_gas', 'categories': {'sector': 'households', 'energy_carrier': 'natural_gas'}},
        ],
    }
    spec = _spec([STANDARD, child])
    standard, local = spec.shapes
    effective = resolve_shapes(spec.shapes)[local.uuid]

    assert effective.dimensions == ('sector', 'energy_carrier')
    assert [(c.identifier, c.origin) for c in effective.combinations] == [
        ('households_electricity', standard.uuid),
        ('households_gas', standard.uuid),
        ('industry_hydrogen', local.uuid),
    ]
    assert effective.redundant == (local.combinations[1].uuid,)
    (group,) = effective.required
    assert group.origin == standard.uuid
    assert group.qualifiers['quality.coverage'].min == 1
    assert not effective.closed


@pytest.mark.parametrize(
    ('shapes', 'message'),
    [
        ([{**STANDARD, 'inherits': ['missing']}], 'unknown shape missing'),
        ([{**STANDARD, 'id': 'a', 'inherits': ['b']}, {**STANDARD, 'id': 'b', 'inherits': ['a']}], 'cycle'),
        (
            [{**STANDARD, 'combinations': [{'id': 'x', 'categories': {'sector': 'households'}}], 'required': []}],
            'must name exactly',
        ),
        (
            [
                {
                    **STANDARD,
                    'combinations': [{'id': 'x', 'categories': {'sector': 'farms', 'energy_carrier': 'hydrogen'}}],
                    'required': [],
                }
            ],
            'has no category farms',
        ),
        ([{**STANDARD, 'required': [{'id': 'r', 'combinations': ['nope']}]}], 'unknown combination nope'),
        ([STANDARD, {'id': 'other', 'inherits': ['std/end_energy'], 'dimensions': ['sector']}], 'constrains'),
    ],
)
def test_inconsistent_shapes_fail_at_parse_time(shapes: list[dict[str, Any]], message: str) -> None:
    with pytest.raises((InstanceParseError, ValueError), match=message):
        _spec(shapes)


def test_spec_without_shapes_serializes_as_before() -> None:
    # Frozen template revisions predate shapes; their content hash must not change.
    spec = InstanceModelSpec(years=YearsSpec(reference=2020, min_historical=2010, target=2030, model_end=2030))
    assert 'shapes' not in spec.model_dump(mode='json')
    with_shapes = _spec([STANDARD])
    dumped = with_shapes.model_dump(mode='json')
    assert InstanceModelSpec.model_validate(dumped).model_dump(mode='json')['shapes'] == dumped['shapes']


def _template_shapes() -> list[ShapeSpec]:
    return _spec([STANDARD, EXTENSION_POINT]).shapes


def test_extension_point_record_replaces_the_template_declaration() -> None:
    template = _template_shapes()
    standard, point = template
    hydrogen = _spec([
        {
            'id': 'h',
            'dimensions': ['sector', 'energy_carrier'],
            'combinations': [{'id': 'industry_hydrogen', 'categories': {'sector': 'industry', 'energy_carrier': 'hydrogen'}}],
        }
    ]).shapes[0]
    record = point.model_copy(update={'combinations': hydrogen.combinations})
    composed = compose_shapes(template, [record])

    assert [shape.uuid for shape in composed] == [standard.uuid, point.uuid]
    effective = resolve_shapes(composed)[point.uuid]
    assert {c.identifier for c in effective.combinations} == {'households_electricity', 'households_gas', 'industry_hydrogen'}
    assert effective.closed


def test_framework_owned_shapes_cannot_be_redefined_locally() -> None:
    template = _template_shapes()
    with pytest.raises(ValueError, match='belongs to the template'):
        compose_shapes(template, [template[0].model_copy(update={'combinations': []})])


@pytest.mark.parametrize('change', [{'closed': False}, {'inherits': []}, {'identifier': 'renamed'}])
def test_extension_point_record_keeps_the_fixed_fields(change: dict[str, Any]) -> None:
    template = _template_shapes()
    with pytest.raises(ValueError, match='must keep'):
        compose_shapes(template, [template[1].model_copy(update=change)])


def test_dependents_get_their_own_record_of_each_extension_point() -> None:
    base = InstanceModelSpec(
        years=YearsSpec(reference=2020, min_historical=2010, target=2030, model_end=2030), shapes=_template_shapes()
    )
    local = InstanceModelSpec(years=base.years)

    assert ensure_instance_shapes(local, base) == ['end_energy']
    assert [shape.uuid for shape in local.shapes] == [base.shapes[1].uuid]
    assert local.shapes[0].inherits == base.shapes[1].inherits
    assert ensure_instance_shapes(local, base) == []


def test_module_shapes_merge_from_includes(tmp_path) -> None:
    (tmp_path / 'module.yaml').write_text("""
shapes:
- id: std/end_energy
  dimensions: [sector]
  combinations:
  - {id: households, categories: {sector: households}}
""")
    (tmp_path / 'test.yaml').write_text("""
id: test
default_language: en
name: Test
owner: Owner
target_year: 2030
reference_year: 2020
minimum_historical_year: 2010
include:
- file: module.yaml
shapes:
- id: std/end_energy
  dimensions: [sector]
""")
    with pytest.raises(ValueError, match='already declared'):
        InstanceYAMLConfig.load_for_entrypoint(tmp_path / 'test.yaml')


def test_instance_spec_composition_uses_the_local_record() -> None:
    years = YearsSpec(reference=2020, min_historical=2010, target=2030, model_end=2030)
    base = InstanceModelSpec(years=years, shapes=_template_shapes())
    local = InstanceModelSpec(years=years)
    ensure_instance_shapes(local, base)
    hydrogen = _spec([
        {
            'id': 'h',
            'dimensions': ['sector', 'energy_carrier'],
            'combinations': [{'id': 'industry_hydrogen', 'categories': {'sector': 'industry', 'energy_carrier': 'hydrogen'}}],
        }
    ]).shapes[0]
    local.shapes[0] = local.shapes[0].model_copy(update={'combinations': hydrogen.combinations})

    composed = compose_instance_spec(base, local, [], errors=[])

    point = composed.shapes[1]
    assert point.uuid == base.shapes[1].uuid
    assert [c.identifier for c in point.combinations] == ['industry_hydrogen']
    assert base.shapes[1].combinations == []
