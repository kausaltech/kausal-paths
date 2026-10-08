"""Consumer requirements must detect absent cells and years independently of input source."""

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import polars as pl
import pytest

from kausal_common.datasets.tests.factories import DataPointFactory, DatasetFactory, DatasetMetricFactory

from common import qualifiers
from common.polars import DataFrameMeta, to_ppdf
from nodes.constants import VALUE_COLUMN, YEAR_COLUMN
from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.defs.node_defs import NodeSpec, SimpleConfig
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.defs.shape_defs import ShapeCombinationSpec, ShapeRequiredGroupSpec, ShapeSpec
from nodes.instance_loader import InstanceLoader
from nodes.instance_parser import InstanceParseError, parse_instance_snapshot
from nodes.models import NodeInputPortBinding
from nodes.shapes import resolve_shapes
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory
from nodes.units import unit_registry
from nodes.value_validation import (
    InstanceValueValidationError,
    QualifierRequirement,
    ValueContract,
    collect_instance_value_violations,
    validate_value_contract,
)

if TYPE_CHECKING:
    from common.polars import PathsDataFrame
    from nodes.shapes import EffectiveShape

pytestmark = pytest.mark.django_db


def _frame(rows: list[tuple[int, str, float | None]]) -> PathsDataFrame:
    return to_ppdf(
        pl.DataFrame(
            {YEAR_COLUMN: [r[0] for r in rows], 'carrier': [r[1] for r in rows], VALUE_COLUMN: [r[2] for r in rows]},
            schema={YEAR_COLUMN: pl.Int64, 'carrier': pl.String, VALUE_COLUMN: pl.Float64},
        ),
        meta=DataFrameMeta(primary_keys=[YEAR_COLUMN, 'carrier'], units={VALUE_COLUMN: unit_registry.parse_units('kWh')}),
    )


def _shape(
    *required: str | tuple[str, ...],
    allowed: tuple[str, ...] = (),
    closed: bool = False,
    qualifiers: dict[str, QualifierRequirement] | None = None,
) -> EffectiveShape:
    """Build a carrier shape; each required entry is a group, and a tuple offers alternatives."""
    groups = [(entry,) if isinstance(entry, str) else entry for entry in required]
    carriers = list(dict.fromkeys([*allowed, *(carrier for group in groups for carrier in group)]))
    combinations = {
        carrier: ShapeCombinationSpec(uuid=uuid4(), identifier=carrier, categories={'carrier': carrier}) for carrier in carriers
    }
    spec = ShapeSpec(
        uuid=uuid4(),
        identifier='carriers',
        dimensions=['carrier'],
        closed=closed,
        combinations=list(combinations.values()),
        required=[
            ShapeRequiredGroupSpec(
                uuid=uuid4(),
                identifier='_'.join(group),
                combinations=[combinations[carrier].uuid for carrier in group],
                qualifiers=qualifiers or {},
            )
            for group in groups
        ],
    )
    return resolve_shapes([spec])[spec.uuid]


def _problems(
    frame: PathsDataFrame, contract: ValueContract, shape: EffectiveShape | None = None
) -> list[tuple[str, list[int], dict[str, str]]]:
    if shape is not None:
        contract = contract.model_copy(update={'shape': shape.uuid})
    return [
        (p.code, p.years, p.categories)
        for p in validate_value_contract(frame, contract, [2020, 2021], node_uuid=uuid4(), port_uuid=uuid4(), shape=shape)
    ]


@pytest.mark.parametrize('missing', [None, float('nan'), float('inf')])
def test_null_and_nonfinite_cells_cannot_satisfy_requirements(missing: float | None) -> None:
    assert _problems(_frame([(2020, 'gas', missing), (2021, 'gas', 0.0)]), ValueContract(), _shape('gas')) == [
        ('missing_required_value', [2020], {'carrier': 'gas'})
    ]


def test_missing_combinations_and_entire_years_are_detected_after_null_dropping() -> None:
    shape = _shape('gas', 'electricity')
    assert _problems(_frame([(2020, 'gas', 0.0)]), ValueContract(), shape) == [
        ('missing_required_value', [2021], {'carrier': 'gas'}),
        ('missing_required_value', [2020], {'carrier': 'electricity'}),
        ('missing_required_value', [2021], {'carrier': 'electricity'}),
    ]
    assert len(_problems(_frame([]), ValueContract(), shape)) == 4


def test_optional_route_is_checked_only_in_reported_years() -> None:
    frame = _frame([(2020, 'gas', 3.0), (2021, 'gas', 0.0)]).with_columns(
        qualifiers.make(reported=pl.col(YEAR_COLUMN) == 2020).alias('Value__qual')
    )
    assert _problems(frame, ValueContract(years='active'), _shape('gas', 'electricity')) == [
        ('missing_required_value', [2020], {'carrier': 'electricity'})
    ]


def test_zero_activity_requires_a_grade_but_does_not_turn_grade_d_into_missing() -> None:
    catalog = qualifiers.QualifierCatalog((
        *qualifiers.BUILTIN_QUALIFIERS.definitions,
        qualifiers.QualifierDefinition('assessment', qualifiers.Propagation.COVERED_SCORE),
    ))
    frame = _frame([(2020, 'gas', 0.0), (2021, 'gas', 0.0)]).with_columns(
        qualifiers.make(
            catalog=catalog,
            reported=pl.lit(value=True),
            assessments={'assessment': qualifiers.covered_score(pl.lit(0.0), pl.lit(1.0))},
        ).alias('Value__qual')
    )
    coverage = _shape('gas', qualifiers={'assessment.coverage': QualifierRequirement(min=1)})
    assert _problems(frame, ValueContract(), coverage) == []
    score = _shape('gas', qualifiers={'assessment.score': QualifierRequirement(min=1)})
    assert [p[0] for p in _problems(frame, ValueContract(), score)] == ['required_qualifier', 'required_qualifier']


def _computed_binding_config() -> dict[str, Any]:
    return {
        'id': 'value-contract',
        'name': 'Value contract',
        'owner': 'Test',
        'default_language': 'en',
        'target_year': 2030,
        'reference_year': 2020,
        'minimum_historical_year': 2020,
        'maximum_historical_year': 2021,
        'dimensions': [
            {
                'id': 'carrier',
                'label': 'Carrier',
                'categories': [{'id': 'gas', 'label': 'Gas'}, {'id': 'electricity', 'label': 'Electricity'}],
            }
        ],
        'nodes': [
            {
                'id': 'computed',
                'name': 'Computed',
                'type': 'simple.AdditiveNode',
                'quantity': 'energy',
                'unit': 'kWh',
                'historical_values': [[2020, 0.0]],
                'output_nodes': [{'id': 'consumer', 'to_dimensions': [{'id': 'carrier', 'categories': ['gas']}]}],
            },
            {
                'id': 'consumer',
                'name': 'Consumer',
                'type': 'simple.AdditiveNode',
                'quantity': 'energy',
                'unit': 'kWh',
                'input_dimensions': ['carrier'],
                'output_dimensions': ['carrier'],
                'input_validation': {'computed': {'shape': 'carriers'}},
            },
        ],
        'shapes': [
            {
                'id': 'carriers',
                'dimensions': ['carrier'],
                'combinations': [
                    {'id': 'gas', 'categories': {'carrier': 'gas'}},
                    {'id': 'electricity', 'categories': {'carrier': 'electricity'}},
                ],
                'required': [{'id': 'gas', 'combinations': ['gas']}, {'id': 'electricity', 'combinations': ['electricity']}],
            }
        ],
    }


def test_computed_binding_is_validated_after_its_dimension_transformation() -> None:
    snapshot = parse_instance_snapshot(_computed_binding_config(), instance_uuid=uuid4())
    loader = InstanceLoader(snapshot=snapshot)
    consumer = loader.context.get_node('consumer')
    assert len(consumer.runtime_input_bindings) == 1
    with loader.context.run():
        problems = collect_instance_value_violations(loader.instance, undeclared='evaluate')
    assert {(p.code, tuple(p.years), p.categories['carrier']) for p in problems} == {
        ('missing_required_value', (2020,), 'electricity'),
        ('missing_required_value', (2021,), 'electricity'),
    }
    assert consumer.runtime_node_meta is not None
    assert all(p.node_uuid == consumer.runtime_node_meta.id for p in problems)


def test_undeclared_calendar_is_reported_once_but_evaluated_for_the_model() -> None:
    loader = InstanceLoader(snapshot=parse_instance_snapshot(_computed_binding_config(), instance_uuid=uuid4()))
    with loader.context.run():
        reported = collect_instance_value_violations(loader.instance)
        evaluated = collect_instance_value_violations(loader.instance, undeclared='evaluate')
    assert [(p.code, p.years, p.enforcement) for p in reported] == [
        ('inventory_years_undeclared', [2020, 2021], 'block_submission')
    ]
    assert {p.code for p in evaluated} == {'missing_required_value'}


def test_tier_follows_what_the_contract_guards() -> None:
    frame = _frame([(2020, 'gas', -1.0)])
    oil = _shape('oil')
    static = ValueContract(min=0, shape=oil.uuid)
    assert [
        (p.code, p.enforcement)
        for p in validate_value_contract(frame, static, [2020], node_uuid=uuid4(), port_uuid=uuid4(), shape=oil)
    ] == [
        ('missing_required_value', 'block_submission'),
        ('value_range', 'block_publish'),
    ]
    # A factor missing for reported activity is a wrong result, not an incomplete one.
    factors = ValueContract(combinations_from_positive=uuid4())
    problems = validate_value_contract(
        _frame([]), factors, [2020], node_uuid=uuid4(), port_uuid=uuid4(), required_values=_frame([(2020, 'gas', 1.0)])
    )
    assert [(p.code, p.enforcement) for p in problems] == [('missing_required_value', 'block_publish')]


def test_a_conditional_contract_can_declare_that_it_only_blocks_submission() -> None:
    """Heat output missing for plant fuel input sends the year to a fallback route; the result is not wrong."""
    gas = _shape('gas')
    contract = ValueContract(required_if_positive=uuid4(), enforcement='block_submission', shape=gas.uuid)
    problems = validate_value_contract(_frame([]), contract, [2020], node_uuid=uuid4(), port_uuid=uuid4(), shape=gas)
    assert [(p.code, p.enforcement) for p in problems] == [('missing_required_value', 'block_submission')]


@pytest.mark.parametrize('value', [None, Decimal(0)])
def test_publication_validates_delivered_dataset_values_atomically(value: Decimal | None) -> None:
    instance = InstanceFactory.create()
    config = InstanceConfigFactory.create(
        identifier=instance.id,
        instance=instance,
        config_source='database',
        owner='Test',
        spec=InstanceModelSpec(
            years=YearsSpec(reference=2020, min_historical=2020, max_historical=2020, target=2030, skipped=[])
        ),
    )
    assert config.spec is not None
    config.spec.features.use_datasets_from_db = True
    config.save(update_fields=['spec'])
    dataset = DatasetFactory.create(identifier='activity', scope=config)
    metric = DatasetMetricFactory.create(schema=dataset.schema, name='value', label='Value', unit='t/a')
    DataPointFactory.create(dataset=dataset, metric=metric, date=date(2020, 1, 1), value=value)
    port_id = uuid4()
    unit = unit_registry.parse_units('t/a')
    node = NodeConfigFactory.create(
        instance=config,
        identifier='consumer',
        spec=NodeSpec(
            type_config=SimpleConfig(node_class='nodes.simple.SimpleNode'),
            input_ports=[
                InputPortDef(
                    id=port_id,
                    unit=unit,
                    quantity='emissions',
                    validation=ValueContract(required=True),
                )
            ],
            output_ports=[OutputPortDef(id=uuid4(), unit=unit, quantity='emissions')],
        ),
    )
    NodeInputPortBinding.objects.create(instance=config, node=node, port_id=port_id, dataset=dataset, metric=metric)
    if value is None:
        with pytest.raises(InstanceValueValidationError) as error:
            config.publish_instance(require_submittable=True)
        assert [(p.code, p.enforcement) for p in error.value.violations] == [('missing_required_value', 'block_submission')]
        config.refresh_from_db()
        assert config.live_revision_id is None
        dataset.refresh_from_db()
        assert dataset.latest_revision_id is None
        # An incomplete inventory is still a correct draft.
        config.publish_instance()
        config.refresh_from_db()
        assert config.live_revision_id is not None
    else:
        config.publish_instance()
        config.refresh_from_db()
        assert config.live_revision_id is not None


@pytest.mark.parametrize('consumption', [0.0, 1.0])
def test_factor_is_required_only_in_years_with_positive_consumption(consumption: float) -> None:
    config = {
        'id': 'conditional-factor',
        'name': 'Conditional factor',
        'owner': 'Test',
        'default_language': 'en',
        'target_year': 2030,
        'reference_year': 2020,
        'minimum_historical_year': 2020,
        'maximum_historical_year': 2021,
        'nodes': [
            {
                'id': 'activity',
                'name': 'Activity',
                'type': 'simple.AdditiveNode',
                'quantity': 'energy',
                'unit': 'kWh',
                'historical_values': [[2020, consumption]],
                'output_nodes': [{'id': 'consumer'}],
            },
            {
                'id': 'factor',
                'name': 'Factor',
                'type': 'simple.AdditiveNode',
                'quantity': 'fraction',
                'unit': 'dimensionless',
                'historical_values': [[2021, 1.0]],
                'output_nodes': [{'id': 'consumer'}],
            },
            {
                'id': 'consumer',
                'name': 'Consumer',
                'type': 'formula.FormulaNode',
                'quantity': 'energy',
                'unit': 'kWh',
                'params': {'formula': 'activity * factor'},
                'input_validation': {'factor': {'required_if_positive': 'activity', 'required': True}},
            },
        ],
    }
    loader = InstanceLoader(snapshot=parse_instance_snapshot(config, instance_uuid=uuid4()))
    problems = collect_instance_value_violations(loader.instance)
    assert [(p.code, p.years) for p in problems] == ([('missing_required_value', [2020])] if consumption else [])


def test_factors_are_required_for_every_positive_activity_cell() -> None:
    contract = ValueContract(min=0, combinations_from_positive=uuid4())
    factors = _frame([(2020, 'gas', 0.3)])
    activity = _frame([(2020, 'gas', 10.0), (2020, 'oil', 3.0), (2021, 'oil', 0.0)])
    problems = validate_value_contract(
        factors, contract, [2020, 2021], node_uuid=uuid4(), port_uuid=uuid4(), required_values=activity
    )
    assert [(p.code, p.years, p.categories) for p in problems] == [('missing_required_value', [2020], {'carrier': 'oil'})]


def test_variant_contract_rejects_two_simultaneously_declared_variants() -> None:
    contract = ValueContract(years='active', required=True, min=1, max=1, max_rows=1)
    assert _problems(_frame([(2020, 'gas', 1.0), (2020, 'electricity', 1.0)]), contract) == [('row_count', [2020], {})]


def test_bounds_reject_nonfinite_values_even_without_static_combination_requirements() -> None:
    assert _problems(_frame([(2020, 'gas', float('inf'))]), ValueContract(min=0)) == [('value_range', [2020], {})]


def test_a_group_is_satisfied_by_any_of_its_combinations() -> None:
    shape = _shape(('gas', 'electricity'))
    assert _problems(_frame([(2020, 'electricity', 1.0), (2021, 'gas', 0.0)]), ValueContract(), shape) == []
    assert _problems(_frame([(2020, 'electricity', 1.0)]), ValueContract(), shape) == [('missing_required_value', [2021], {})]


def test_a_group_qualifier_holds_for_all_of_its_values() -> None:
    catalog = qualifiers.QualifierCatalog((
        *qualifiers.BUILTIN_QUALIFIERS.definitions,
        qualifiers.QualifierDefinition('assessment', qualifiers.Propagation.COVERED_SCORE),
    ))
    frame = _frame([(2020, 'gas', 1.0), (2020, 'electricity', 1.0)]).with_columns(
        qualifiers.make(
            catalog=catalog,
            reported=pl.lit(value=True),
            assessments={
                'assessment': qualifiers.covered_score(
                    pl.when(pl.col('carrier') == 'gas').then(pl.lit(1.0)).otherwise(pl.lit(0.0)), pl.lit(1.0)
                )
            },
        ).alias('Value__qual')
    )
    graded = _shape(('gas', 'electricity'), qualifiers={'assessment.score': QualifierRequirement(min=1)})
    # Gas meets the grade, electricity does not: one well-graded member does not carry the group.
    assert [p[0] for p in _problems(frame, ValueContract(years='active'), graded)] == ['required_qualifier']
    assert _problems(frame.filter(pl.col('carrier') == 'gas'), ValueContract(years='active'), graded) == []


def test_a_closed_shape_reports_values_outside_it() -> None:
    shape = _shape('gas', allowed=('electricity',), closed=True)
    frame = _frame([(2020, 'gas', 1.0), (2021, 'gas', 1.0), (2020, 'hydrogen', 1.0), (2021, 'hydrogen', None)])
    assert _problems(frame, ValueContract(), shape) == [('outside_shape', [2020], {'carrier': 'hydrogen'})]
    # An open shape is a minimum: the same values pass.
    assert _problems(frame, ValueContract(), _shape('gas', allowed=('electricity',))) == []


def test_an_input_contract_must_name_a_declared_shape() -> None:
    config = _computed_binding_config()
    config['nodes'][1]['input_validation'] = {'computed': {'shape': 'missing'}}
    with pytest.raises(InstanceParseError, match='unknown shape missing'):
        parse_instance_snapshot(config, instance_uuid=uuid4())
