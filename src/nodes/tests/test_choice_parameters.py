"""Choice parameters, and choosing a category of a dimension by one (`select_category`)."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import polars as pl
import pytest

from common import polars as ppl
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.instance_loader import InstanceLoader
from nodes.instance_parser import InstanceParseError, parse_instance_snapshot
from nodes.units import unit_registry
from params import ChoiceParameter, DimensionCategoryParameter
from params.param import ValidationError

pytestmark = pytest.mark.django_db


def _config(**extra: Any) -> dict[str, Any]:
    return {
        'id': 'choice_test',
        'default_language': 'en',
        'supported_languages': [],
        'name': 'Choice test',
        'owner': 'Owner',
        'target_year': 2030,
        'model_end_year': 2030,
        'minimum_historical_year': 2020,
        'maximum_historical_year': 2020,
        'reference_year': 2020,
        'dimensions': [
            {
                'id': 'variant',
                'label': 'Implementation variant',
                'categories': [{'id': 'ambitious', 'label': 'Ambitious'}, {'id': 'conservative', 'label': 'Conservative'}],
            },
        ],
        'params': [
            {'id': 'knsv_variant', 'type': 'dimension_category', 'dimension': 'variant', 'value': 'ambitious'},
            {
                'id': 'reading',
                'type': 'choice',
                'value': 'by_2045',
                'choices': [{'id': 'by_2045', 'label_en': 'By 2045'}, {'id': 'by_2035', 'label_en': 'By 2035'}],
            },
        ],
        'nodes': [
            {
                'id': 'shares',
                'type': 'formula.FormulaNode',
                'name': 'Shares',
                'unit': 'dimensionless',
                'quantity': 'fraction',
                'params': [{'id': 'formula', 'value': 'select_category(shares_by_variant, variant=knsv_variant)'}],
            },
        ],
        **extra,
    }


def _context(config: dict[str, Any]):
    return InstanceLoader(snapshot=parse_instance_snapshot(config, instance_uuid=uuid4())).context


def _frame() -> ppl.PathsDataFrame:
    df = pl.DataFrame({
        YEAR_COLUMN: [2025, 2025],
        'variant': ['ambitious', 'conservative'],
        VALUE_COLUMN: [0.5, 0.7],
        FORECAST_COLUMN: [True, True],
    })
    return ppl.to_ppdf(
        df,
        meta=ppl.DataFrameMeta(
            units={VALUE_COLUMN: unit_registry.parse_units('dimensionless')}, primary_keys=[YEAR_COLUMN, 'variant']
        ),
    )


def test_instance_defined_choice_parameters() -> None:
    ctx = _context(_config())
    variant = ctx.global_parameters['knsv_variant']
    assert isinstance(variant, DimensionCategoryParameter)
    assert [(choice.id, str(choice.label)) for choice in variant.get_choices() or []] == [
        ('ambitious', 'Ambitious'),
        ('conservative', 'Conservative'),
    ]
    reading = ctx.global_parameters['reading']
    assert isinstance(reading, ChoiceParameter)
    assert [str(choice.label) for choice in reading.get_choices() or []] == ['By 2045', 'By 2035']
    with pytest.raises(ValidationError, match='is not one of'):
        variant.set('neither')
    with pytest.raises(ValidationError, match='is not one of'):
        reading.set('by_2030')


def test_a_default_outside_the_dimension_is_rejected() -> None:
    config = _config()
    config['params'][0]['value'] = 'neither'
    with pytest.raises(ValidationError, match='is not one of'):
        _context(config)


def test_a_global_parameter_needs_a_type_unless_code_defines_it() -> None:
    config = _config()
    del config['params'][0]['type']
    with pytest.raises(InstanceParseError, match='give its `type`'):
        parse_instance_snapshot(config, instance_uuid=uuid4())


def test_select_category_follows_the_parameter_and_the_node_depends_on_it() -> None:
    ctx = _context(_config())
    node = ctx.get_node('shares')
    param = ctx.global_parameters['knsv_variant']
    assert 'knsv_variant' in node.global_parameters
    assert node in param._subscription_nodes
    with ctx.run():
        varss = node._collect_eval_vars()
        varss.datasets['shares_by_variant'] = _frame()
        formula = 'select_category(shares_by_variant, variant=knsv_variant)'
        assert node.evaluate_formula(formula, varss)[VALUE_COLUMN].to_list() == [0.5]
        param.set('conservative')
        df = node.evaluate_formula(formula, varss)
        assert df[VALUE_COLUMN].to_list() == [0.7]
        assert 'variant' not in df.dim_ids
        fixed = node.evaluate_formula("select_category(shares_by_variant, variant='ambitious')", varss)
        assert fixed[VALUE_COLUMN].to_list() == [0.5]
