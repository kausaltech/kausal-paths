"""
The behavioural contract of ``AdditiveNode`` and ``MultiplicativeNode``.

Every test here pins down one rule about how the two classes combine their inputs:
how ragged years are reconciled, what happens to nulls and NaNs, how dimensions
and units have to line up, whether input order matters, and what the classes do
with an input *dataset* as opposed to an input *node*.

Where ``GenericNode`` is expected to compute the same thing (its ``add`` and
``multiply`` operations are the modern implementation of the same two ideas), the
test asserts the two agree. Where they do **not** agree today, the test says so
explicitly and names the divergence — those are the decisions to settle before
``AdditiveNode``/``MultiplicativeNode`` are rebuilt on the shared machinery.

See ``docs/plans/additive-multiplicative-modernization.md``.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

from kausal_common.i18n.pydantic import TranslatedString

from common.polars import DataFrameMeta, to_ppdf
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.datasets import Dataset
from nodes.dimensions import Dimension, DimensionCategory
from nodes.edges import Edge
from nodes.exceptions import NodeError
from nodes.generic import GenericNode
from nodes.node import Node, NodeStatus
from nodes.simple import AdditiveNode, AdditiveNode2, MultiplicativeNode, MultiplicativeNode2
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory
from nodes.units import unit_registry
from params.param import BoolParameter, StringParameter

if TYPE_CHECKING:
    from common.polars import PathsDataFrame
    from nodes.context import Context

pytestmark = pytest.mark.django_db

NAN = float('nan')


# --- test doubles ----------------------------------------------------------------


class _FixedOutputNode(Node):
    """A leaf node whose output is a fixed, caller-supplied PathsDataFrame."""

    def __init__(self, *args, fixed_df: PathsDataFrame, **kwargs):
        super().__init__(*args, **kwargs)
        self._fixed_df = fixed_df

    def compute(self) -> PathsDataFrame:
        return self._fixed_df


@dataclass
class _FixedDataset(Dataset):
    """A dataset returning a caller-supplied PathsDataFrame, through the real post-processing."""

    fixed_df: PathsDataFrame | None = None

    def load_internal(self) -> PathsDataFrame:
        assert self.fixed_df is not None
        # Real datasets run post_process, which is where `interpolate` and `extend` apply.
        return self.post_process(self.fixed_df)

    def hash_data(self) -> dict[str, Any]:
        return {'id': self.id}


# --- builders --------------------------------------------------------------------


def _make_context(identifier: str) -> Context:
    instance = InstanceFactory.create(id=identifier, name=identifier)
    InstanceConfigFactory.create(identifier=instance.id, instance=instance, name=identifier)
    ctx = instance.context
    ctx.dimensions['sector'] = Dimension(
        id='sector',
        label=TranslatedString('Sector', default_language='en'),
        categories=[
            DimensionCategory(id='x', label=TranslatedString('X', default_language='en')),
            DimensionCategory(id='y', label=TranslatedString('Y', default_language='en')),
        ],
    )
    ctx.dimensions['fuel'] = Dimension(
        id='fuel',
        label=TranslatedString('Fuel', default_language='en'),
        categories=[
            DimensionCategory(id='gas', label=TranslatedString('Gas', default_language='en')),
            DimensionCategory(id='oil', label=TranslatedString('Oil', default_language='en')),
        ],
    )
    return ctx


def _ppdf(rows: list[tuple], unit: str = 'kWh', dim: str | None = None) -> PathsDataFrame:
    """Build a frame from ``(year, value)`` rows, or ``(year, category, value)`` when ``dim`` is given."""
    if dim is None:
        df = pl.DataFrame(
            {YEAR_COLUMN: [r[0] for r in rows], VALUE_COLUMN: [r[1] for r in rows]},
            schema={YEAR_COLUMN: pl.Int64, VALUE_COLUMN: pl.Float64},
        )
        pks = [YEAR_COLUMN]
    else:
        df = pl.DataFrame(
            {YEAR_COLUMN: [r[0] for r in rows], dim: [r[1] for r in rows], VALUE_COLUMN: [r[2] for r in rows]},
            schema={YEAR_COLUMN: pl.Int64, dim: pl.String, VALUE_COLUMN: pl.Float64},
        )
        pks = [YEAR_COLUMN, dim]
    df = df.with_columns(pl.lit(value=False).alias(FORECAST_COLUMN))
    meta = DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units(unit)}, primary_keys=pks)
    return to_ppdf(df, meta)


def _source(
    context: Context,
    identifier: str,
    rows: list[tuple],
    unit: str = 'kWh',
    dim: str | None = None,
    quantity: str = 'energy',
) -> _FixedOutputNode:
    dims = [dim] if dim else None
    return _FixedOutputNode(
        id=identifier,
        context=context,
        name=TranslatedString(identifier, default_language='en'),
        unit=unit_registry.parse_units(unit),
        quantity=quantity,
        fixed_df=_ppdf(rows, unit=unit, dim=dim),
        output_dimension_ids=dims,
        input_dimension_ids=dims,
    )


def _additive(context: Context, identifier: str, unit: str = 'kWh', dims: list[str] | None = None) -> AdditiveNode:
    return AdditiveNode(
        id=identifier,
        context=context,
        name=TranslatedString(identifier, default_language='en'),
        unit=unit_registry.parse_units(unit),
        quantity='energy',
        output_dimension_ids=dims,
        input_dimension_ids=dims,
    )


def _multiplicative(context: Context, identifier: str, unit: str = 'kWh', dims: list[str] | None = None) -> MultiplicativeNode:
    return MultiplicativeNode(
        id=identifier,
        context=context,
        name=TranslatedString(identifier, default_language='en'),
        unit=unit_registry.parse_units(unit),
        quantity='energy',
        output_dimension_ids=dims,
        input_dimension_ids=dims,
    )


def _generic(context: Context, identifier: str, operations: str, unit: str = 'kWh', dims: list[str] | None = None) -> GenericNode:
    node = GenericNode(
        id=identifier,
        context=context,
        name=TranslatedString(identifier, default_language='en'),
        unit=unit_registry.parse_units(unit),
        quantity='energy',
        output_dimension_ids=dims,
        input_dimension_ids=dims,
    )
    param = StringParameter(local_id='operations', is_customizable=False)
    param.set(operations)
    node.add_parameter(param)
    return node


def _connect(source: Node, target: Node, tags: list[str] | None = None) -> None:
    edge = Edge(input_node=source, output_node=target, tags=tags or [])
    source.add_edge(edge)
    target.add_edge(edge)


def _attach(
    node: Node,
    identifier: str,
    df: PathsDataFrame,
    tags: list[str] | None = None,
    *,
    interpolate: bool = False,
    extend: bool = False,
    backfill: bool = False,
) -> None:
    node.input_dataset_instances.append(
        _FixedDataset(
            id=identifier,
            context=node.context,
            fixed_df=df,
            tags=tags or [],
            interpolate=interpolate,
            extend=extend,
            backfill=backfill,
        ),
    )


def _values(df: PathsDataFrame, dim: str | None = None) -> dict:
    """Return ``{year: value}``, or ``{(year, category): value}`` when ``dim`` is given."""
    df = df.sort(df.primary_keys)
    if dim is None:
        return {row[YEAR_COLUMN]: row[VALUE_COLUMN] for row in df.to_dicts()}
    return {(row[YEAR_COLUMN], row[dim]): row[VALUE_COLUMN] for row in df.to_dicts()}


# =================================================================================
# AdditiveNode: one additive multiport — identical dimensions, compatible units,
# missing values count as zero.
# =================================================================================


def test_additive_ragged_years_fills_missing_with_zero():
    """A year present in one input and absent from another is summed as if the absent value were zero."""
    ctx = _make_context('add-ragged')
    a = _source(ctx, 'a', [(2020, 2.0), (2021, 2.0)])
    b = _source(ctx, 'b', [(2020, 3.0)])
    node = _additive(ctx, 'sum')
    _connect(a, node)
    _connect(b, node)

    assert _values(node.compute()) == {2020: 5.0, 2021: 2.0}


def test_additive_ragged_years_matches_generic_add():
    ctx = _make_context('add-ragged-vs-generic')
    for identifier, factory in (('add', _additive), ('gen', None)):
        a = _source(ctx, f'a_{identifier}', [(2020, 2.0), (2021, 2.0)])
        b = _source(ctx, f'b_{identifier}', [(2020, 3.0)])
        node = factory(ctx, identifier) if factory else _generic(ctx, identifier, 'add')
        _connect(a, node)
        _connect(b, node)
        assert _values(node.compute()) == {2020: 5.0, 2021: 2.0}, identifier


def test_additive_converts_compatible_units():
    ctx = _make_context('add-units')
    a = _source(ctx, 'a', [(2020, 2.0)], unit='kWh')
    b = _source(ctx, 'b', [(2020, 1.0)], unit='MWh')
    node = _additive(ctx, 'sum', unit='kWh')
    _connect(a, node)
    _connect(b, node)

    out = node.compute()
    assert out.get_unit(VALUE_COLUMN) == unit_registry.parse_units('kWh')
    assert _values(out) == {2020: 1002.0}


def test_additive_incompatible_unit_raises():
    ctx = _make_context('add-bad-units')
    a = _source(ctx, 'a', [(2020, 2.0)], unit='kWh')
    b = _source(ctx, 'b', [(2020, 1.0)], unit='kg', quantity='mass')
    node = _additive(ctx, 'sum', unit='kWh')
    _connect(a, node)
    _connect(b, node)

    with pytest.raises(Exception, match=r'(?i)unit'):
        node.compute()


def test_additive_dimension_mismatch_raises():
    """The additive multiport requires an identical dimension set on every input."""
    ctx = _make_context('add-dim-mismatch')
    a = _source(ctx, 'a', [(2020, 'x', 2.0)], dim='sector')
    b = _source(ctx, 'b', [(2020, 1.0)])
    node = _additive(ctx, 'sum', dims=['sector'])
    _connect(a, node)
    _connect(b, node)

    with pytest.raises(NodeError, match=r'(?i)dimensions do not match'):
        node.compute()


def test_additive_missing_category_counts_as_zero():
    ctx = _make_context('add-ragged-cats')
    a = _source(ctx, 'a', [(2020, 'x', 2.0), (2020, 'y', 4.0)], dim='sector')
    b = _source(ctx, 'b', [(2020, 'x', 3.0)], dim='sector')
    node = _additive(ctx, 'sum', dims=['sector'])
    _connect(a, node)
    _connect(b, node)

    assert _values(node.compute(), dim='sector') == {(2020, 'x'): 5.0, (2020, 'y'): 4.0}


def test_additive_explicit_null_counts_as_zero():
    """A null *value* (not just an absent row) is also treated as zero."""
    ctx = _make_context('add-null')
    a = _source(ctx, 'a', [(2020, 2.0), (2021, 2.0)])
    b = _source(ctx, 'b', [(2020, 3.0), (2021, None)])
    node = _additive(ctx, 'sum')
    _connect(a, node)
    _connect(b, node)

    assert _values(node.compute()) == {2020: 5.0, 2021: 2.0}


def test_additive_nan_propagates_unlike_null():
    """NaN is *not* treated as zero: it poisons the sum. Divergence from the null rule above."""
    ctx = _make_context('add-nan')
    a = _source(ctx, 'a', [(2020, 2.0), (2021, 2.0)])
    b = _source(ctx, 'b', [(2020, 3.0), (2021, NAN)])
    node = _additive(ctx, 'sum')
    _connect(a, node)
    _connect(b, node)

    out = _values(node.compute())
    assert out[2020] == 5.0
    assert out[2021] != out[2021], 'NaN is expected to survive into the output'


def test_additive_order_independent():
    ctx = _make_context('add-order')
    results = []
    for order in ('ab', 'ba'):
        a = _source(ctx, f'a_{order}', [(2020, 2.0), (2021, 2.0)])
        b = _source(ctx, f'b_{order}', [(2020, 3.0)], unit='MWh')
        node = _additive(ctx, f'sum_{order}')
        first, second = (a, b) if order == 'ab' else (b, a)
        _connect(first, node)
        _connect(second, node)
        results.append(_values(node.compute()))

    assert results[0] == results[1]


def test_additive_sums_dataset_together_with_nodes_but_extends_only_the_dataset():
    """
    Combine a dataset with node inputs — but the two are not interchangeable.

    A dataset input runs through ``_process_input_dataset_df``, whose last step is
    ``extend_last_historical_value_pl``: the dataset's final value is carried to the model
    end year. A node input gets no such treatment. So the sum below is 12 for the two years
    both sides cover, and then the dataset's 10 alone all the way to 2030.
    """
    ctx = _make_context('add-dataset')
    a = _source(ctx, 'a', [(2020, 2.0), (2021, 2.0)])
    node = _additive(ctx, 'sum')
    _connect(a, node)
    _attach(node, 'ds', _ppdf([(2020, 10.0), (2021, 10.0)]))

    out = _values(node.compute())
    assert out[2020] == 12.0
    assert out[2021] == 12.0
    assert out[2030] == 10.0
    assert max(out) == node.get_end_year()


def test_additive_refuses_two_datasets():
    """Objective 1, the gap: two datasets are a hard error, however compatible they are."""
    ctx = _make_context('add-two-datasets')
    node = _additive(ctx, 'sum')
    _attach(node, 'ds1', _ppdf([(2020, 10.0)]))
    _attach(node, 'ds2', _ppdf([(2020, 5.0)]))

    with pytest.raises(NodeError, match=r'(?i)expected only 1 input dataset'):
        node.compute()


def test_additive_silently_drops_non_additive_input():
    """The base class collects 'non_additive' inputs and then never uses them. No error, no warning."""
    ctx = _make_context('add-non-additive')
    a = _source(ctx, 'a', [(2020, 2.0)])
    factor = _source(ctx, 'f', [(2020, 100.0)], unit='dimensionless', quantity='fraction')
    node = _additive(ctx, 'sum')
    _connect(a, node)
    _connect(factor, node, tags=['non_additive'])

    assert _values(node.compute()) == {2020: 2.0}


# =================================================================================
# MultiplicativeNode: >= 2 single-input factor ports (union of dimensions, inner
# join), plus one additive multiport added to the product.
# =================================================================================


def test_multiplicative_multiplies_two_factors():
    ctx = _make_context('mul-basic')
    a = _source(ctx, 'a', [(2020, 2.0), (2021, 3.0)], unit='kW', quantity='energy')
    b = _source(ctx, 'b', [(2020, 3.0), (2021, 4.0)], unit='h', quantity='duration')
    node = _multiplicative(ctx, 'product')
    _connect(a, node)
    _connect(b, node)

    out = node.compute()
    assert out.get_unit(VALUE_COLUMN) == unit_registry.parse_units('kWh')
    assert _values(out) == {2020: 6.0, 2021: 12.0}


def test_multiplicative_ragged_years_intersect():
    """A year missing from any factor cannot yield a product, so it is absent from the output."""
    ctx = _make_context('mul-ragged')
    a = _source(ctx, 'a', [(2020, 2.0), (2021, 2.0)], unit='kW', quantity='energy')
    b = _source(ctx, 'b', [(2020, 3.0)], unit='h', quantity='duration')
    node = _multiplicative(ctx, 'product')
    _connect(a, node)
    _connect(b, node)

    assert _values(node.compute()) == {2020: 6.0}


def test_multiplicative_ragged_years_matches_generic_multiply():
    ctx = _make_context('mul-ragged-vs-generic')
    outputs = {}
    for identifier in ('mul', 'gen'):
        a = _source(ctx, f'a_{identifier}', [(2020, 2.0), (2021, 2.0)], unit='kW', quantity='energy')
        b = _source(ctx, f'b_{identifier}', [(2020, 3.0)], unit='h', quantity='duration')
        node = _multiplicative(ctx, identifier) if identifier == 'mul' else _generic(ctx, identifier, 'multiply')
        _connect(a, node)
        _connect(b, node)
        outputs[identifier] = _values(node.compute())

    assert outputs['mul'] == outputs['gen'] == {2020: 6.0}


def test_multiplicative_takes_union_of_dimensions():
    """A dimensionless factor without the dimension broadcasts across its categories."""
    ctx = _make_context('mul-union')
    a = _source(ctx, 'a', [(2020, 'x', 2.0), (2020, 'y', 4.0)], unit='kW', dim='sector', quantity='energy')
    b = _source(ctx, 'b', [(2020, 3.0)], unit='h', quantity='duration')
    node = _multiplicative(ctx, 'product', dims=['sector'])
    _connect(a, node)
    _connect(b, node)

    out = node.compute()
    assert set(out.dim_ids) == {'sector'}
    assert _values(out, dim='sector') == {(2020, 'x'): 6.0, (2020, 'y'): 12.0}


def test_multiplicative_missing_category_intersects():
    ctx = _make_context('mul-ragged-cats')
    a = _source(ctx, 'a', [(2020, 'x', 2.0), (2020, 'y', 4.0)], unit='kW', dim='sector', quantity='energy')
    b = _source(ctx, 'b', [(2020, 'x', 3.0)], unit='h', dim='sector', quantity='duration')
    node = _multiplicative(ctx, 'product', dims=['sector'])
    _connect(a, node)
    _connect(b, node)

    assert _values(node.compute(), dim='sector') == {(2020, 'x'): 6.0}


def test_multiplicative_drops_null_rows_where_generic_propagates_them():
    """
    The remaining divergence between the two implementations.

    A factor with a null value at 2021: ``GenericNode`` propagates the null, which is the
    agreed rule — the row stays, with no value. ``MultiplicativeNode.perform_operation``
    ends with ``drop_nulls(Value)``, so the row disappears instead: silent data loss, a
    year that simply stops existing downstream. ``MultiplicativeNode2`` adopts the
    ``GenericNode`` behaviour.
    """
    ctx = _make_context('mul-null')
    outputs = {}
    for identifier in ('mul', 'gen'):
        a = _source(ctx, f'a_{identifier}', [(2020, 2.0), (2021, 2.0)], unit='kW', quantity='energy')
        b = _source(ctx, f'b_{identifier}', [(2020, 3.0), (2021, None)], unit='h', quantity='duration')
        node = _multiplicative(ctx, identifier) if identifier == 'mul' else _generic(ctx, identifier, 'multiply')
        _connect(a, node)
        _connect(b, node)
        outputs[identifier] = _values(node.compute())

    assert outputs['mul'] == {2020: 6.0}
    assert outputs['gen'] == {2020: 6.0, 2021: None}


def test_multiplicative_keeps_nan_rows():
    """NaN is not null, so ``drop_nulls`` does not remove it and the NaN reaches the output."""
    ctx = _make_context('mul-nan')
    a = _source(ctx, 'a', [(2020, 2.0), (2021, 2.0)], unit='kW', quantity='energy')
    b = _source(ctx, 'b', [(2020, 3.0), (2021, NAN)], unit='h', quantity='duration')
    node = _multiplicative(ctx, 'product')
    _connect(a, node)
    _connect(b, node)

    out = _values(node.compute())
    assert out[2020] == 6.0
    assert out[2021] != out[2021], 'NaN is expected to survive into the output'


def test_multiplicative_adds_additive_side_input_to_product():
    ctx = _make_context('mul-plus-add')
    a = _source(ctx, 'a', [(2020, 2.0)], unit='kW', quantity='energy')
    b = _source(ctx, 'b', [(2020, 3.0)], unit='h', quantity='duration')
    side = _source(ctx, 'side', [(2020, 10.0)], unit='kWh')
    node = _multiplicative(ctx, 'product')
    _connect(a, node)
    _connect(b, node)
    _connect(side, node)

    assert _values(node.compute()) == {2020: 16.0}


def test_multiplicative_additive_input_dimension_mismatch_raises():
    ctx = _make_context('mul-add-dim-mismatch')
    a = _source(ctx, 'a', [(2020, 'x', 2.0)], unit='kW', dim='sector', quantity='energy')
    b = _source(ctx, 'b', [(2020, 3.0)], unit='h', quantity='duration')
    side = _source(ctx, 'side', [(2020, 10.0)], unit='kWh')
    node = _multiplicative(ctx, 'product', dims=['sector'])
    _connect(a, node)
    _connect(b, node)
    _connect(side, node)

    with pytest.raises(NodeError, match=r'(?i)dimensions do not match'):
        node.compute()


def test_multiplicative_order_independent_with_three_factors():
    ctx = _make_context('mul-order')
    results = []
    for order in ('abc', 'cba'):
        a = _source(ctx, f'a_{order}', [(2020, 2.0), (2021, 2.0)], unit='kW', quantity='energy')
        b = _source(ctx, f'b_{order}', [(2020, 3.0)], unit='h', quantity='duration')
        c = _source(ctx, f'c_{order}', [(2020, 5.0), (2021, 5.0), (2022, 5.0)], unit='dimensionless', quantity='fraction')
        node = _multiplicative(ctx, f'product_{order}')
        for source in (a, b, c) if order == 'abc' else (c, b, a):
            _connect(source, node)
        results.append(_values(node.compute()))

    assert results[0] == results[1] == {2020: 30.0}


def test_multiplicative_order_independent_with_differing_dimensions():
    ctx = _make_context('mul-order-dims')
    results = []
    for order in ('ab', 'ba'):
        a = _source(ctx, f'a_{order}', [(2020, 'x', 2.0), (2020, 'y', 4.0)], unit='kW', dim='sector', quantity='energy')
        b = _source(ctx, f'b_{order}', [(2020, 'gas', 3.0), (2020, 'oil', 5.0)], unit='h', dim='fuel', quantity='duration')
        node = _multiplicative(ctx, f'product_{order}', dims=['sector', 'fuel'])
        first, second = (a, b) if order == 'ab' else (b, a)
        _connect(first, node)
        _connect(second, node)
        out = node.compute()
        results.append({(row[YEAR_COLUMN], row['sector'], row['fuel']): row[VALUE_COLUMN] for row in out.to_dicts()})

    assert results[0] == results[1]
    assert results[0] == {
        (2020, 'x', 'gas'): 6.0,
        (2020, 'x', 'oil'): 10.0,
        (2020, 'y', 'gas'): 12.0,
        (2020, 'y', 'oil'): 20.0,
    }


def test_multiplicative_single_factor_raises():
    """Objective 4: at least two factor ports. One factor plus one additive input is not enough."""
    ctx = _make_context('mul-single')
    a = _source(ctx, 'a', [(2020, 2.0)], unit='kW', quantity='energy')
    side = _source(ctx, 'side', [(2020, 10.0)], unit='kWh')
    node = _multiplicative(ctx, 'product')
    _connect(a, node)
    _connect(side, node)

    with pytest.raises(NodeError, match=r'(?i)at least two inputs'):
        node.compute()


def test_multiplicative_ignores_input_dataset():
    """
    Objective 1, the gap: a dataset bound to a MultiplicativeNode is loaded and then ignored.

    Two factor nodes plus a dataset of the node's own unit — the dataset contributes
    nothing, silently. Config that looks like it multiplies (or adds) a dataset does not.
    """
    ctx = _make_context('mul-dataset')
    a = _source(ctx, 'a', [(2020, 2.0)], unit='kW', quantity='energy')
    b = _source(ctx, 'b', [(2020, 3.0)], unit='h', quantity='duration')
    node = _multiplicative(ctx, 'product')
    _connect(a, node)
    _connect(b, node)
    _attach(node, 'ds', _ppdf([(2020, 1000.0)]))

    assert _values(node.compute()) == {2020: 6.0}


# =================================================================================
# Cross-cutting: the null/NaN boundary both classes sit on.
# =================================================================================


def test_unit_conversion_preserves_nulls():
    """
    A null survives a unit conversion as a null.

    ``ensure_unit`` converts through numpy, which renders nulls as NaN; the null mask is
    restored afterwards. Without that, every converted null would arrive downstream as a
    NaN — poisoning sums and failing ``Node.check()`` — and neither "missing counts as
    zero" (addition) nor "null propagates" (multiplication) would be reachable.
    """
    df = _ppdf([(2020, 1.0), (2021, None)], unit='MWh')
    converted = df.ensure_unit(VALUE_COLUMN, unit_registry.parse_units('kWh'))

    assert converted[VALUE_COLUMN].is_null().sum() == 1
    assert converted[VALUE_COLUMN].is_nan().sum() == 0
    assert converted[VALUE_COLUMN][0] == 1000.0


def test_unit_conversion_keeps_genuine_nans_as_nans():
    """Only the null mask is restored; a NaN that was in the data stays a NaN."""
    df = _ppdf([(2020, 1.0), (2021, NAN), (2022, None)], unit='MWh')
    converted = df.ensure_unit(VALUE_COLUMN, unit_registry.parse_units('kWh'))

    assert converted[VALUE_COLUMN].is_nan().sum() == 1
    assert converted[VALUE_COLUMN].is_null().sum() == 1


def test_additive_null_counts_as_zero_across_a_unit_conversion():
    """The null-counts-as-zero rule holds whether or not the operand needed converting."""
    ctx = _make_context('add-null-units')
    a = _source(ctx, 'a', [(2020, 2.0), (2021, 2.0)], unit='kWh')
    b = _source(ctx, 'b', [(2020, 3.0), (2021, None)], unit='MWh')
    node = _additive(ctx, 'sum', unit='kWh')
    _connect(a, node)
    _connect(b, node)

    assert _values(node.compute()) == {2020: 3002.0, 2021: 2.0}


def test_multiplicative_with_only_datasets_raises():
    """Two datasets that could be multiplied are not multiplied; the node reports no inputs at all."""
    ctx = _make_context('mul-datasets-only')
    node = _multiplicative(ctx, 'product')
    _attach(node, 'ds1', _ppdf([(2020, 2.0)], unit='kW'))
    _attach(node, 'ds2', _ppdf([(2020, 3.0)], unit='h'))

    with pytest.raises(NodeError, match=r'(?i)at least two inputs'):
        node.compute()


# =================================================================================
# AdditiveNode2 / MultiplicativeNode2: the rebuilt classes.
#
# Where v1 and v2 should agree, the case is asserted against both. Where they differ,
# the difference is one of the four decisions in
# docs/plans/additive-multiplicative-modernization.md, and says so.
# =================================================================================


def _additive2(context: Context, identifier: str, unit: str = 'kWh', dims: list[str] | None = None) -> AdditiveNode2:
    return AdditiveNode2(
        id=identifier,
        context=context,
        name=TranslatedString(identifier, default_language='en'),
        unit=unit_registry.parse_units(unit),
        quantity='energy',
        output_dimension_ids=dims,
        input_dimension_ids=dims,
    )


def _multiplicative2(context: Context, identifier: str, unit: str = 'kWh', dims: list[str] | None = None) -> MultiplicativeNode2:
    return MultiplicativeNode2(
        id=identifier,
        context=context,
        name=TranslatedString(identifier, default_language='en'),
        unit=unit_registry.parse_units(unit),
        quantity='energy',
        output_dimension_ids=dims,
        input_dimension_ids=dims,
    )


ADDITIVE_BUILDERS = pytest.mark.parametrize('build', [_additive, _additive2], ids=['v1', 'v2'])
MULTIPLICATIVE_BUILDERS = pytest.mark.parametrize('build', [_multiplicative, _multiplicative2], ids=['v1', 'v2'])


# --- rules both versions must obey ------------------------------------------------


@ADDITIVE_BUILDERS
def test_both_additive_versions_fill_missing_years_with_zero(build, request):
    ctx = _make_context(request.node.name[:60])
    _connect(_source(ctx, 'a', [(2020, 2.0), (2021, 2.0)]), node := build(ctx, 'sum'))
    _connect(_source(ctx, 'b', [(2020, 3.0)]), node)

    assert _values(node.compute()) == {2020: 5.0, 2021: 2.0}


@ADDITIVE_BUILDERS
def test_both_additive_versions_count_nulls_as_zero(build, request):
    ctx = _make_context(request.node.name[:60])
    _connect(_source(ctx, 'a', [(2020, 2.0), (2021, 2.0)]), node := build(ctx, 'sum'))
    _connect(_source(ctx, 'b', [(2020, 3.0), (2021, None)], unit='MWh'), node)

    assert _values(node.compute()) == {2020: 3002.0, 2021: 2.0}


@ADDITIVE_BUILDERS
def test_both_additive_versions_reject_mismatched_dimensions(build, request):
    ctx = _make_context(request.node.name[:60])
    _connect(_source(ctx, 'a', [(2020, 'x', 2.0)], dim='sector'), node := build(ctx, 'sum', dims=['sector']))
    _connect(_source(ctx, 'b', [(2020, 1.0)]), node)

    with pytest.raises(NodeError, match=r'(?i)dimensions do not match'):
        node.compute()


@ADDITIVE_BUILDERS
def test_both_additive_versions_are_order_independent(build, request):
    results = []
    for order in ('ab', 'ba'):
        ctx = _make_context(f'{request.node.name[:50]}-{order}')
        a = _source(ctx, 'a', [(2020, 2.0), (2021, 2.0)])
        b = _source(ctx, 'b', [(2020, 3.0)], unit='MWh')
        node = build(ctx, 'sum')
        for source in (a, b) if order == 'ab' else (b, a):
            _connect(source, node)
        results.append(_values(node.compute()))

    assert results[0] == results[1]


@MULTIPLICATIVE_BUILDERS
def test_both_multiplicative_versions_multiply_two_factors(build, request):
    ctx = _make_context(request.node.name[:60])
    node = build(ctx, 'product')
    _connect(_source(ctx, 'a', [(2020, 2.0), (2021, 3.0)], unit='kW'), node)
    _connect(_source(ctx, 'b', [(2020, 3.0), (2021, 4.0)], unit='h', quantity='duration'), node)

    out = node.compute()
    assert out.get_unit(VALUE_COLUMN) == unit_registry.parse_units('kWh')
    assert _values(out) == {2020: 6.0, 2021: 12.0}


@MULTIPLICATIVE_BUILDERS
def test_both_multiplicative_versions_intersect_ragged_years(build, request):
    ctx = _make_context(request.node.name[:60])
    node = build(ctx, 'product')
    _connect(_source(ctx, 'a', [(2020, 2.0), (2021, 2.0)], unit='kW'), node)
    _connect(_source(ctx, 'b', [(2020, 3.0)], unit='h', quantity='duration'), node)

    assert _values(node.compute()) == {2020: 6.0}


@MULTIPLICATIVE_BUILDERS
def test_both_multiplicative_versions_take_the_union_of_dimensions(build, request):
    ctx = _make_context(request.node.name[:60])
    node = build(ctx, 'product', dims=['sector'])
    _connect(_source(ctx, 'a', [(2020, 'x', 2.0), (2020, 'y', 4.0)], unit='kW', dim='sector'), node)
    _connect(_source(ctx, 'b', [(2020, 3.0)], unit='h', quantity='duration'), node)

    assert _values(node.compute(), dim='sector') == {(2020, 'x'): 6.0, (2020, 'y'): 12.0}


@MULTIPLICATIVE_BUILDERS
def test_both_multiplicative_versions_add_the_additive_input_to_the_product(build, request):
    ctx = _make_context(request.node.name[:60])
    node = build(ctx, 'product')
    _connect(_source(ctx, 'a', [(2020, 2.0)], unit='kW'), node)
    _connect(_source(ctx, 'b', [(2020, 3.0)], unit='h', quantity='duration'), node)
    _connect(_source(ctx, 'side', [(2020, 10.0)], unit='kWh'), node)

    assert _values(node.compute()) == {2020: 16.0}


@MULTIPLICATIVE_BUILDERS
def test_both_multiplicative_versions_need_two_factors(build, request):
    ctx = _make_context(request.node.name[:60])
    node = build(ctx, 'product')
    _connect(_source(ctx, 'a', [(2020, 2.0)], unit='kW'), node)
    _connect(_source(ctx, 'side', [(2020, 10.0)], unit='kWh'), node)

    with pytest.raises(NodeError, match=r'(?i)at least two'):
        node.compute()


@MULTIPLICATIVE_BUILDERS
def test_both_multiplicative_versions_are_order_independent(build, request):
    results = []
    for order in ('abc', 'cba'):
        ctx = _make_context(f'{request.node.name[:50]}-{order}')
        a = _source(ctx, 'a', [(2020, 2.0), (2021, 2.0)], unit='kW')
        b = _source(ctx, 'b', [(2020, 3.0)], unit='h', quantity='duration')
        c = _source(ctx, 'c', [(2020, 5.0)], unit='dimensionless', quantity='fraction')
        node = build(ctx, 'product')
        for source in (a, b, c) if order == 'abc' else (c, b, a):
            _connect(source, node)
        results.append(_values(node.compute()))

    assert results[0] == results[1] == {2020: 30.0}


# --- objective 1: a dataset and a node are the same kind of input -----------------


def test_additive2_sums_several_datasets():
    """Where v1 raises on a second dataset, v2 simply adds them."""
    ctx = _make_context('add2-datasets')
    node = _additive2(ctx, 'sum')
    _attach(node, 'ds1', _ppdf([(2020, 10.0), (2021, 10.0)]))
    _attach(node, 'ds2', _ppdf([(2020, 5.0)], unit='MWh'))

    assert _values(node.compute()) == {2020: 5010.0, 2021: 10.0}


def test_additive2_mixes_datasets_and_nodes_on_equal_terms():
    ctx = _make_context('add2-mixed')
    node = _additive2(ctx, 'sum')
    _connect(_source(ctx, 'a', [(2020, 2.0), (2021, 2.0)]), node)
    _attach(node, 'ds', _ppdf([(2020, 10.0), (2021, 10.0)]))

    out = _values(node.compute())
    assert out == {2020: 12.0, 2021: 12.0}
    assert max(out) == 2021, 'the dataset must not be extended past its own last year'


def test_multiplicative2_multiplies_a_dataset_by_a_node():
    """
    The pilot's proof case: longmont-dev's waste_composted / waste_landfilled / waste_recycled.

    They are GenericNodes today precisely because MultiplicativeNode ignores datasets.
    """
    ctx = _make_context('mul2-dataset-node')
    node = _multiplicative2(ctx, 'product')
    _connect(_source(ctx, 'hours', [(2020, 3.0), (2021, 4.0)], unit='h', quantity='duration'), node)
    _attach(node, 'power', _ppdf([(2020, 2.0), (2021, 3.0)], unit='kW'))

    assert _values(node.compute()) == {2020: 6.0, 2021: 12.0}


def test_multiplicative2_multiplies_two_datasets():
    ctx = _make_context('mul2-two-datasets')
    node = _multiplicative2(ctx, 'product')
    _attach(node, 'power', _ppdf([(2020, 2.0)], unit='kW'))
    _attach(node, 'hours', _ppdf([(2020, 3.0)], unit='h'))

    assert _values(node.compute()) == {2020: 6.0}


def test_dataset_operands_are_classified_by_tag_like_nodes():
    """A dataset whose unit matches the node is additive unless a tag says otherwise."""
    ctx = _make_context('mul2-tagged-dataset')
    node = _multiplicative2(ctx, 'product', unit='kWh')
    _connect(_source(ctx, 'hours', [(2020, 3.0)], unit='h', quantity='duration'), node)
    _attach(node, 'power', _ppdf([(2020, 2.0)], unit='kW'))
    # Same unit as the node, so additive by the unit rule.
    _attach(node, 'extra', _ppdf([(2020, 10.0)], unit='kWh'))

    assert _values(node.compute()) == {2020: 16.0}


# --- the four decisions -----------------------------------------------------------


def test_multiplicative2_propagates_a_null_factor_instead_of_dropping_the_row():
    """Decision 1. v1 drops the row; v2 keeps it and reports the value as unknown."""
    ctx = _make_context('mul2-null')
    v1 = _multiplicative(ctx, 'v1')
    v2 = _multiplicative2(ctx, 'v2')
    for node in (v1, v2):
        suffix = node.id
        _connect(_source(ctx, f'a_{suffix}', [(2020, 2.0), (2021, 2.0)], unit='kW'), node)
        _connect(_source(ctx, f'b_{suffix}', [(2020, 3.0), (2021, None)], unit='h', quantity='duration'), node)

    assert _values(v1.compute()) == {2020: 6.0}
    assert _values(v2.compute()) == {2020: 6.0, 2021: None}


def test_additive2_does_not_extend_a_dataset_to_the_end_year():
    """
    Decision 3. v1 carries a dataset's last value to the model end year; v2 does not.

    Extension becomes a property of the binding, so that a dataset means the same thing
    however it is wired. Converting a model therefore has to say where it wants extension.
    """
    ctx = _make_context('add2-no-extend')
    v1 = _additive(ctx, 'v1')
    v2 = _additive2(ctx, 'v2')
    for node in (v1, v2):
        _attach(node, f'ds_{node.id}', _ppdf([(2020, 10.0), (2021, 10.0)]))

    assert max(_values(v1.compute())) == v1.get_end_year()
    assert max(_values(v2.compute())) == 2021


def test_additive2_refuses_a_non_additive_input_instead_of_dropping_it():
    """F6. v1 silently drops the factor; v2 says it cannot multiply."""
    ctx = _make_context('add2-refuses-factor')
    v1 = _additive(ctx, 'v1')
    v2 = _additive2(ctx, 'v2')
    for node in (v1, v2):
        _connect(_source(ctx, f'a_{node.id}', [(2020, 2.0)]), node)
        _connect(
            _source(ctx, f'f_{node.id}', [(2020, 100.0)], unit='dimensionless', quantity='fraction'),
            node,
            tags=['non_additive'],
        )

    assert _values(v1.compute()) == {2020: 2.0}
    with pytest.raises(NodeError, match=r'(?i)cannot multiply'):
        v2.compute()


def test_additive2_refuses_an_input_claimed_by_an_operation_it_does_not_have():
    ctx = _make_context('add2-refuses-claimed')
    node = _additive2(ctx, 'sum')
    _connect(_source(ctx, 'a', [(2020, 2.0)]), node)
    _connect(_source(ctx, 'splitter', [(2020, 1.0)]), node, tags=['use_as_totals'])

    with pytest.raises(NodeError, match=r'(?i)tagged for an operation'):
        node.compute()


def test_additive2_marks_itself_incomplete_when_nothing_is_wired_up():
    """Fault tolerance survives the rebuild: an unwired node reports itself, not a crash."""
    ctx = _make_context('add2-incomplete')
    node = _additive2(ctx, 'sum')

    out = node.compute()
    assert len(out) == 0
    assert node.status is NodeStatus.INCOMPLETE


def test_additive2_inventory_only_marks_the_whole_result_historical():
    ctx = _make_context('add2-inventory')
    node = _additive2(ctx, 'sum')
    param = BoolParameter(local_id='inventory_only', is_customizable=False)
    param.set(True)
    node.add_parameter(param)
    _attach(node, 'ds', _ppdf([(2020, 10.0)]))

    assert not node.compute()[FORECAST_COLUMN].any()


def test_impute_overlays_the_result_in_both_versions():
    ctx = _make_context('add2-impute')
    node = _additive2(ctx, 'sum')
    _connect(_source(ctx, 'a', [(2020, 2.0), (2021, 2.0)]), node)
    _connect(_source(ctx, 'override', [(2021, 99.0)]), node, tags=['impute'])

    assert _values(node.compute()) == {2020: 2.0, 2021: 99.0}


# --- the binding flags, at runtime -------------------------------------------------


def test_extend_flag_carries_a_dataset_to_the_model_end_year():
    """Decision 3 in practice: the binding asks for extension, and gets exactly that."""
    ctx = _make_context('add2-extend-flag')
    node = _additive2(ctx, 'sum')
    _attach(node, 'ds', _ppdf([(2020, 10.0), (2021, 10.0)]), extend=True)

    out = _values(node.compute())
    assert max(out) == node.get_end_year()
    assert out[node.get_end_year()] == 10.0


def test_interpolate_flag_fills_an_interior_year_gap():
    ctx = _make_context('add2-interpolate-flag')
    node = _additive2(ctx, 'sum')
    _attach(node, 'ds', _ppdf([(2020, 10.0), (2023, 40.0)]), interpolate=True)

    assert _values(node.compute()) == {2020: 10.0, 2021: 20.0, 2022: 30.0, 2023: 40.0}


def test_without_the_interpolate_flag_the_gap_stays_a_gap():
    ctx = _make_context('add2-no-interpolate')
    node = _additive2(ctx, 'sum')
    _attach(node, 'ds', _ppdf([(2020, 10.0), (2023, 40.0)]))

    assert _values(node.compute()) == {2020: 10.0, 2023: 40.0}


def test_backfill_flag_copies_the_first_known_value_backwards():
    """
    Back-filling stays available, but only when a binding asks for it.

    ``GenericNode`` did this silently for every dataset, which is how four longmont-dev
    series got values for years their data does not cover. It is a back-cast, so it is now
    something a binding says out loud.
    """
    ctx = _make_context('add2-backfill')
    node = _additive2(ctx, 'sum')
    _attach(node, 'ds', _ppdf([(2020, None), (2021, None), (2022, 30.0)]), backfill=True)

    assert _values(node.compute()) == {2020: 30.0, 2021: 30.0, 2022: 30.0}


def test_without_backfill_the_leading_nulls_stay_null():
    ctx = _make_context('add2-no-backfill')
    node = _additive2(ctx, 'sum')
    _attach(node, 'ds', _ppdf([(2020, None), (2021, None), (2022, 30.0)]))

    assert _values(node.compute()) == {2020: None, 2021: None, 2022: 30.0}


def test_backfill_is_per_category():
    """Each category's own first value goes backwards; categories do not borrow from each other."""
    ctx = _make_context('add2-backfill-dims')
    node = _additive2(ctx, 'sum', dims=['sector'])
    rows = [
        (2020, 'x', None),
        (2021, 'x', 10.0),
        (2020, 'y', 5.0),
        (2021, 'y', 50.0),
    ]
    _attach(node, 'ds', _ppdf(rows, dim='sector'), backfill=True)

    assert _values(node.compute(), dim='sector') == {
        (2020, 'x'): 10.0,
        (2021, 'x'): 10.0,
        (2020, 'y'): 5.0,
        (2021, 'y'): 50.0,
    }
