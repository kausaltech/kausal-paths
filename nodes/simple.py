from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, ClassVar

from django.utils.translation import gettext_lazy as _

import polars as pl

from common import polars as ppl
from common.i18n import TranslatedString
from nodes.calc import convert_to_co2e, extend_last_historical_value_pl
from nodes.units import Quantity
from params.param import BoolParameter, NumberParameter, Parameter, StringParameter

from .constants import FORECAST_COLUMN, MIX_QUANTITY, VALUE_COLUMN, YEAR_COLUMN
from .exceptions import NodeError
from .node import Node, NodeMetric

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

EMISSION_UNIT = 'kg'


class SimpleNode(Node):
    allowed_parameters: ClassVar[Sequence[Parameter[Any]]] = [
        BoolParameter(
            local_id='fill_gaps_using_input_dataset',
            label=TranslatedString(en="Fill in gaps in computation using input dataset"),
            is_customizable=False,
        ),
        BoolParameter(
            local_id='replace_output_using_input_dataset',
            label=TranslatedString(en="Replace output using input dataset"),
            is_customizable=False,
        ),
        BoolParameter(
            local_id='drop_nulls',
            description='At the end of compute() do you want to drop nulls?',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='replace_nans',
            description='At the end of compute() replace nans with this value',
            is_customizable=False,
        ),
        StringParameter(
            local_id='reference_category',
            description='Category to which all others are compared',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='reference_year',
            description='Year to which all others are compared',
            is_customizable=False,
        ),
        StringParameter(
            local_id='share_dimension',
            description='Dimension over which values are converted to shares',
            is_customizable=False,
        ),
        NumberParameter(  # FIXME Make sure that the treatment is systematic in all node classes.
            local_id='multiplier',
            description='Multiplier to implement after operation and before additions',
            is_customizable=False,
        ),
        StringParameter(
            local_id='slice_category_at_edge',
            description='A category is sliced at edge before offering as input to another node',
            is_customizable=False,
        ),
        StringParameter(
            local_id='filter_categories',
            description='Categories to filter in format dimension:category,category2',
            is_customizable=False,
        ),
    ]

    def replace_output_using_input_dataset_pl(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        # If we have also data from an input dataset, we only fill in the gaps from the
        # calculated data.
        df = df.drop_nulls()

        input_df = self.get_input_dataset_pl(required=False)
        if input_df is None:
            return df
        data_df = input_df

        data_latest_year: int = data_df[YEAR_COLUMN].max()  # type: ignore
        df_latest_year: int = df[YEAR_COLUMN].max()  # type: ignore
        df_meta = df.get_meta()
        data_meta = data_df.get_meta()
        if df_latest_year > data_latest_year:
            for col in data_meta.metric_cols:
                data_df = data_df.ensure_unit(col, df_meta.units[col])
            data_df = data_df.paths.join_over_index(df, how='outer')
            fills = [pl.col(col).fill_null(pl.col(col + '_right')) for col in data_meta.metric_cols]
            data_df = data_df.select([YEAR_COLUMN, *data_meta.dim_ids, FORECAST_COLUMN, *fills], units=df_meta.units)

        return data_df

    def replace_output_using_input_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.replace_output_using_input_dataset_pl(ppl.from_pandas(df)).to_pandas()

    def fill_gaps_using_input_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        ndf = ppl.from_pandas(df)
        out = self.fill_gaps_using_input_dataset_pl(ndf)
        return out.to_pandas()

    def fill_gaps_using_input_dataset_pl(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        data_df = self.get_input_dataset_pl(required=False)
        if data_df is None:
            return df

        meta = df.get_meta()
        df = df.paths.join_over_index(data_df, how='outer')
        for metric_col in meta.metric_cols:
            right = '%s_right' % metric_col  # FIXME Not clear that the right column has same metric name as left
            df = df.ensure_unit(right, meta.units[metric_col])
            df = df.with_columns([
                pl.col(metric_col).fill_null(pl.col(right)),
            ])
        return df

    def maybe_drop_nulls(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        if self.get_parameter_value('drop_nulls', required=False):
            df = df.drop_nulls()
        return df

    def replace_nans(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        rep = self.get_parameter_value('replace_nans', required=False)
        if rep is not None:
            df = df.with_columns(
                pl.when(pl.col(VALUE_COLUMN).is_nan() | pl.col(VALUE_COLUMN).is_infinite())
                .then(pl.lit(rep))
                .otherwise(pl.col(VALUE_COLUMN))
                .alias(VALUE_COLUMN)
            )
        return df

    def scale_by_reference_category(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        param = self.get_parameter_value_str('reference_category', required=False)
        if param:
            col, cat = param.split(':')
            reference = df.filter(pl.col(col).eq(cat)).drop(col)
            df = df.paths.join_over_index(reference)
            df = df.divide_cols([VALUE_COLUMN, VALUE_COLUMN + '_right'], VALUE_COLUMN).drop(VALUE_COLUMN + '_right')

        return df

    def scale_by_reference_year(self, df: ppl.PathsDataFrame, year: int | None = None) -> ppl.PathsDataFrame:
        if not year:
            year = self.get_parameter_value('reference_year', required=False)
        if year:
            df = self._scale_by_reference_year(df, year)
            df = df.ensure_unit(VALUE_COLUMN, self.unit)
        return df

    def get_shares(self, df: ppl.PathsDataFrame, dim: str | None = None) -> ppl.PathsDataFrame:
        if not dim:
            dim = self.get_parameter_value('share_dimension', required = False)
        if dim:
            df = df.paths.calculate_shares(VALUE_COLUMN, VALUE_COLUMN, [dim])

        return df

    # See also sister function in ActionNode
    def apply_multiplier(self, df: ppl.PathsDataFrame, required: bool, units: bool) -> ppl.PathsDataFrame:
        multiplier = self.get_parameter_value('multiplier', required=required, units=units)
        if multiplier is not None:
            if isinstance(multiplier, Quantity):
                df = df.multiply_quantity(VALUE_COLUMN, multiplier)
            else:
                df = df.with_columns((pl.col(VALUE_COLUMN) * pl.lit(multiplier)).alias(VALUE_COLUMN))
            df = df.ensure_unit(VALUE_COLUMN, self.unit)
        return df


class AdditiveNode(SimpleNode):
    explanation = _("""This is an Additive Node. It performs a simple addition of inputs.
Missing values are assumed to be zero.""")
    allowed_parameters = [
        *SimpleNode.allowed_parameters,
        BoolParameter(local_id='drop_nans', is_customizable=False),
        StringParameter(local_id='metric', is_customizable=False),
        BoolParameter(
            local_id='inventory_only',
            description='Node represents historical (inventory) values only',
            is_customizable=False,
        ),
        BoolParameter(
            local_id='use_input_node_unit_when_adding',
            description='Use input node unit when doing add_nodes_pl()',
            is_customizable=False,
        ),
    ]

    def add_nodes(self, ndf: pd.DataFrame | None, nodes: list[Node], metric: str | None = None) -> pd.DataFrame:
        if ndf is not None:
            df = ppl.from_pandas(ndf)
        else:
            df = None
        out = self.add_nodes_pl(df, nodes, metric)
        return out.to_pandas()

    def _process_input_dataset_df(self, df: ppl.PathsDataFrame, metric: str | None) -> ppl.PathsDataFrame:
        if VALUE_COLUMN not in df.columns:
            if len(df.metric_cols) == 1:
                df = df.rename({df.metric_cols[0]: VALUE_COLUMN})
            elif metric is not None:
                if metric in df.columns:
                    df = df.rename({metric: VALUE_COLUMN})
                    cols = [YEAR_COLUMN, *df.dim_ids, VALUE_COLUMN]
                    if FORECAST_COLUMN in df.columns:
                        cols.append(FORECAST_COLUMN)
                    df = df.select(cols)
                else:
                    raise NodeError(self, "Metric is not found in metric columns")
            else:
                compatible_cols = [
                    col for col, unit in df.get_meta().units.items()
                    if self.is_compatible_unit(unit, self.unit)
                ]
                if len(compatible_cols) == 1:
                    df = df.rename({compatible_cols[0]: VALUE_COLUMN})
                    cols = [YEAR_COLUMN, *df.dim_ids, VALUE_COLUMN]
                    if FORECAST_COLUMN in df.columns:
                        cols.append(FORECAST_COLUMN)
                    df = df.select(cols)
                else:
                    raise NodeError(self, "Input dataset has multiple metric columns, but no Value column")

        df = df.ensure_unit(VALUE_COLUMN, self.single_metric_unit)

        if self.get_parameter_value('inventory_only', required=False):
            df = df.with_columns([pl.lit(value=False).alias(FORECAST_COLUMN)])
        else:
            df = extend_last_historical_value_pl(df, self.get_end_year())
        return df

    def compute(self) -> ppl.PathsDataFrame:
        idf = self.get_input_dataset_pl(required=False)
        metric = self.get_parameter_value_str('metric', required=False)
        assert self.unit is not None
        if idf is not None:
            idf = self._process_input_dataset_df(idf, metric)

        na_nodes = self.get_input_nodes(tag='non_additive')
        input_nodes = [node for node in self.input_nodes if node not in na_nodes]

        if self.get_parameter_value('use_input_node_unit_when_adding', required=False):
            unit = self.input_nodes[0].unit
        else:
            unit = self.unit
        if self.get_parameter_value('fill_gaps_using_input_dataset', required=False):
            df = self.add_nodes_pl(None, input_nodes, metric, unit=unit)
            df = self.fill_gaps_using_input_dataset_pl(df)
        else:
            df = self.add_nodes_pl(idf, input_nodes, metric, unit=unit)
        df = self.maybe_drop_nulls(df)  # FIXME Check where this should be done.
        if self.get_parameter_value('drop_nans', required=False):  # FIXME: Implement this in the same way as drop_nulls
            df = df.filter(~pl.col(VALUE_COLUMN).is_nan())
        df = self.scale_by_reference_category(df)
        df = self.scale_by_reference_year(df)
        df = self.get_shares(df)

        return df


class SubtractiveNode(Node):
    explanation = _(
        'This is a Subtractive Node. It takes the first input node and subtracts all other input nodes from it.',
    )  # FIXME Is this needed? Edge process arithmetic_inverse could be used instead.
    allowed_parameters = [
        BoolParameter(
            local_id='only_historical', description='Perform subtraction on only historical data', is_customizable=False,
        ),
    ]

    def compute(self):
        nodes = list(self.input_nodes)
        mults = [1.0 if i == 0 else -1.0 for i, _ in enumerate(nodes)]
        df = self.add_nodes_pl(None, nodes, node_multipliers=mults)
        only_historical = self.get_parameter_value('only_historical', required=False)
        if only_historical:
            df = df.filter(~pl.col(FORECAST_COLUMN))
        df = extend_last_historical_value_pl(df, self.get_end_year())
        return df


class SectorEmissions(AdditiveNode):
    explanation = _("This is a Sector Emissions Node. It is like Additive Node but for subsector emissions")
    # FIXME Is this needed?
    quantity = 'emissions'

    allowed_parameters = [
        *AdditiveNode.allowed_parameters,
        StringParameter(local_id='category', description='Category id for the emission sector dimension', is_customizable=False),
    ]

    def compute(self):
        val = self.get_parameter_value('category', required=False)
        if val is not None:
            df = self.get_input_dataset_pl()
            df_dims = df.dim_ids
            for dim_id in self.input_dimensions.keys():
                if dim_id not in df_dims:
                    raise NodeError(self, "Dataset doesn't have dimension %s" % dim_id)
                df_dims.remove(dim_id)
            if len(df_dims) != 1:
                raise NodeError(self, "Emission sector dimension missing")
            sector_dim = df_dims[0]
            df = df.filter(pl.col(sector_dim).eq(val))
            if not len(df):
                raise NodeError(self, "Emission sector %s not found in input data" % val)
            df = df.drop(sector_dim)
            m = self.get_default_output_metric()
            if len(df.metric_cols) != 1:
                raise NodeError(self, "Input dataset has more than 1 metric")
            df = df.rename({df.metric_cols[0]: m.column_id})
            df = extend_last_historical_value_pl(df, self.get_end_year())
            df = df.drop_nulls()
            return super().add_nodes_pl(df, self.input_nodes)

        return super().compute()


class MultiplicativeNode(SimpleNode):
    explanation = _("""This is a Multiplicative Node. It multiplies nodes together with potentially adding other input nodes.

    Multiplication and addition is determined based on the input node units.
    """)

    allowed_parameters = [
        *SimpleNode.allowed_parameters,
        BoolParameter(
            local_id='only_historical',
            description='Process only historical rows',
            is_customizable=False,
        ),
        BoolParameter(
            local_id='extend_rows',
            description='Extend last row to future years',
            is_customizable=False,
        ),
    ]
    operation_label = 'multiplication'

    def operate_pairwise(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        df = df.multiply_cols(['_Left', '_Right'], '_Left').drop('_Right')
        return df

    def perform_operation(self, nodes: Sequence[Node | None], outputs: list[ppl.PathsDataFrame]) -> ppl.PathsDataFrame:
        for n in nodes:
            if n is None:
                continue
            assert n.unit is not None
        assert self.unit is not None

        df = None
        for n, ndf in zip(nodes, outputs, strict=False):
            if df is None:
                # First output in the list
                df = ndf
                if n is not None:
                    m = n.get_default_output_metric()
                    col = m.column_id
                else:
                    assert len(df.metric_cols) == 1
                    col = df.metric_cols[0]
                df = df.rename({col: '_Left'})
                continue

            if n is not None:
                m = n.get_default_output_metric()
                col = m.column_id
            else:
                assert len(ndf.metric_cols) == 1
                col = df.metric_cols[0]

            ndf = ndf.rename({col: '_Right'})
            df = df.paths.join_over_index(ndf, how='left', index_from='union')
            df = self.operate_pairwise(df)

        assert df is not None
        df = df.rename({'_Left': VALUE_COLUMN})
        df = df.drop_nulls(VALUE_COLUMN)
        df = df.ensure_unit(VALUE_COLUMN, self.unit)
        return df

    def _compute(self, input_df: ppl.PathsDataFrame | None = None) -> ppl.PathsDataFrame:
        additive_nodes: list[Node] = []
        operation_nodes: list[Node] = []
        assert self.unit is not None
        non_additive_nodes = self.get_input_nodes(tag='non_additive')
        for node in self.input_nodes:
            if node.unit is None:
                raise NodeError(self, "Input node %s does not have a unit" % str(node))
            if node in non_additive_nodes:
                operation_nodes.append(node)
            elif self.is_compatible_unit(node.unit, self.unit):
                additive_nodes.append(node)
            else:
                operation_nodes.append(node)

        if len(operation_nodes) < 2 and input_df is None:
            raise NodeError(self, "Must receive at least two inputs to operate %s on. Now received %s."
                            % (self.operation_label, [node.id for node in operation_nodes]))

        outputs: list[ppl.PathsDataFrame] = []
        for idx, n in enumerate(operation_nodes):
            ndf = n.get_output_pl(target_node=self)
            if self.debug:
                print('%s: %s input from node %d (%s):' % (self.operation_label, self.id, idx, str(n)))
                print(ndf)
            outputs.append(ndf)

        if outputs:
            df = self.perform_operation(operation_nodes, outputs)
            if input_df is not None:
                input_df = input_df.rename({VALUE_COLUMN: '_InputSum'})
                assert input_df.dim_ids == df.dim_ids
                df = df.paths.join_over_index(input_df)
                df = df.ensure_unit('_InputSum', df.get_unit(VALUE_COLUMN))
                df = df.with_columns((pl.col(VALUE_COLUMN) + pl.col('_InputSum')).alias(VALUE_COLUMN)).drop('_InputSum')
        else:
            assert input_df is not None
            df = input_df

        if self.get_parameter_value('only_historical', required=False):
            outputs = [df.filter(~pl.col(FORECAST_COLUMN)) for df in outputs]

        if self.get_parameter_value('extend_rows', required=False):
            df = extend_last_historical_value_pl(df, self.get_end_year())

        df = self.add_nodes_pl(df, additive_nodes)
        fill_gaps = self.get_parameter_value('fill_gaps_using_input_dataset', required=False)
        if fill_gaps:
            df = self.fill_gaps_using_input_dataset_pl(df)
        replace_output = self.get_parameter_value('replace_output_using_input_dataset', required=False)
        if replace_output:
            df = self.replace_output_using_input_dataset_pl(df)
        df = self.replace_nans(df)
        if self.debug:
            print('%s: Output:' % str(self))
            self.print(df)

        return df

    def compute(self) -> ppl.PathsDataFrame:
        return self._compute()



class DivisiveNode(MultiplicativeNode):
    explanation = _("""This is a Divisive Node. It divides two nodes together with potentially adding other input nodes.

    Division and addition is determined based on the input node units.
    """)  # FIXME Is this needed as we have edge process geometric_inverse

    operation_label = 'division'

    def operate_pairwise(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        df = df.divide_cols(['_Left', '_Right'], '_Left').drop('_Right')
        return df


class EmissionFactorActivity(MultiplicativeNode):  # FIXME Does not work with Tampere/other_electricity_consumption_emisisons
    explanation = _("""This is an Emission Factor Activity Node. It multiplies an activity by an emission factor.""")
    # FIXME Do we need a separate node class?
    quantity = 'emissions'
    default_unit = '%s/a' % EMISSION_UNIT
    allowed_parameters = [
        *MultiplicativeNode.allowed_parameters,
        BoolParameter(local_id='convert_missing_values_to_zero'),
    ]

    def _get_dataset_emissions(self) -> None | ppl.PathsDataFrame:
        edfs = self.get_input_datasets_pl(tag='emissions')
        ds_list = self.get_input_datasets_pl(exclude_tags=['emissions'])
        if not ds_list:
            return None
        efdf = None  # emission factors
        adf = None   # activity
        for ds in list(ds_list):
            if 'emission_factor' in ds.metric_cols:
                assert efdf is None
                efdf = ds
            else:
                assert adf is None
                adf = ds

        if efdf is None or adf is None:
            raise NodeError(self, "Missing either emission factor or activity datasets")

        a_metric = adf.metric_cols[0]

        df = adf.paths.join_over_index(efdf, how='outer', index_from='union')
        df = df.multiply_cols([a_metric, 'emission_factor'], 'Emissions').with_columns(pl.col('Emissions').fill_null(0.0))
        df = df.select_metrics(['Emissions'])

        if 'greenhouse_gases' in df.dim_ids:
            df = convert_to_co2e(df, 'greenhouse_gases')
        output_dims = set(self.output_dimensions.keys())
        df_dims = set(df.dim_ids)
        sum_dims = df_dims - output_dims
        if sum_dims:
            df = df.paths.sum_over_dims(list(sum_dims))

        m = self.get_default_output_metric()
        df = df.rename({'Emissions': m.column_id}).ensure_unit(m.column_id, m.unit)

        for edf in edfs:
            edf = edf.rename({edf.metric_cols[0]: '_Right'}).ensure_unit('_Right', m.unit)  # noqa: PLW2901
            df = df.paths.join_over_index(edf, how='outer', index_from='union')
            df = df.with_columns((pl.col(m.column_id).fill_null(0.0) + pl.col('_Right').fill_null(0.0)).alias(m.column_id)).drop('_Right')

        return df

    def compute(self) -> ppl.PathsDataFrame:
        input_df = self._get_dataset_emissions()
        convert = self.get_parameter_value('convert_missing_values_to_zero', required=False)
        df = super()._compute(input_df)
        if convert:
            df = df.with_columns(pl.col(VALUE_COLUMN).fill_nan(pl.lit(0)))
            df = df.with_columns(pl.col(VALUE_COLUMN).fill_null(pl.lit(0)))
        return df


class PerCapitaActivity(MultiplicativeNode):
    pass


class Activity(AdditiveNode):
    explanation = _("""This is Activity Node. It adds activity amounts together.""")
    pass


class FixedMultiplierNode(SimpleNode):  # FIXME Convert to a generic parameter instead.
    explanation = _("""This is a Fixed Multiplier Node. It multiplies a single input node with a parameter.""")
    allowed_parameters = [
        *SimpleNode.allowed_parameters,
        NumberParameter(local_id='multiplier'),
        StringParameter(local_id='global_multiplier'),
    ]

    def compute(self) -> ppl.PathsDataFrame:
        if len(self.input_nodes) != 1:
            raise NodeError(self, 'FixedMultiplier needs exactly one input node')

        node = self.input_nodes[0]

        df = node.get_output_pl(target_node=self)
        multiplier_param = self.get_parameter('multiplier')  # FIXME Use get_parameter_value() instead.
        multiplier = multiplier_param.get()
        if multiplier_param.has_unit():
            m_unit = multiplier_param.get_unit()
        else:
            m_unit = self.context.unit_registry.parse_units('dimensionless')

        meta = df.get_meta()
        exprs = [pl.col(col) * multiplier for col in meta.metric_cols]
        units = {col: meta.units[col] * m_unit for col in meta.metric_cols}
        df = df.with_columns(exprs)
        for col, unit in units.items():
            df = df.set_unit(col, unit, force=True)

        for metric in self.output_metrics.values():
            df = df.ensure_unit(metric.column_id, metric.unit)

        replace_output = self.get_parameter_value('replace_output_using_input_dataset', required=False)
        if replace_output:
            df = self.replace_output_using_input_dataset_pl(df)
        return df


class FixedMultiplierNode2(AdditiveNode):  # FIXME Merge functionalities with MultiplicativeNode
    allowed_parameters = [
        *AdditiveNode.allowed_parameters,
        NumberParameter(local_id='multiplier'),
    ]

    def compute(self) -> ppl.PathsDataFrame:  # FIXME Should we instead just use AdditiveNode first?
        if len(self.input_nodes) == 0:
            raise NodeError(self, "Node must have at least one input node.")
        nodes = self.input_nodes
        node = nodes.pop()
        df = node.get_output_pl(self)
        print(df)
        unit = df.get_unit(VALUE_COLUMN)
        df = self.add_nodes_pl(df, nodes, unit=unit)

        multiplier = self.get_parameter_value('multiplier', units=True, required=True)
        df = df.multiply_quantity(VALUE_COLUMN, multiplier)
        df = df.ensure_unit(VALUE_COLUMN, self.unit)
        return df


class MixNode(AdditiveNode):
    output_metrics = {
        MIX_QUANTITY: NodeMetric(unit='%', quantity=MIX_QUANTITY),
    }
    default_unit = '%'
    allowed_parameters = [
        *AdditiveNode.allowed_parameters,
    ]
    skip_normalize: bool = False

    def add_mix_normalized(self, df: ppl.PathsDataFrame, nodes: list[Node], over_dims: list[str] | None = None):
        df = self.add_nodes_pl(df=df, nodes=nodes)
        if len(df.metric_cols) != 1:
            raise NodeError(self, "Must have exactly one metric column")

        # Fill missing values with zeroes
        df = df.paths.to_wide()
        null_fills = [pl.col(col).fill_null(0.0) for col in df.metric_cols]
        df = df.with_columns(null_fills)
        df = df.paths.to_narrow()

        if over_dims is None:
            over_dims = df.dim_ids
        col = df.metric_cols[0]
        df = (df
            .ensure_unit(col, 'dimensionless')
        )
        if not self.skip_normalize:
            # Normalize so that all values are 0 <= x <= 1.0 and
            # the yearly sum is 1.0
            df = df.with_columns(pl.col(col).clip(0, 1))
            sdf = df.paths.sum_over_dims(over_dims).rename({col: '_YearSum'})
            df = df.paths.join_over_index(sdf)
            df = df.divide_cols([col, '_YearSum'], col).drop('_YearSum')

        df = extend_last_historical_value_pl(df, self.get_end_year())
        m = self.get_default_output_metric()
        df = df.ensure_unit(m.column_id, m.unit)
        return df

    def compute(self):
        anode = self.get_input_node(tag='activity')
        adf = anode.get_output_pl(target_node=self)
        am = anode.get_default_output_metric()
        adf = adf.paths.calculate_shares(am.column_id, '_Share')
        m = self.get_default_output_metric()
        df = adf.select_metrics(['_Share']).ensure_unit('_Share', m.unit).rename({'_Share': m.column_id})
        df = extend_last_historical_value_pl(df, self.get_end_year())
        nodes = list(self.input_nodes)
        nodes.remove(anode)
        return self.add_mix_normalized(df, nodes)


class MultiplyLastNode(MultiplicativeNode):  # FIXME Tailored class for a bit wider use. Generalize!
    explanation = _("""First add other input nodes, then multiply the output.

    Multiplication and addition is determined based on the input node units.
    """)

    operation_label = 'multiplication'

    def compute(self) -> ppl.PathsDataFrame:
        additive_nodes: list[Node] = []
        operation_nodes: list[Node] = []
        assert self.unit is not None
        non_additive_nodes = self.get_input_nodes(tag='non_additive')
        for node in self.input_nodes:
            if node in non_additive_nodes:
                operation_nodes.append(node)
            elif self.is_compatible_unit(node.unit, self.unit):
                additive_nodes.append(node)
            else:
                operation_nodes.append(node)

        df = self.get_input_dataset_pl(required=False)
        if df is not None:
            df = extend_last_historical_value_pl(df, self.get_end_year())
        df = self.add_nodes_pl(df, additive_nodes)

        outputs = [n.get_output_pl() for n in operation_nodes]
        assert len(operation_nodes) == 1  # FIXME Multiplication should be generalised to several operation nodes.

        col = VALUE_COLUMN + '_right'
        df = df.paths.join_over_index(outputs.pop(0))
        df = df.ensure_unit(col, 'dimensionless')
        df = df.with_columns([
            pl.col(col).fill_null(pl.lit(0)),
            (1 - pl.col(col)).alias('ratio'),
            ])
        df = df.multiply_cols([VALUE_COLUMN, 'ratio'], VALUE_COLUMN).drop([col, 'ratio'])
        df = df.ensure_unit(VALUE_COLUMN, self.unit)

        return df


class MultiplyLastNode2(MultiplicativeNode):  # FIXME Tailored class for a bit wider use. Generalize!
    explanation = _("""First add other input nodes, then multiply the output.

    Multiplication and addition is determined based on the input node units.
    """)

    operation_label = 'multiplication'

    def perform_operation(self, nodes: list[Node], outputs: list[ppl.PathsDataFrame]) -> ppl.PathsDataFrame:
        for n in nodes:
            assert n.unit is not None
        assert self.unit is not None

        output_unit = functools.reduce(lambda x, y: x * y, [n.unit for n in nodes])  # type: ignore
        assert output_unit is not None
#        if not self.is_compatible_unit(output_unit, self.unit):
#            raise NodeError(
#                self,
#                "Multiplying inputs must in a unit compatible with '%s' (got '%s')" % (self.unit, output_unit)
#            )

        node = nodes.pop(0)
        df = outputs.pop(0)
        m = node.get_default_output_metric()
        df = df.rename({m.column_id: '_Left'})
        for n, ndf in zip(nodes, outputs, strict=False):
            m = n.get_default_output_metric()
            ndf = ndf.rename({m.column_id: '_Right'})
            df = df.paths.join_over_index(ndf, how='left', index_from='union')
            df = df.multiply_cols(['_Left', '_Right'], '_Left').drop('_Right')

        df = df.rename({'_Left': VALUE_COLUMN})
        df = df.drop_nulls(VALUE_COLUMN)
#        df = df.ensure_unit(VALUE_COLUMN, self.unit)
        return df

    def compute(self) -> ppl.PathsDataFrame:
        additive_nodes: list[Node] = []
        operation_nodes: list[Node] = []
        assert self.unit is not None
        non_additive_nodes = self.get_input_nodes(tag='non_additive')
        for node in self.input_nodes:
            if node in non_additive_nodes:
                operation_nodes.append(node)
            elif self.is_compatible_unit(node.unit, self.unit):
                additive_nodes.append(node)
            else:
                operation_nodes.append(node)

        df_add = self.get_input_dataset_pl(required=False)
        if df_add is not None:
            df_add = extend_last_historical_value_pl(df_add, self.get_end_year())
        df_add = self.add_nodes_pl(df_add, additive_nodes)

        df_mult = [n.get_output_pl() for n in operation_nodes]
        df_mult = self.perform_operation(operation_nodes, df_mult)
        df = df_add.paths.join_over_index(df_mult, how='left', index_from='union')  # FIXME This was how='outer' but why?
        df = df.multiply_cols([VALUE_COLUMN, VALUE_COLUMN + '_right'], VALUE_COLUMN)
        df = df.drop(VALUE_COLUMN + '_right')

        return df


class ImprovementNode(MultiplicativeNode):
    explanation = _("""First does what MultiplicativeNode does, then calculates 1 - result.
    Can only be used for dimensionless content (i.e., fractions and percentages)
    """)

    def compute(self):
        if len(self.input_nodes) == 1:
            node = self.input_nodes[0]
            df = node.get_output_pl(target_node=self)
        else:
            df = super().compute()
        if not isinstance(df, ppl.PathsDataFrame):
            df = ppl.from_pandas(df)
        df = df.ensure_unit(VALUE_COLUMN, 'dimensionless')
        df = df.with_columns((pl.lit(1) - pl.col(VALUE_COLUMN)).alias(VALUE_COLUMN))

        return df


class ImprovementNode2(MultiplicativeNode):
    explanation = _("""First does what MultiplicativeNode does, then calculates 1 + result.
    Can only be used for dimensionless content (i.e., fractions and percentages)
    """)

    def compute(self):
        if len(self.input_nodes) == 1:
            node = self.input_nodes[0]
            df = node.get_output_pl(target_node=self)
        else:
            df = super().compute()
        if not isinstance(df, ppl.PathsDataFrame):
            df = ppl.from_pandas(df)
        df = df.ensure_unit(VALUE_COLUMN, 'dimensionless')
        df = df.with_columns((pl.lit(1) + pl.col(VALUE_COLUMN)).alias(VALUE_COLUMN))

        return df


class RelativeNode(AdditiveNode):
    explanation = _("""
    First like AdditiveNode, then multiply with a node with "non_additive".
    The relative node is assumed to be the relative difference R = V / N - 1,
    where V is the expected output value and N is the comparison value from
    the other input nodes. So, the output value V = (R + 1)N.
    If there is no "non-additive" node, it will behave like AdditiveNode except
    it never creates a temporary dimension Sectors.
    """)

    def compute(self) -> ppl.PathsDataFrame:
        n = self.get_input_node(tag='non_additive', required=False)
        df = super().compute()
        if n is not None:
            dfn = n.get_output_pl(target_node=self)
            if dfn.get_unit(VALUE_COLUMN).dimensionless:
                dfn = dfn.ensure_unit(VALUE_COLUMN, 'dimensionless')
            df = df.paths.join_over_index(dfn, how='outer', index_from='union')
            rn = VALUE_COLUMN + '_right'
            df = df.with_columns([pl.col(rn).fill_null(0)])
            df = df.with_columns(pl.col(rn) + pl.lit(1))
            df = df.multiply_cols([VALUE_COLUMN, rn], VALUE_COLUMN).drop(rn)
            df = df.ensure_unit(VALUE_COLUMN, self.unit)
        return df

class TrajectoryNode(SimpleNode):
    explanation = _("""
    TrajectoryNode uses select_category() to select a category from a dimension.
    """)
    allowed_parameters = [
        *SimpleNode.allowed_parameters,
        StringParameter(local_id='dimension'),
        StringParameter(local_id='category'),
        NumberParameter(local_id='category_number'),
        BoolParameter(local_id='keep_dimension'),
    ]
    def compute(self):
        df = self.get_input_dataset_pl()
        dim_id = self.get_parameter_value('dimension', required=True)
        cat_id = self.get_parameter_value('category', required=False)
        cat_no = self.get_parameter_value('category_number', units=False, required=False)
        if cat_no is not None:
            cat_no = int(cat_no)
        keep = self.get_parameter_value('keep_dimension', required=False)

        df = df.select_category(dim_id, cat_id, cat_no, keep_dimension=keep)

        df = df.ensure_unit(VALUE_COLUMN, self.unit)
        return df


class FillNewCategoryNode(AdditiveNode):
    explanation = _('This is a Fill New Category Node. It behaves like Additive Node, but in the end of computation it creates a new category such that the values along that dimension sum up to 1. The input nodes must have a dimensionless unit. The new category in an existing dimension is given as parameter "new_category" in format "dimension:category"')
    allowed_parameters = [
        *AdditiveNode.allowed_parameters,
        StringParameter(local_id='new_category'),
    ]

    def compute(self):
        category = self.get_parameter_value('new_category', required=True, units=False)
        dim, cat = category.split(':')

        df: ppl.PathsDataFrame = self.add_nodes_pl(None, self.input_nodes)
        df = df.ensure_unit(VALUE_COLUMN, 'dimensionless')

        df2 = df.paths.sum_over_dims(dim)
        df2 = df2.with_columns((pl.lit(1.0) - pl.col(VALUE_COLUMN)).alias(VALUE_COLUMN))
        df2 = df2.with_columns(pl.lit(cat).cast(pl.Categorical).alias(dim))
        df2 = df2.select(df.columns)

        df = df.paths.concat_vertical(df2)
        df = df.ensure_unit(VALUE_COLUMN, self.unit)
        if self.get_parameter_value('drop_nans', required=False):  # FIXME Not consistent with the parameter name!
            df = df.paths.to_wide()
            for col in df.metric_cols:
                df = df.filter(~pl.col(col).is_null())
            df = df.paths.to_narrow()
        return df
