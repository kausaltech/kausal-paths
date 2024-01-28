import functools
from common.perf import PerfCounter
from nodes.calc import convert_to_co2e, extend_last_historical_value_pl, nafill_all_forecast_years
from nodes.units import Unit
from params.param import Parameter, BoolParameter, NumberParameter, ParameterWithUnit, StringParameter
from typing import List, ClassVar, Sequence, Tuple, Union
import polars as pl
import pandas as pd
import pint

from common.i18n import TranslatedString
from common import polars as ppl
from .constants import FORECAST_COLUMN, MIX_QUANTITY, NODE_COLUMN, VALUE_COLUMN, YEAR_COLUMN, DEFAULT_METRIC
from .node import Node, NodeMetric
from .exceptions import NodeError


EMISSION_UNIT = 'kg'


class SimpleNode(Node):
    allowed_parameters: ClassVar[List[Parameter]] = [
        BoolParameter(
            local_id='fill_gaps_using_input_dataset',
            label=TranslatedString(en="Fill in gaps in computation using input dataset"),
            is_customizable=False
        ),
        BoolParameter(
            local_id='replace_output_using_input_dataset',
            label=TranslatedString(en="Replace output using input dataset"),
            is_customizable=False
        ),
        BoolParameter(
            local_id='drop_nulls',
            description='At the end of compute() do you want to drop nulls?',
            is_customizable=False
        )
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
                pl.col(metric_col).fill_null(pl.col(right))
            ])
        return df

    def maybe_drop_nulls(self, df):
        if self.get_parameter_value('drop_nulls', required=False):
            df = df.drop_nulls()
        return df


class AdditiveNode(SimpleNode):
    """Simple addition of inputs"""
    allowed_parameters: ClassVar[list[Parameter]] = SimpleNode.allowed_parameters + [
        StringParameter(local_id='metric', is_customizable=False),
        BoolParameter(
            local_id='inventory_only',
            description='Node represents historical (inventory) values only',
            is_customizable=False,
        ),
    ]

    def add_nodes(self, ndf: pd.DataFrame | None, nodes: List[Node], metric: str | None = None) -> pd.DataFrame:
        if ndf is not None:
            df = ppl.from_pandas(ndf)
        else:
            df = None
        out = self.add_nodes_pl(df, nodes, metric)
        return out.to_pandas()

    def compute(self):
        df = self.get_input_dataset_pl(required=False)
        metric = self.get_parameter_value('metric', required=False)
        assert self.unit is not None
        if df is not None:
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
            df = df.ensure_unit(VALUE_COLUMN, self.unit)

            if self.get_parameter_value('inventory_only', required=False):
                df = df.with_columns([pl.lit(False).alias(FORECAST_COLUMN)])
            else:
                df = extend_last_historical_value_pl(df, self.get_end_year())

        na_nodes = self.get_input_nodes(tag='non_additive')
        input_nodes = [node for node in self.input_nodes if node not in na_nodes]

        if self.get_parameter_value('fill_gaps_using_input_dataset', required=False):
            df = self.add_nodes_pl(None, input_nodes, metric)
            df = self.fill_gaps_using_input_dataset_pl(df)
        else:
            df = self.add_nodes_pl(df, input_nodes, metric)

        return df


class SubtractiveNode(Node):
    allowed_parameters = [
        BoolParameter(local_id='only_historical', description='Perform subtraction on only historical data', is_customizable=False)
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
    quantity = 'emissions'
    """Simple addition of subsector emissions"""

    allowed_parameters = AdditiveNode.allowed_parameters + [
        StringParameter(local_id='category', description='Category id for the emission sector dimension', is_customizable=False)
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
    """Multiply nodes together with potentially adding other input nodes.

    Multiplication and addition is determined based on the input node units.
    """

    allowed_parameters = SimpleNode.allowed_parameters + [
        BoolParameter(
            local_id='only_historical',
            description='Process only historical rows',
            is_customizable=False,
        ),
        BoolParameter(
            local_id='extend_rows',
            description='Extend last row to future years',
            is_customizable=False,
        )
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
        for n, ndf in zip(nodes, outputs):
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

    def _compute(self, input_df: ppl.PathsDataFrame | None = None):
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
            raise NodeError(self, "Must receive at least two inputs to operate %s on" % self.operation_label)

        outputs = [n.get_output_pl(target_node=self) for n in operation_nodes]

        if self.debug:
            for idx, (n, df) in enumerate(zip(operation_nodes, outputs)):
                print('%s: %s input from node %d (%s):' % (self.operation_label, self.id, idx, n.id))

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
        if self.debug:
            print('%s: Output:' % self.id)
            self.print(df)

        return df

    def compute(self) -> ppl.PathsDataFrame:
        return self._compute()



class DivisiveNode(MultiplicativeNode):
    """Divide two nodes together with potentially adding other input nodes.

    Division and addition is determined based on the input node units.
    """

    operation_label = 'division'

    def operate_pairwise(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        df = df.divide_cols(['_Left', '_Right'], '_Left').drop('_Right')
        return df


class EmissionFactorActivity(MultiplicativeNode):  # FIXME Does not work with Tampere/other_electricity_consumption_emisisons
    """Multiply an activity by an emission factor."""
    quantity = 'emissions'
    default_unit = '%s/a' % EMISSION_UNIT
    allowed_parameters = MultiplicativeNode.allowed_parameters + [
        BoolParameter(local_id='convert_missing_values_to_zero')
    ]

    def _get_dataset_emissions(self):
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
            edf = edf.rename({edf.metric_cols[0]: '_Right'}).ensure_unit('_Right', m.unit)
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
    """Add activity amounts together."""
    pass


class FixedMultiplierNode(SimpleNode):  # FIXME Merge functionalities with MultiplicativeNode
    allowed_parameters = [
        NumberParameter(local_id='multiplier'),
        StringParameter(local_id='global_multiplier'),
    ] + SimpleNode.allowed_parameters

    def compute(self) -> ppl.PathsDataFrame:
        if len(self.input_nodes) != 1:
            raise NodeError(self, 'FixedMultiplier needs exactly one input node')

        node = self.input_nodes[0]

        df = node.get_output_pl(target_node=self)
        multiplier_param = self.get_parameter('multiplier')
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
        NumberParameter(local_id='multiplier'),
        StringParameter(local_id='global_multiplier'),
    ] + AdditiveNode.allowed_parameters

    def compute(self) -> ppl.PathsDataFrame:
        df = super().compute()
        multiplier = self.get_parameter_value('multiplier', required=True)

        # FIXME Use actual function from branch clean-nodes and  update it for unitless quantities
        def multiply_quantity(df, col: str, quantity, out_unit: Unit | None = None):
            if not isinstance(quantity, float):
                res_unit = df._units[col] * quantity.units
                quant = quantity.m
            else:
                res_unit = df._units[col]
                quant = quantity
            df = df.with_columns([pl.col(col) * pl.lit(quant)])
            df._units[col] = res_unit
            if out_unit:
                df = df.ensure_unit(col, out_unit)
            return df

        df = multiply_quantity(df, VALUE_COLUMN, multiplier)

        return df


class MixNode(AdditiveNode):
    output_metrics = {
        MIX_QUANTITY: NodeMetric(unit='%', quantity=MIX_QUANTITY)
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
    """First add other input nodes, then multiply the output.

    Multiplication and addition is determined based on the input node units.
    """

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
            (1 - pl.col(col)).alias('ratio')
            ])
        df = df.multiply_cols([VALUE_COLUMN, 'ratio'], VALUE_COLUMN).drop([col, 'ratio'])
        df = df.ensure_unit(VALUE_COLUMN, self.unit)

        return df


class MultiplyLastNode2(MultiplicativeNode):  # FIXME Tailored class for a bit wider use. Generalize!
    """First add other input nodes, then multiply the output.

    Multiplication and addition is determined based on the input node units.
    """

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
        for n, ndf in zip(nodes, outputs):
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
    '''First does what MultiplicativeNode does, then calculates 1 - result.
    Can only be used for dimensionless content (i.e., fractions and percentages)
    '''

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
    '''First does what MultiplicativeNode does, then calculates 1 + result.
    Can only be used for dimensionless content (i.e., fractions and percentages)
    '''

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
    '''
    First like AdditiveNode, then multiply with a node with "non_additive".
    The relative node is assumed to be the relative difference R = V / N - 1,
    where V is the expected output value and N is the comparison value from
    the other input nodes. So, the output value V = (R + 1)N.
    If there is no "non-additive" node, it will behave like AdditiveNode except
    it never creates a temporary dimension Sectors.
    '''

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
            df = df.with_columns((pl.col(rn) + pl.lit(1)))
            df = df.multiply_cols([VALUE_COLUMN, rn], VALUE_COLUMN).drop(rn)
            df = df.ensure_unit(VALUE_COLUMN, self.unit)
        return df

class TrajectoryNode(SimpleNode):
    '''
    TrajectoryNode uses select_category() to select a category from a dimension.
    '''
    allowed_parameters = SimpleNode.allowed_parameters + [
        StringParameter(local_id='dimension'),
        StringParameter(local_id='category'),
        NumberParameter(local_id='category_number'),
        BoolParameter(local_id='keep_dimension')
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
