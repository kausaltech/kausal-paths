import functools

import pandas as pd
import polars as pl
from nodes.units import Unit, unit_registry

from nodes.node import NodeMetric, Node, NodeError
import common.polars as ppl
from params.param import NumberParameter, StringParameter, BoolParameter
from params.utils import sep_unit, sep_unit_pt
from .constants import DEFAULT_METRIC, FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from .simple import AdditiveNode, SimpleNode


class SelectiveNode(AdditiveNode):
    global_parameters: list[str] = [
        'include_co2', 'include_health', 'include_el_avoided',
    ]

    def compute(self) -> ppl.PathsDataFrame:
        # Global parameters
        include_co2 = self.get_global_parameter_value('include_co2')
        include_health = self.get_global_parameter_value('include_health')
        include_el_avoided = self.get_global_parameter_value('include_el_avoided')

        # Input nodes
        nodes = self.input_nodes
        out = None
        included_nodes: list[Node] = []

        for node in nodes:
            df = node.get_output_pl(target_node=self)
            if 'co2_cost' in node.tags:
                if include_co2:
                    included_nodes.append(node)
            elif 'capacity_cost' in node.tags:
                if include_el_avoided:
                    included_nodes.append(node)
            elif 'health_cost' in node.tags:
                if include_health:
                    included_nodes.append(node)
            else:
                included_nodes.append(node)

        assert len(included_nodes)
        output_unit = self.output_metrics[DEFAULT_METRIC].unit
        for node in included_nodes:
            df = node.get_output_pl(target_node=self).ensure_unit(VALUE_COLUMN, output_unit)
            if out is None:
                out = df
            else:
                out = out.paths.add_with_dims(df, how='outer')
        assert out is not None
        return out


class ExponentialNode(AdditiveNode):
    '''
    Takes in either input nodes as AdditiveNode, or builds a dataframe from current_value.
    Builds an exponential multiplier based on annual_change and multiplies the VALUE_COLUMN.
    Optionally, touches also historical values.
    Parameter is_decreasing_rate is used to give discount rates instead.
    '''
    allowed_parameters = AdditiveNode.allowed_parameters + [
        NumberParameter(
            local_id='current_value',
            is_customizable=True,
        ),
        NumberParameter(
            local_id='annual_change',
            is_customizable=True,
        ),
        BoolParameter(
            local_id='is_decreasing_rate',
            is_customizable=True,
            value=False
        ),
        BoolParameter(
            local_id='touch_historical_values',
            is_customizable=True,
            value=False
        )
    ]

    def compute(self):
        current_value = self.get_parameter_value('current_value', required=False, units=True)

        if current_value is not None:
            if len(self.input_nodes) > 0:
                raise NodeError(self, 'You must give either input node(s) or parameter current_value but not both.')

            unit = current_value.units
            current_value = current_value.m
            start_year = self.context.instance.minimum_historical_year
            current_year = self.context.instance.maximum_historical_year
            end_year = self.context.instance.model_end_year

            df = pl.DataFrame({
                YEAR_COLUMN: range(start_year, end_year + 1)
            })
            df = df.with_columns([
                pl.when(pl.col(YEAR_COLUMN) > pl.lit(current_year))
                .then(pl.lit(True)).otherwise(pl.lit(False)).alias(FORECAST_COLUMN),
                pl.lit(current_value).alias(VALUE_COLUMN)
            ])
            meta = ppl.DataFrameMeta(units={VALUE_COLUMN: unit}, primary_keys=[YEAR_COLUMN])
            df = ppl.to_ppdf(df, meta=meta)

        else:
            df = super().compute()
            current_year = self.get_last_historical_year()
            if current_year is None:
                current_year = df[YEAR_COLUMN].min() - 1

        annual_change = self.get_parameter_value('annual_change', required=True, units=True)
        base_value = 1 + annual_change.to('dimensionless').m

        if self.get_parameter_value('is_decreasing_rate', required=False):
            base_value = 1 / base_value
        
        df = df.with_columns(
            (pl.col(YEAR_COLUMN) - pl.lit(current_year)).alias('power')
        )
        if not self.get_parameter_value('touch_historical_values', required=False):
            df = df.with_columns(
                pl.when(pl.col('power') < pl.lit(0)).then(pl.lit(0))
                .otherwise(pl.col('power')).alias('power')
            )

        df = df.with_columns(
            (pl.lit(base_value) ** pl.col('power') * pl.col(VALUE_COLUMN)).alias(VALUE_COLUMN)
        ).drop('power')

        return df
    

class InternalGrowthNode(ExponentialNode):
    '''
    Calculates internal growth of e.g. a forest, accounting for forest cuts. Takes in additive and
    non-additive nodes and a dataset.
    Parameter annual_change is used where the rate node(s) have null values.
    '''
#    output_metrics = {  # FIXME Causes an unknown error but not critical yet.
#        DEFAULT_METRIC: NodeMetric('Mt/a', 'mass', column_id=VALUE_COLUMN),
#        'Reductions': NodeMetric('Mt/a', 'mass', column_id='Reductions'),
#        'Annual_change': NodeMetric('', 'fraction', 'base_value')
#    }

    def compute(self):
        df = self.get_input_dataset_pl()
        current_value = self.get_parameter_value('current_value', required=False)  # FIXME Not used

        nonadd = self.get_input_nodes(tag='non_additive')
        if nonadd:
            df_nonadd = self.add_nodes(None, nonadd)  # FIXME Not used
        annual_change = self.get_parameter_value('annual_change', units=True, required=False)
        base_value = 1 + annual_change.to('dimensionless').m

        add = [node for node in self.input_nodes if node not in nonadd]
        df_add = self.add_nodes_pl(None, add, unit=self.input_nodes[0].unit)
        df_add = df_add.ensure_unit(VALUE_COLUMN, self.unit * unit_registry.parse_units('1/a'))
        df = df.paths.join_over_index(df_add, how='outer')

        df = df.with_columns([  # FIXME Fails if there are missing years
            pl.lit(base_value).alias('base_value'),
            pl.col('Value_right').shift(1).alias('Value_right'),
            pl.col('Value').shift(1).alias('Previous')
            ])
        for year in df[YEAR_COLUMN]:  # FIXME Fails if multiple dimensions
            if df.filter(pl.col(YEAR_COLUMN) == year)[VALUE_COLUMN].is_null()[0]:
                df = df.with_columns([
                    pl.when(pl.col(YEAR_COLUMN).eq(year))
                    .then((pl.col('Previous') + pl.col('Value_right')) * pl.col('base_value'))
                    .otherwise(pl.col('Value')).alias('Value')])
                df = df.with_columns(
                    pl.when(pl.col(YEAR_COLUMN).eq(year+1)).then(pl.col('Value').shift(1))
                    .otherwise(pl.col('Previous')).alias('Previous'))

#        df = df.drop(['Previous', 'Value_right', 'base_value'])
#        df = df.rename({'Value_right': 'Reductions'})
        return df


class CumulativeNode(AdditiveNode):
    def compute(self):
        unit = self.unit
        self.unit = unit * unit_registry.parse_units('1/a')
        df = super().compute()
        self.unit = unit
        df = df.cumulate(VALUE_COLUMN)
        unit = df.get_unit(VALUE_COLUMN) * unit_registry.parse_units('1 a') # FIXME Should this happen inside cumulate()?
        df = df.set_unit(VALUE_COLUMN, unit, force=True)

        return df


class EnergyCostNode(AdditiveNode):
    output_metrics = {
        VALUE_COLUMN: NodeMetric('SEK/kWh', 'currency'),
        #'EnergyPrice': NodeMetric('SEK/kWh', 'currency'),
        #'AddedValueTax': NodeMetric('SEK/kWh', 'currency'),
        #'NetworkPrice': NodeMetric('SEK/kWh', 'currency'),
        #'HandlingFee': NodeMetric('SEK/kWh', 'currency'),
        #'Certificate': NodeMetric('SEK/kWh', 'currency'),
        #'EnergyTax': NodeMetric('SEK/kWh', 'currency')
    }
    global_parameters: list[str] = ['include_energy_taxes']
    allowed_parameters = AdditiveNode.allowed_parameters + [
        NumberParameter(
            local_id='added_value_tax',
            label='Added value tax (%)',
            unit_str='%',
            is_customizable=False
        ),
        NumberParameter(
            local_id='network_price',
            label='Network price (SEK/kWh)',
            unit_str='SEK/kWh',
            is_customizable=False
        ),
        NumberParameter(
            local_id='handling_fee',
            label='Handling fee (SEK/kWh)',
            unit_str='SEK/kWh',
            is_customizable=False
        ),
        NumberParameter(
            local_id='certificate',
            label='Certificate (SEK/kWh)',
            unit_str='SEK/kWh',
            is_customizable=False
        ),
        NumberParameter(
            local_id='energy_tax',
            label='Energy tax (SEK/kWh)',
            unit_str='SEK/kWh',
            is_customizable=False
        ),
    ]

    def compute(self):
        added_value_tax = self.get_parameter_value('added_value_tax', units=True).to('dimensionless').m
        output_unit = self.output_metrics[VALUE_COLUMN].unit
        network_price, net_pt = sep_unit(self.get_parameter_value('network_price', units=True), output_unit)
        handling_fee, han_pt = sep_unit(self.get_parameter_value('handling_fee', units=True), output_unit)
        certificate, cer_pt = sep_unit(self.get_parameter_value('certificate', units=True), output_unit)
        energy_tax, ene_pt = sep_unit(self.get_parameter_value('energy_tax', units=True), output_unit)
        include_energy_taxes = self.get_global_parameter_value('include_energy_taxes')

        metric = self.get_parameter_value('metric', required=False)
        if self.get_parameter_value('fill_gaps_using_input_dataset', required=False):
            df = self.add_nodes_pl(None, self.input_nodes, metric)
            df = self.fill_gaps_using_input_dataset_pl(df)
        else:
            df = self.add_nodes_pl(None, self.input_nodes, metric)

        meta = df.get_meta()
        df = df.with_columns([
            pl.col(VALUE_COLUMN).alias('EnergyPrice'),
            (pl.col(VALUE_COLUMN) * added_value_tax).alias('AddedValueTax'),
            pl.lit(network_price).alias('NetworkPrice'),
            pl.lit(handling_fee).alias('HandlingFee'),
            pl.lit(certificate).alias('Certificate'),
            pl.lit(energy_tax).alias('EnergyTax'),
        ])
        meta.units.update(dict(
            EnergyPrice=meta.units[VALUE_COLUMN],
            AddedValueTax=meta.units[VALUE_COLUMN],
            NetworkPrice=net_pt,
            HandlingFee=han_pt,
            Certificate=cer_pt,
            EnergyTax=ene_pt,
        ))
        df = ppl.to_ppdf(df=df, meta=meta)

        if include_energy_taxes:
            cols = ['AddedValueTax', 'NetworkPrice', 'HandlingFee', 'Certificate', 'EnergyTax']
        else:
            cols = ['NetworkPrice']

        add_expr = functools.reduce(lambda x, y: x + y, [pl.col(x) for x in cols])
        df = df.with_columns([
            (pl.col(VALUE_COLUMN) + add_expr).alias(VALUE_COLUMN)
        ])
        return df
