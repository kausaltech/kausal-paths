import pandas as pd
import polars as pl
from pint_pandas import PintType

import common.polars as ppl
from nodes.calc import extend_last_historical_value, extend_last_historical_value_pl
from nodes.node import NodeMetric, NodeError, Node
from nodes.simple import AdditiveNode
from nodes.constants import DEFAULT_METRIC, EMISSION_FACTOR_QUANTITY, EMISSION_QUANTITY, ENERGY_QUANTITY, FORECAST_COLUMN, POPULATION_QUANTITY, VALUE_COLUMN, YEAR_COLUMN


class BuildingEnergy(AdditiveNode):
    output_metrics = {
        ENERGY_QUANTITY: NodeMetric(unit='GWh/a', quantity=ENERGY_QUANTITY)
    }
    output_dimension_ids = [
        'energy_carrier',
    ]
    input_dimension_ids = [
        'energy_carrier',
    ]

    def compute(self) -> pd.DataFrame:
        df = self.get_input_dataset()

        ec_dim = self.output_dimensions['energy_carrier']
        df[ec_dim.id] = ec_dim.series_to_ids(df['energy_carrier'])
        df[YEAR_COLUMN] = df['year']
        df[ENERGY_QUANTITY] = df['energy'].astype('pint[GWh/a]')
        df = df.set_index([YEAR_COLUMN, ec_dim.id])
        df[FORECAST_COLUMN] = False
        df = df[[ENERGY_QUANTITY, FORECAST_COLUMN]]
        df = df.rename(columns={ENERGY_QUANTITY: VALUE_COLUMN})

        df = extend_last_historical_value(df, self.get_end_year())

        return df


class ElectricityEmissionFactor(AdditiveNode):
    output_metrics = {
        EMISSION_FACTOR_QUANTITY: NodeMetric(unit='g/kWh', quantity=EMISSION_FACTOR_QUANTITY)
    }
    #output_dimension_ids = [
    #    'electricity_source',
    #]
    input_dimension_ids = [
        'electricity_source',
    ]
    default_unit = 'g/kWh'

    def compute(self) -> pd.DataFrame:
        dfs = self.get_input_datasets()
        mix_df = None
        ef_df = None
        for df in dfs:
            if 'share' in df.columns:
                mix_df = df
            elif 'emission_factor' in df.columns:
                ef_df = df
        if mix_df is None:
            raise NodeError(self, "Electricity mix dataset not supplied")
        if ef_df is None:
            raise NodeError(self, "Emission factor dataset not supplied")

        es_dim = self.input_dimensions['electricity_source']

        mix_df[es_dim.id] = es_dim.series_to_ids(mix_df['electricity_source'])
        mix_df[YEAR_COLUMN] = mix_df.pop('year')
        mix_df = mix_df.set_index([YEAR_COLUMN, es_dim.id])

        ef_df[es_dim.id] = es_dim.series_to_ids(ef_df['electricity_source'])
        ef_df[YEAR_COLUMN] = ef_df.pop('year')
        ef_df = ef_df.set_index([YEAR_COLUMN, es_dim.id])

        df = ef_df
        df['Share'] = mix_df['share'].astype('pint[dimensionless]')
        df['EF'] = (df['Share'] * df['emission_factor']).fillna(0)
        s = df['EF'].unstack(es_dim.id).sum(axis=1)
        s = s.astype(PintType(self.unit))
        df = pd.DataFrame(data=s, index=s.index, columns=[VALUE_COLUMN])
        df[FORECAST_COLUMN] = False

        df = extend_last_historical_value(df, self.get_end_year())

        return df


class EmissionFactor(Node):
    input_dimension_ids = ['energy_carrier']
    output_dimension_ids = ['energy_carrier']

    def compute(self) -> ppl.PathsDataFrame:
        df = ppl.from_pandas(self.get_input_dataset())
        meta = df.get_meta()

        metric_cols = list(meta.units.keys())
        if len(metric_cols) == 1:
            metric_col = metric_cols[0]
        else:
            metric_col = 'emission_factor'

        dim = self.input_dimensions['energy_carrier']
        ids = dim.series_to_ids_pl(df[dim.id])
        df = ppl.to_ppdf(df.with_column(ids.alias(dim.id)), meta=meta)\
            .with_column(pl.lit(False).alias(FORECAST_COLUMN))\
            .with_column(pl.col(dim.id).cast(str))

        df = df.rename({metric_col: VALUE_COLUMN})
        meta = df.get_meta()
        if dim.id not in meta.primary_keys:
            meta.primary_keys.append(dim.id)
        if YEAR_COLUMN not in meta.primary_keys:
            meta.primary_keys.append(YEAR_COLUMN)
        for node in self.input_nodes:
            ndf = node.get_output_pl(self)
            ndf = ndf.ensure_unit(VALUE_COLUMN, meta.units[VALUE_COLUMN])
            ndf = ndf.select(df.columns)
            df = ppl.to_ppdf(pl.concat([df, ndf], how='vertical'), meta=meta)

        counts = df.groupby([YEAR_COLUMN, dim.id]).count()
        duplicates = counts.filter(pl.col('count') > 1)
        if len(duplicates):
            raise NodeError(self, "Duplicate rows detected")

        df = extend_last_historical_value_pl(df, self.get_end_year())
        return df


class EmissionFactorActivity(Node):
    output_metrics = {
        DEFAULT_METRIC: NodeMetric('kt/a', quantity=EMISSION_QUANTITY, column_id=VALUE_COLUMN),
    }
    input_dimension_ids = ['energy_carrier']

    def compute(self) -> ppl.PathsDataFrame:
        en = self.get_input_node(quantity=ENERGY_QUANTITY)
        fn = self.get_input_node(quantity=EMISSION_FACTOR_QUANTITY)
        edf = ppl.from_pandas(en.get_output(self))
        edf = edf.rename({VALUE_COLUMN: 'Energy'})
        fdf = ppl.from_pandas(fn.get_output(self))
        fdf = fdf.rename({VALUE_COLUMN: 'EF'})
        dim_cols = list(self.input_dimensions.keys())
        emdf = edf.join(fdf, on=[YEAR_COLUMN, *dim_cols], how='left')
        if emdf['EF'].has_validity():
            raise NodeError(self, "Emission factor not found for some categories")
        df = ppl.to_ppdf(emdf, edf.get_meta())
        df = df.set_unit('EF', fdf.get_unit('EF'))
        em = pl.col('Energy') * pl.col('EF')
        em_unit = df.get_unit('EF') * df.get_unit('Energy')
        df = df.with_column(em.alias('Emissions'), unit=em_unit)
        output_unit = self.output_metrics[DEFAULT_METRIC].unit
        df = df.ensure_unit('Emissions', output_unit)
        meta = df.get_meta()
        if YEAR_COLUMN not in meta.primary_keys:
            meta.primary_keys.append(YEAR_COLUMN)
        df = df.groupby([YEAR_COLUMN]).agg([pl.sum('Emissions').alias(VALUE_COLUMN), pl.first(FORECAST_COLUMN)]).sort(YEAR_COLUMN)
        df = ppl.to_ppdf(df, meta=meta)
        df = df.set_unit(VALUE_COLUMN, output_unit)
        df = extend_last_historical_value_pl(df, self.context.model_end_year)
        return df


class ToPerCapita(Node):
    def compute(self) -> ppl.PathsDataFrame:
        input_nodes = list(self.input_nodes)
        pop_node = self.get_input_node(quantity=POPULATION_QUANTITY)
        input_nodes.remove(pop_node)
        if len(input_nodes) > 1:
            act_node = self.get_input_node(tag='activity')
        else:
            act_node = input_nodes[0]
        input_nodes.remove(act_node)

        pop_df = ppl.from_pandas(pop_node.get_output(self))
        pop_df = pop_df.rename({VALUE_COLUMN: 'Pop'})
        act_df = ppl.from_pandas(act_node.get_output(self))

        meta = act_df.get_meta()
        df = ppl.to_ppdf(act_df.join(pop_df, on=YEAR_COLUMN, how='left'), meta=meta)

        pc_unit = act_df.get_unit('Value') / pop_df.get_unit('Pop')
        df = df.with_column((pl.col(VALUE_COLUMN) / pl.col('Pop')).alias('PerCapita'))
        df = df.with_column((pl.col(FORECAST_COLUMN) | pl.col(FORECAST_COLUMN + '_right').alias(FORECAST_COLUMN)))
        df = df.set_unit('PerCapita', pc_unit)
        output_unit = self.output_metrics[DEFAULT_METRIC].unit
        df = df.ensure_unit('PerCapita', output_unit)
        df = df.drop(VALUE_COLUMN).rename(dict(PerCapita=VALUE_COLUMN))
        meta = df.get_meta()
        df = df.select([YEAR_COLUMN, VALUE_COLUMN, FORECAST_COLUMN])
        for node in input_nodes:
            ndf = ppl.from_pandas(node.get_output(self))
            ndf = ndf.ensure_unit(VALUE_COLUMN, output_unit)
            df = ppl.to_ppdf(df.join(ndf, on=YEAR_COLUMN, how='left'), meta=meta)
            other = df[VALUE_COLUMN + '_right'].fill_null(0)
            df = df.with_column(pl.col(VALUE_COLUMN) + other)
            df = df.with_column(pl.col(FORECAST_COLUMN) | pl.col(FORECAST_COLUMN + '_right').fill_null(False))
            df = df.select([YEAR_COLUMN, VALUE_COLUMN, FORECAST_COLUMN])
        df = ppl.to_ppdf(df, meta=meta)
        return df
