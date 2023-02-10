import pandas as pd
import polars as pl
from pint_pandas import PintType

import common.polars as ppl
from nodes.calc import extend_last_historical_value, extend_last_historical_value_pl
from nodes.node import NodeMetric, NodeError, Node
from nodes.simple import AdditiveNode
from nodes.constants import DEFAULT_METRIC, EMISSION_FACTOR_QUANTITY, EMISSION_QUANTITY, ENERGY_QUANTITY, FORECAST_COLUMN, POPULATION_QUANTITY, VALUE_COLUMN, YEAR_COLUMN, MILEAGE_QUANTITY


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

    def compute(self) -> ppl.PathsDataFrame:
        df = self.get_input_dataset_pl()

        ec_dim = self.output_dimensions['energy_carrier']
        df = df.with_columns([ec_dim.series_to_ids_pl(df['energy_carrier'])])
        meta = df.get_meta()
        metric_ids = meta.metric_cols
        if len(metric_ids) == 1:
            col = metric_ids[0]
        else:
            col = ENERGY_QUANTITY
            assert col in df.columns

        m = self.output_metrics[ENERGY_QUANTITY]
        output_unit = m.unit

        df = df.ensure_unit(col, output_unit)
        df = df.with_columns([
            pl.col(col).alias(VALUE_COLUMN),
            pl.lit(False).alias(FORECAST_COLUMN)
        ])
        df = df.select([YEAR_COLUMN, *meta.dim_ids, VALUE_COLUMN, FORECAST_COLUMN])
        # df = df.set_unit(VALUE_COLUMN, output_unit)

        df = extend_last_historical_value_pl(df, self.get_end_year())

        for node in self.input_nodes:
            ndf = node.get_output_pl(self)
            df = df.paths.add_with_dims(ndf)

        return df


class EnergyProductionEmissionFactor(AdditiveNode):
    output_metrics = {
        EMISSION_FACTOR_QUANTITY: NodeMetric(unit='g/kWh', quantity=EMISSION_FACTOR_QUANTITY)
    }
    #output_dimension_ids = [
    #    'electricity_source',
    #]
    #input_dimension_ids = [
    #    'electricity_source',
    #]
    default_unit = 'g/kWh'

    def compute(self) -> ppl.PathsDataFrame:
        dfs = self.get_input_datasets_pl()
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

        if len(self.input_dimensions) != 1:
            raise NodeError(self, "Must have exactly 1 input dimensions (%d given)" % len(self.input_dimensions))

        dim_id = list(self.input_dimensions.keys())[0]
        es_dim = self.input_dimensions[dim_id]

        mix_df = mix_df.with_columns([es_dim.series_to_ids_pl(mix_df[dim_id])])
        mix_df = mix_df.ensure_unit('share', self.context.unit_registry.parse_units('dimensionless'))
        ef_df = ef_df.with_columns([es_dim.series_to_ids_pl(ef_df[dim_id])])

        df = ef_df.paths.join_over_index(mix_df)
        df = df.multiply_cols(['share', 'emission_factor'], 'EF')
        df = df.with_columns([pl.col('EF').fill_null(0)])
        meta = df.get_meta()
        zdf = df.groupby(YEAR_COLUMN).agg(pl.sum('EF')).sort(YEAR_COLUMN)
        df = ppl.to_ppdf(zdf, meta=meta)
        df = df.rename(dict(EF=VALUE_COLUMN))
        df = df.with_columns([pl.lit(False).alias(FORECAST_COLUMN)])
        df = extend_last_historical_value_pl(df, self.get_end_year())

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
        ids = dim.series_to_ids_pl(df[dim.id]).cast(pl.Utf8)
        df = df.with_columns([
            ids.alias(dim.id).cast(str),
            pl.lit(False).alias(FORECAST_COLUMN),
        ])

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
    # input_dimension_ids = ['energy_carrier']

    def compute(self) -> ppl.PathsDataFrame:
        en = self.get_input_node(quantity=ENERGY_QUANTITY)
        fn = self.get_input_node(quantity=EMISSION_FACTOR_QUANTITY)
        with pl.StringCache():
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
        df = df.with_columns([em.alias('Emissions')], units=dict(Emissions=em_unit))
        output_unit = self.output_metrics[DEFAULT_METRIC].unit
        df = df.ensure_unit('Emissions', output_unit)
        meta = df.get_meta()
        if YEAR_COLUMN not in meta.primary_keys:
            meta.primary_keys.append(YEAR_COLUMN)
        zdf = df.groupby([YEAR_COLUMN]).agg([pl.sum('Emissions').alias(VALUE_COLUMN), pl.first(FORECAST_COLUMN)]).sort(YEAR_COLUMN)
        df = ppl.to_ppdf(zdf, meta=meta)
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
        df = df.with_columns([
            (pl.col(VALUE_COLUMN) / pl.col('Pop')).alias('PerCapita'),
            (pl.col(FORECAST_COLUMN) | pl.col(FORECAST_COLUMN + '_right')).alias(FORECAST_COLUMN)
        ])
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
            df = df.with_columns([
                pl.col(VALUE_COLUMN) + other,
                pl.col(FORECAST_COLUMN) | pl.col(FORECAST_COLUMN + '_right').fill_null(False)
            ])
            df = df.select([YEAR_COLUMN, VALUE_COLUMN, FORECAST_COLUMN])
        df = ppl.to_ppdf(df, meta=meta)
        return df


class VehicleDatasetNode(AdditiveNode):  # Based on BuildingEnergy.
    output_metrics = {
        MILEAGE_QUANTITY: NodeMetric(unit='km/a', quantity=MILEAGE_QUANTITY)
    }
    output_dimension_ids = [
        'vehicle_type',
    ]
    input_dimension_ids = [
        'vehicle_type',
    ]

    def compute(self, dimension_ids: str, quantity: str, col: str | None = None) -> ppl.PathsDataFrame:
        df = self.get_input_dataset_pl()

        for dimension_id in dimension_ids:
            ec_dim = self.output_dimensions[dimension_id]
            df = df.with_columns([ec_dim.series_to_ids_pl(df[dimension_id])])
        meta = df.get_meta()
        metric_ids = meta.metric_cols

        if col is None:
            if len(metric_ids) == 1:
                col = metric_ids[0]
            else:
                col = quantity
        assert col in df.columns

        m = self.output_metrics[quantity]
        output_unit = m.unit

        df = df.ensure_unit(col, output_unit)
        df = df.with_columns([
            pl.col(col).alias(VALUE_COLUMN),
            pl.lit(False).alias(FORECAST_COLUMN)
        ])
        df = df.select([YEAR_COLUMN, *meta.dim_ids, VALUE_COLUMN, FORECAST_COLUMN])
        # df = df.set_unit(VALUE_COLUMN, output_unit)

        df = extend_last_historical_value_pl(df, self.get_end_year())

        for node in self.input_nodes:
            ndf = node.get_output_pl(self)
            df = df.paths.add_with_dims(ndf)

        return df


class VehicleMileage(VehicleDatasetNode):  # Based on BuildingEnergy. Should be generalised at some point.
    output_metrics = {
        MILEAGE_QUANTITY: NodeMetric(unit='km/a', quantity=MILEAGE_QUANTITY)
    }
    output_dimension_ids = [
        'vehicle_type',
    ]
    input_dimension_ids = [
        'vehicle_type',
    ]

    def compute(self) -> ppl.PathsDataFrame:
        return super().compute(dimension_ids=['vehicle_type'], quantity=MILEAGE_QUANTITY)


class EmissionFactorNode(VehicleDatasetNode):  # Based on BuildingEnergy. Should be generalised at some point.
    output_metrics = {
        EMISSION_FACTOR_QUANTITY: NodeMetric(unit='kg/km', quantity=EMISSION_FACTOR_QUANTITY)
    }
    output_dimension_ids = [
        'energy_carrier', 'vehicle_type',
    ]
    input_dimension_ids = [
        'energy_carrier', 'vehicle_type',
    ]

    def compute(self) -> ppl.PathsDataFrame:
        return super().compute(dimension_ids=['energy_carrier', 'vehicle_type'], quantity=EMISSION_FACTOR_QUANTITY, col='fuel')
