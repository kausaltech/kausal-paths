import pandas as pd
import polars as pl
from pint_pandas import PintType

import common.polars as ppl
from nodes.calc import convert_to_co2e, extend_last_historical_value, extend_last_historical_value_pl
from nodes.node import NodeMetric, NodeError, Node
from nodes.simple import AdditiveNode, MultiplicativeNode
from nodes.constants import DEFAULT_METRIC, EMISSION_FACTOR_QUANTITY, EMISSION_QUANTITY, ENERGY_QUANTITY, FORECAST_COLUMN, MIX_QUANTITY, POPULATION_QUANTITY, VALUE_COLUMN, YEAR_COLUMN, MILEAGE_QUANTITY


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


class EnergyProductionMix(AdditiveNode):
    output_metrics = {
        MIX_QUANTITY: NodeMetric(unit='%', quantity=MIX_QUANTITY)
    }
    default_unit = '%'

    def add_mix_normalized(self, df: ppl.PathsDataFrame, nodes: list[Node]):
        df = self.add_nodes_pl(df=df, nodes=nodes)
        df = df.paths.to_wide()
        metric_cols = df.metric_cols
        df = (
            df.with_columns([pl.col(col).clip_min(0) for col in metric_cols])
            .with_columns([pl.sum(metric_cols).alias('YearSum')])
            .with_columns([pl.col(col) / pl.col('YearSum') for col in metric_cols]).drop('YearSum')
            .paths.to_narrow()
        )
        m = self.get_default_output_metric()
        df = df.set_unit(m.column_id, 'dimensionless', force=True).ensure_unit(m.column_id, m.unit)
        return df


class ElectricityProductionMix(EnergyProductionMix):
    def compute(self) -> ppl.PathsDataFrame:
        dfs = self.get_input_datasets_pl()
        gen_mix_df, sub_mix_df, ext_energy_df = dfs

        energy_node = self.get_input_node(tag='consumption')
        energy_m = energy_node.get_default_output_metric()
        df = energy_node.get_output_pl(target_node=self)
        energy_unit = df.get_unit(energy_m.column_id)

        df = df.filter(~pl.col(FORECAST_COLUMN))

        df = df.rename({energy_m.column_id: 'TotalEnergy'}).ensure_unit('TotalEnergy', energy_unit)
        # Account the externally supplied energy separately
        ext_energy_df = ext_energy_df.rename({'energy': 'ExtEnergy'})
        df = df.paths.join_over_index(ext_energy_df)
        df = df.ensure_unit('ExtEnergy', energy_unit).with_columns([pl.col('ExtEnergy').fill_null(0)])
        # TotalEnergy -> amount of electricity consumed without the externally accounted electricity
        df = df.with_columns([(pl.col('TotalEnergy') - pl.col('ExtEnergy')).alias('TotalEnergy')])

        gdf = gen_mix_df.paths.join_over_index(df.select([YEAR_COLUMN, 'TotalEnergy']))
        gdf = gdf.multiply_cols(['share', 'TotalEnergy'], 'TotalEnergy', energy_unit)

        assert len(gdf.dim_ids) == 1
        es_dim = gdf.dim_ids[0]

        sdf = gdf.filter(pl.col(es_dim).eq('subsidized')).drop(es_dim).rename(dict(TotalEnergy='SubsidizedEnergy'))
        sdf = sub_mix_df.paths.join_over_index(sdf.select([YEAR_COLUMN, 'SubsidizedEnergy']))
        sdf = sdf.multiply_cols(['share', 'SubsidizedEnergy'], 'SubsidizedEnergy', energy_unit)

        gdf = gdf.filter(~pl.col(es_dim).eq('subsidized'))
        gdf = gdf.paths.join_over_index(sdf.select([YEAR_COLUMN, es_dim, 'SubsidizedEnergy']))
        gdf = gdf.with_columns([pl.col('TotalEnergy') + pl.col('SubsidizedEnergy').fill_null(0)])

        idf = (df
            .select([YEAR_COLUMN, 'ExtEnergy', pl.lit('import').alias(es_dim)])
            .replace_meta(ppl.DataFrameMeta(units={'ExtEnergy': energy_unit}, primary_keys=[YEAR_COLUMN, es_dim]))
        )

        gdf = gdf.paths.join_over_index(idf)
        gdf = gdf.select([YEAR_COLUMN, es_dim, pl.col('TotalEnergy') + pl.col('ExtEnergy').fill_null(0)])
        sum_df = gdf.groupby([YEAR_COLUMN]).agg(pl.sum('TotalEnergy').alias('YearSum')).sort(YEAR_COLUMN)
        sum_df = ppl.to_ppdf(sum_df, meta=ppl.DataFrameMeta(units={'YearSum': energy_unit}, primary_keys=[YEAR_COLUMN]))
        gdf = gdf.paths.join_over_index(sum_df)

        m = self.get_default_output_metric()
        gdf = gdf.divide_cols(['TotalEnergy', 'YearSum'], m.column_id, m.unit)
        dim_id = list(self.output_dimensions.keys())[0]
        df = gdf.select([YEAR_COLUMN, pl.col(es_dim).alias(dim_id), m.column_id])

        df = extend_last_historical_value_pl(df, self.get_end_year())

        input_nodes = list(self.input_nodes)
        input_nodes.remove(energy_node)
        df = self.add_mix_normalized(df, input_nodes)

        return df


class DistrictHeatProductionMix(EnergyProductionMix):
    def compute(self) -> ppl.PathsDataFrame:
        mix_df = self.get_input_dataset_pl()
        assert len(mix_df.metric_cols) == 1
        assert len(mix_df.dim_ids) == 1
        m = self.get_default_output_metric()
        ec_dim_id, ec_dim = list(self.input_dimensions.items())[0]
        ec_s = ec_dim.series_to_ids_pl(mix_df[mix_df.dim_ids[0]])
        df = mix_df.select([pl.col(YEAR_COLUMN), ec_s.alias(ec_dim_id), pl.col(mix_df.metric_cols[0]).alias(m.column_id)])
        df = extend_last_historical_value_pl(df, self.get_end_year())
        df = self.add_mix_normalized(df, self.input_nodes)
        return df


class EnergyProductionEmissionFactor(AdditiveNode):
    output_metrics = {
        EMISSION_FACTOR_QUANTITY: NodeMetric(unit='g/kWh', quantity=EMISSION_FACTOR_QUANTITY)
    }
    default_unit = 'g/kWh'

    def compute(self) -> ppl.PathsDataFrame:
        mix_node = self.get_input_node(tag='mix')
        mix_df = mix_node.get_output_pl(target_node=self)
        mix_m = mix_node.get_default_output_metric()
        mix_df = mix_df.rename({mix_m.column_id: 'Share'})

        ef_df = self.get_input_dataset_pl()
        if len(self.input_dimensions) != 1:
            raise NodeError(self, "Must have exactly 1 input dimensions (%d given)" % len(self.input_dimensions))

        dim_id, es_dim = list(self.input_dimensions.items())[0]
        ef_df = ef_df.with_columns([es_dim.series_to_ids_pl(ef_df[dim_id])])
        ef_df = ef_df.rename({ef_df.metric_cols[0]: 'EF'})

        for node in self.get_input_nodes(tag='emission_factor'):
            node_df = node.get_output_pl(target_node=self)
            node_df = node_df.select([YEAR_COLUMN, *node_df.dim_ids, pl.col(node_df.metric_cols[0]).alias('NodeEF')])
            ef_df = ef_df.paths.join_over_index(node_df)
            ef_df = ef_df.with_columns([pl.col('EF').fill_null(pl.col('NodeEF'))]).drop('NodeEF')

        ef_df = extend_last_historical_value_pl(ef_df, self.get_end_year())
        df = mix_df.paths.join_over_index(ef_df)

        m = self.output_metrics[EMISSION_FACTOR_QUANTITY]
        df = df.multiply_cols(['Share', 'EF'], 'EF', out_unit=m.unit)
        df = df.with_columns([pl.col('EF').fill_null(0).fill_nan(0)])
        meta = df.get_meta()
        zdf = df.groupby(YEAR_COLUMN).agg([pl.sum('EF'), pl.first(FORECAST_COLUMN)]).sort(YEAR_COLUMN)
        df = ppl.to_ppdf(zdf, meta=meta)
        df = df.rename(dict(EF=VALUE_COLUMN))

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

    def process_input(self, dimension_ids: list[str], quantity: str, col: str | None = None) -> ppl.PathsDataFrame:
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
        ]).drop_nulls()
        df = df.select([YEAR_COLUMN, *meta.dim_ids, VALUE_COLUMN, FORECAST_COLUMN])
        # df = df.set_unit(VALUE_COLUMN, output_unit)

        df = extend_last_historical_value_pl(df, self.get_end_year())
        return df


class VehicleMileage(VehicleDatasetNode):
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
        df = self.process_input(dimension_ids=['vehicle_type'], quantity=MILEAGE_QUANTITY)
        for node in self.input_nodes:
            ndf = node.get_output_pl(self)
            df = df.paths.add_with_dims(ndf)
        self.print(df.filter(pl.col('vehicle_type').eq('car_diesel')))
        return df


class TransportFuelFactor(VehicleDatasetNode):
    output_metrics = {
        EMISSION_FACTOR_QUANTITY: NodeMetric(unit='kg/km', quantity=EMISSION_FACTOR_QUANTITY)  # FIXME Not really emission but fuel
    }
    output_dimension_ids = [
        'energy_carrier', 'vehicle_type',
    ]
    input_dimension_ids = [
        'energy_carrier', 'vehicle_type',
    ]

    def compute(self) -> ppl.PathsDataFrame:
        return self.process_input(
            dimension_ids=['energy_carrier', 'vehicle_type'],
            quantity=EMISSION_FACTOR_QUANTITY,
            col='fuel'
        )


class TransportEmissionFactor(VehicleDatasetNode):
    output_metrics = {
        EMISSION_FACTOR_QUANTITY: NodeMetric(unit='kg/km', quantity=EMISSION_FACTOR_QUANTITY)
    }
    output_dimension_ids = [
        'emission_scope', 'greenhouse_gases', 'vehicle_type',
    ]
    input_dimension_ids = [
        'emission_scope', 'greenhouse_gases', 'vehicle_type',
    ]

    def compute(self) -> ppl.PathsDataFrame:
        df = self.process_input(
            dimension_ids=['emission_scope', 'greenhouse_gases', 'vehicle_type'],
            quantity=EMISSION_FACTOR_QUANTITY
        )
        return df


class TransportEmissions(MultiplicativeNode):
    #input_dimension_ids = [
    #    'emission_scope', 'vehicle_type', 'greenhouse_gases',
    #]
    #output_dimension_ids = [
    #    'emission_scope', 'vehicle_type'
    #]

    def compute(self) -> ppl.PathsDataFrame:
        df = super().compute()
        df = df.filter(~pl.col(self.get_default_output_metric().column_id).is_null())
        df = convert_to_co2e(df, 'greenhouse_gases')
        return df
