import polars as pl

import common.polars as ppl
from nodes.calc import convert_to_co2e, extend_last_historical_value, extend_last_historical_value_pl
from nodes.node import NodeMetric, NodeError, Node
from nodes.simple import AdditiveNode, MultiplicativeNode, SimpleNode, MixNode
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

        #df = extend_last_historical_value_pl(df, self.get_end_year())
        #for node in self.input_nodes:
        #    ndf = node.get_output_pl(self)
        #    df = df.paths.add_with_dims(ndf)

        return df


class BuildingFloorAreaHistorical(Node):
    def compute(self) -> ppl.PathsDataFrame:
        df = self.get_input_dataset_pl()
        df = df.with_columns(
            pl.col('building_use_extended').map_dict({
                'residential': 'residential'
            }, default='nonresidential').alias('building_use')
        )
        df = df.add_to_index('building_use')
        df = df.paths.sum_over_dims(['building_use_extended'])
        df = df.rename({df.metric_cols[0]: self.get_default_output_metric().column_id})
        df = df.with_columns(pl.lit(False).alias(FORECAST_COLUMN))
        return df


class BuildingHeatHistorical(Node):
    def compute(self) -> ppl.PathsDataFrame:
        cop_node = self.get_input_node(tag='heat_pump_cop')
        cop_df = cop_node.get_output_pl(target_node=self)
        cop_df = cop_df.rename({VALUE_COLUMN: 'HeatPumpCOP'})

        cnode = self.get_input_node(tag='consumption')
        edf = cnode.get_output_pl(target_node=self)
        edf = edf.filter(pl.col('energy_carrier') != 'electricity')
        edf = edf.paths.to_wide(only_category_names=True)
        edf = edf.paths.join_over_index(cop_df)
        edf = edf.with_columns([
            (pl.col('natural_gas') + pl.col('biogas')).alias('natural_gas'),
            (pl.col('environmental_heat') / (1 - 1/pl.col('HeatPumpCOP'))).alias('heat_pumps'),
        ])
        edf = edf.set_unit('heat_pumps', edf.get_unit('environmental_heat'))
        edf = edf.drop(['HeatPumpCOP', 'biogas', 'environmental_heat'])
        renames = {col: 'Value@heating_system:%s' % col for col in edf.metric_cols}
        edf = edf.rename(renames).paths.to_narrow()
        return edf


class BuildingHeatPerArea(Node):
    def compute(self):
        e_node = self.get_input_node(tag='consumption')
        f_node = self.get_input_node(tag='floor_area')
        edf = e_node.get_output_pl(target_node=self)
        adf = f_node.get_output_pl(target_node=self)
        adf = adf.rename({VALUE_COLUMN: 'Area'})
        edf = edf.paths.sum_over_dims(['heating_system'])
        edf = edf.rename({VALUE_COLUMN: 'Energy'})

        sdf = adf.paths.sum_over_dims(['building_use'])
        sdf = sdf.rename({'Area': 'TotalArea'})
        adf = adf.paths.join_over_index(sdf)
        adf = adf.divide_cols(['Area', 'TotalArea'], 'AreaShare')

        df = adf.paths.join_over_index(edf, how='left', index_from='union')
        df = df.multiply_cols(['Energy', 'AreaShare'], 'Energy')

        # Residential buildings use about 8 % more heat per area
        edf = df.select_metrics(['Energy']).paths.to_wide(only_category_names=True)
        edf = edf.with_columns([
            (pl.col('residential') * (1 + 0.03)).alias('residential_new'),
        ])
        edf = edf.with_columns([
            (pl.col('nonresidential') - (pl.col('residential_new') - pl.col('residential'))).alias('nonresidential_new')
        ])
        edf = edf.set_unit('nonresidential_new', edf.get_unit('residential'))
        edf = edf.drop(['residential', 'nonresidential']).rename({
            'residential_new': 'Energy@building_use:residential',
            'nonresidential_new': 'Energy@building_use:nonresidential',
        })
        edf = edf.paths.to_narrow()
        df = df.select_metrics(['Area']).paths.join_over_index(edf)
        m = self.get_default_output_metric()
        df = df.divide_cols(['Energy', 'Area'], 'Efficiency', m.unit)

        df = df.filter(~pl.col(FORECAST_COLUMN))
        df = df.sort(YEAR_COLUMN).replace_meta(df.get_meta())
        df = df.select_metrics(['Efficiency']).rename({'Efficiency': VALUE_COLUMN}).drop_nulls()
        df = extend_last_historical_value_pl(df, self.get_end_year())
        nodes = list(self.input_nodes)
        nodes.remove(e_node)
        nodes.remove(f_node)
        df = self.add_nodes_pl(df, nodes)
        return df


class BuildingGeneralElectricityEfficiency(AdditiveNode):
    def compute(self):
        idf = self.get_input_dataset_pl()
        e_node = self.get_input_node(tag='consumption')
        h_node = self.get_input_node(tag='heat_consumption')
        f_node = self.get_input_node(tag='floor_area')
        edf = e_node.get_output_pl(target_node=self)
        adf = f_node.get_output_pl(target_node=self)
        hdf = h_node.get_output_pl(target_node=self)

        adf = adf.rename({VALUE_COLUMN: 'Area'})
        hdf = hdf.filter(pl.col('energy_carrier').eq('electricity')).drop('energy_carrier')
        edf = edf.filter(pl.col('energy_carrier').eq('electricity')).drop('energy_carrier')
        hdf = hdf.rename({VALUE_COLUMN: 'HeatElectricity'})
        edf = edf.rename({VALUE_COLUMN: 'AllElectricity'})

        df = adf.paths.join_over_index(hdf)
        df = df.paths.join_over_index(edf)

        df = df.paths.join_over_index(idf)
        df = df.multiply_cols(['Area', 'energy_per_area'], 'EstimatedElectricity', out_unit=df.get_unit('AllElectricity'))
        df = df.with_columns([pl.col('AllElectricity') - pl.col('HeatElectricity')])
        df = df.paths.add_sum_column('EstimatedElectricity', 'SumEstimated')
        df = df.with_columns([pl.col('EstimatedElectricity') * pl.col('AllElectricity') / pl.col('SumEstimated')])
        m = self.get_default_output_metric()
        df = df.divide_cols(['EstimatedElectricity', 'Area'], m.column_id, out_unit=m.unit)

        df = df.select_metrics([m.column_id])
        df = extend_last_historical_value_pl(df, self.get_end_year())
        nodes = list(self.input_nodes)
        nodes.remove(e_node)
        nodes.remove(h_node)
        nodes.remove(f_node)
        df = self.add_nodes_pl(df, nodes)
        return df


class BuildingHeatUseMix(MixNode):
    def compute(self):
        cnode = self.get_input_node(tag='consumption')
        edf = cnode.get_output_pl(target_node=self)

        sdf = edf.paths.sum_over_dims(['heating_system']).rename({VALUE_COLUMN: 'Total'})
        edf = edf.paths.join_over_index(sdf)
        edf = edf.divide_cols([VALUE_COLUMN, 'Total'], 'Share').select_metrics(['Share']).rename(dict(Share=VALUE_COLUMN))

        df = extend_last_historical_value_pl(edf, self.get_end_year())
        input_nodes = list(self.input_nodes)
        input_nodes.remove(cnode)
        df = self.add_mix_normalized(df, input_nodes)
        return df


class BiogasShare(AdditiveNode):
    def compute(self):
        cnode = self.get_input_node(tag='consumption')
        cdf = cnode.get_output_pl(target_node=self)
        df = cdf.paths.to_wide(only_category_names=True)
        df = df.with_columns(
            (pl.col('natural_gas') + pl.col('biogas')).alias('Total')
        )
        df = df.set_unit('Total', df.get_unit('natural_gas'))
        df = df.divide_cols(['biogas', 'Total'], VALUE_COLUMN)

        output_unit = self.get_default_output_metric().unit
        df = df.select_metrics([VALUE_COLUMN]).ensure_unit(VALUE_COLUMN, output_unit)
        df = extend_last_historical_value_pl(df, self.get_end_year())
        input_nodes = list(self.input_nodes)
        input_nodes.remove(cnode)
        df = self.add_nodes_pl(df, input_nodes)

        max_val = (1.0 * self.context.unit_registry.parse_units('dimensionless')).to(output_unit)
        df = df.with_columns(pl.col(VALUE_COLUMN).clip(0, max_val.m))
        return df


class BuildingHeatByCarrier(Node):
    def compute(self):
        cop_node = self.get_input_node(tag='heat_pump_cop')
        cop_df = cop_node.get_output_pl(target_node=self)
        cop_df = cop_df.rename({VALUE_COLUMN: 'HeatPumpCOP'})

        cnode = self.get_input_node(tag='consumption')
        edf = cnode.get_output_pl(target_node=self)
        bio_share = self.get_input_node(tag='biogas_share')
        sdf = bio_share.get_output_pl(target_node=self)
        sdf = sdf.rename({VALUE_COLUMN: 'BioShare'})

        edf = edf.paths.to_wide(only_category_names=True)
        edf = edf.paths.join_over_index(sdf)
        edf = edf.paths.join_over_index(cop_df)
        edf = edf.multiply_cols(['natural_gas', 'BioShare'], 'biogas', out_unit=edf.get_unit('natural_gas'))
        edf = edf.divide_cols(['heat_pumps', 'HeatPumpCOP'], 'electricity', out_unit=edf.get_unit('heat_pumps'))
        edf = edf.with_columns([
            (pl.col('natural_gas') - pl.col('biogas')).alias('natural_gas'),
            (pl.col('heat_pumps') - pl.col('electricity')).alias('environmental_heat'),
        ])
        edf = edf.set_unit('environmental_heat', edf.get_unit('heat_pumps'))
        edf = edf.drop(['BioShare', 'HeatPumpCOP', 'heat_pumps'])
        renames = {col: 'Value@energy_carrier:%s' % col for col in edf.metric_cols}
        edf = edf.rename(renames).paths.to_narrow()
        return edf


class ElectricityProductionMix(MixNode):
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


class DistrictHeatProductionMix(MixNode):
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

        es_dim_id, es_dim = list(self.input_dimensions.items())[0]
        ef_df = ef_df.with_columns([es_dim.series_to_ids_pl(ef_df[es_dim_id])])
        ef_df = ef_df.rename({ef_df.metric_cols[0]: 'EF'})

        for node in self.get_input_nodes(tag='emission_factor'):
            node_df = node.get_output_pl(target_node=self)
            node_df = node_df.select([YEAR_COLUMN, *node_df.dim_ids, pl.col(node_df.metric_cols[0]).alias('NodeEF')])
            assert set(ef_df.dim_ids) == set(node_df.dim_ids)
            ef_df = ef_df.paths.join_over_index(node_df)
            ef_df = ef_df.with_columns([pl.col('EF').fill_null(pl.col('NodeEF'))]).drop('NodeEF')

        ef_df = extend_last_historical_value_pl(ef_df, self.get_end_year())
        df = mix_df.paths.join_over_index(ef_df, index_from='union')
        m = self.output_metrics[EMISSION_FACTOR_QUANTITY]
        df = df.multiply_cols(['Share', 'EF'], 'EF', out_unit=m.unit)
        df = df.with_columns([pl.col('EF').fill_null(0).fill_nan(0)])
        df = df.drop_nulls()
        meta = df.get_meta()
        other_dims = df.dim_ids
        other_dims.remove(es_dim_id)
        zdf = df.groupby([YEAR_COLUMN, *other_dims]).agg([pl.sum('EF'), pl.first(FORECAST_COLUMN)]).sort(YEAR_COLUMN)
        df = ppl.to_ppdf(zdf, meta=meta)
        df = df.rename(dict(EF=VALUE_COLUMN))
        return df


class EmissionFactor(Node):
    input_dimension_ids = ['energy_carrier', 'emission_scope']
    output_dimension_ids = ['energy_carrier', 'emission_scope']

    def compute(self) -> ppl.PathsDataFrame:
        df = self.get_input_dataset_pl()
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

        df = df.rename({metric_col: VALUE_COLUMN}).drop_nulls()
        meta = df.get_meta()
        if dim.id not in meta.primary_keys:
            meta.primary_keys.append(dim.id)
        if YEAR_COLUMN not in meta.primary_keys:
            meta.primary_keys.append(YEAR_COLUMN)
        df = df.paths.cast_index_to_str()

        for node in self.input_nodes:
            ndf = node.get_output_pl(self)
            ndf = ndf.ensure_unit(VALUE_COLUMN, meta.units[VALUE_COLUMN])
            ndf = ndf.select(df.columns).drop_nulls()
            ndf = ndf.paths.cast_index_to_str()
            df = ppl.to_ppdf(pl.concat([df, ndf], how='vertical'), meta=meta)

        if df.paths.index_has_duplicates():
            dupes = df.groupby(df._primary_keys).agg(pl.count()).filter(pl.col('count') > 1)
            self.print(dupes)
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
        edf = en.get_output_pl(self)
        edf = edf.rename({VALUE_COLUMN: 'Energy'})
        fdf = fn.get_output_pl(self)
        fdf = fdf.rename({VALUE_COLUMN: 'EF'})
        df = edf.paths.join_over_index(fdf, index_from='union')
        if df['EF'].has_validity():
            self.print(df.filter(pl.col('EF').is_null()))
            raise NodeError(self, "Emission factor not found for some categories")

        m = self.get_default_output_metric()
        df = df.multiply_cols(['Energy', 'EF'], m.column_id)
        df = df.ensure_unit(m.column_id, m.unit)
        meta = df.get_meta()
        zdf = (
            df.groupby([YEAR_COLUMN, *self.output_dimensions.keys()])
            .agg([pl.sum(m.column_id), pl.first(FORECAST_COLUMN)])
            .sort(YEAR_COLUMN)
        )
        df = ppl.to_ppdf(zdf, meta=meta)
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


class VehicleMileageHistorical(Node):
    output_dimension_ids = [
        'vehicle_type',
    ]
    input_dimension_ids = [
        'vehicle_type',
    ]

    def compute(self) -> ppl.PathsDataFrame:
        df = self.get_input_dataset_pl()
        m = self.get_default_output_metric()
        unit = df.get_unit('mileage')
        if '[vehicle]' not in unit.dimensionality:
            unit = unit * self.context.unit_registry('vehicle')
            df = df.set_unit('mileage', unit, force=True)
        df = df.rename({'mileage': m.column_id}).ensure_unit(m.column_id, m.unit)
        df = df.with_columns(pl.lit(False).alias(FORECAST_COLUMN))
        return df


class PassengerKilometers(Node):
    input_dimension_ids = [
        'vehicle_type',
    ]
    output_dimension_ids = [
        'transport_mode'
    ]

    def compute(self) -> ppl.PathsDataFrame:
        vnode = self.get_input_node(tag='vehicle_mileage')
        vdf = vnode.get_output_pl(target_node=self)
        onode = self.get_input_node(tag='occupancy_factor')
        odf = onode.get_output_pl(target_node=self)

        tm_dim = self.output_dimensions[self.output_dimension_ids[0]]
        vt_dim = self.input_dimensions[self.input_dimension_ids[0]]
        vdf = vdf.with_columns([
            vt_dim.ids_to_groups(pl.col(vt_dim.id)).alias('vehicle_group')
        ])
        vdf = (vdf
            .with_columns(tm_dim.series_to_ids_pl(vdf['vehicle_group']).alias('transport_mode'))
            .drop('vehicle_group')
            .add_to_index('transport_mode')
        )
        vdf = vdf.paths.sum_over_dims(['vehicle_type']).drop_nulls(['transport_mode'])

        vdf = vdf.rename({VALUE_COLUMN: 'VehicleMileage'})
        odf = odf.rename({VALUE_COLUMN: 'OccupancyFactor'})
        vdf = vdf.paths.join_over_index(odf).filter(pl.col('OccupancyFactor').is_not_null())
        unit = self.get_default_output_metric().unit
        vdf = vdf.multiply_cols(['VehicleMileage', 'OccupancyFactor'], 'PassengerKilometers', out_unit=unit)
        vdf = vdf.select_metrics(['PassengerKilometers']).rename(dict(PassengerKilometers=VALUE_COLUMN))

        return vdf


class VehicleKilometersPerInhabitant(Node):
    def compute(self) -> ppl.PathsDataFrame:
        nodes = list(self.input_nodes)
        pkm_node = self.get_input_node(tag='passenger_kilometers')
        pdf = pkm_node.get_output_pl(target_node=self)
        nodes.remove(pkm_node)

        of_node = self.get_input_node(tag='occupancy_factor')
        odf = of_node.get_output_pl(target_node=self)
        nodes.remove(of_node)

        m_node = self.get_input_node(tag='mileage_historical')
        mdf = m_node.get_output_pl(target_node=self)
        nodes.remove(m_node)

        pop_node = self.get_input_node(tag='population')
        popdf = pop_node.get_output_pl(target_node=self)
        popdf = popdf.rename({VALUE_COLUMN: 'Pop'})
        nodes.remove(pop_node)

        m = self.get_default_output_metric()
        pdf = pdf.rename({VALUE_COLUMN: 'Pkm'})
        odf = odf.rename({VALUE_COLUMN: 'OF'})
        pdf = pdf.paths.join_over_index(odf)
        pdf = pdf.divide_cols(['Pkm', 'OF'], 'Vkm')
        pdf = pdf.paths.join_over_index(popdf)
        pdf = pdf.divide_cols(['Vkm', 'Pop'], 'LocalTransportVkm', out_unit=m.unit).select_metrics(['LocalTransportVkm'])

        tm_dim = self.context.dimensions['transport_mode']
        vt_dim = self.context.dimensions['vehicle_type']

        mdf = mdf.with_columns([
            vt_dim.ids_to_groups(pl.col(vt_dim.id)).alias('vehicle_group')
        ])
        mdf = (mdf
            .with_columns(tm_dim.series_to_ids_pl(mdf['vehicle_group']).alias('transport_mode'))
            .drop('vehicle_group')
            .add_to_index('transport_mode')
        )
        mdf = mdf.paths.sum_over_dims(['vehicle_type']).drop_nulls(['transport_mode'])
        mdf = mdf.rename({VALUE_COLUMN: 'Vkm'})
        mdf = mdf.paths.join_over_index(popdf).divide_cols(['Vkm', 'Pop'], 'Vkm', out_unit=m.unit)
        mdf = mdf.paths.join_over_index(pdf, how='outer').sort(YEAR_COLUMN)
        mdf = mdf.with_columns(pl.col('Vkm').fill_null(pl.col('LocalTransportVkm'))).select_metrics(['Vkm'])
        mdf = extend_last_historical_value_pl(mdf, self.get_end_year())

        mdf = mdf.rename(dict(Vkm=m.column_id))
        return self.add_nodes_pl(mdf, nodes)


class VehicleEngineTypeSplit(MixNode):
    def compute(self) -> ppl.PathsDataFrame:
        mnode = self.get_input_node(tag='mileage')
        mdf = mnode.get_output_pl(target_node=self)
        dim = self.input_dimensions['vehicle_type']
        mdf = (
            mdf.with_columns(dim.ids_to_groups(pl.col('vehicle_type')).alias('group'))
            .add_to_index('group')
        )
        mdf = mdf.paths.calculate_shares(VALUE_COLUMN, 'Share', over_dims=['vehicle_type'])
        m = self.get_default_output_metric()
        mdf = (
            mdf.select_metrics(['Share'])
            .rename(dict(Share=m.column_id))
            .ensure_unit(m.column_id, m.unit)
        )
        nodes = list(self.input_nodes)
        nodes.remove(mnode)
        df = mdf.with_columns(pl.lit(False).alias(FORECAST_COLUMN))

        gdf = df.select(['vehicle_type', 'group']).unique()
        df = df.drop('group')
        df = extend_last_historical_value_pl(df, self.get_end_year())
        df = self.add_nodes_pl(df, nodes)
        df = ppl.to_ppdf(df.join(gdf, on='vehicle_type', how='left'), df.get_meta()).sort(YEAR_COLUMN).add_to_index('group')
        df = self.add_mix_normalized(df, [], over_dims=['vehicle_type'])
        df = df.drop('group')
        return df


class VehicleMileage(Node):
    def compute(self) -> ppl.PathsDataFrame:
        pop_node = self.get_input_node(tag='population')
        popdf = pop_node.get_output_pl(target_node=self)
        popdf = popdf.rename({VALUE_COLUMN: 'Pop'})

        et_node = self.get_input_node(tag='engine_type_split')
        etdf = et_node.get_output_pl(target_node=self)
        etdf = etdf.rename({VALUE_COLUMN: 'EngineTypeShare'})

        m_node = self.get_input_node(tag='mileage_per_inhabitant')
        mdf = m_node.get_output_pl(target_node=self)
        mdf = mdf.rename({VALUE_COLUMN: 'MileagePerPop'})

        m = self.get_default_output_metric()
        mdf = mdf.paths.join_over_index(popdf)
        mdf = mdf.multiply_cols(['MileagePerPop', 'Pop'], 'TotalMileage', out_unit=m.unit)

        vt_dim = self.context.dimensions['vehicle_type']
        tm_dim = self.context.dimensions['transport_mode']
        etdf = etdf.with_columns(vt_dim.ids_to_groups(pl.col('vehicle_type')).alias('vehicle_group'))
        etdf = etdf.with_columns(tm_dim.series_to_ids_pl(etdf['vehicle_group']).alias('transport_mode')).add_to_index('transport_mode')
        df = etdf.paths.join_over_index(mdf)
        df = df.multiply_cols(['TotalMileage', 'EngineTypeShare'], 'Mileage', out_unit=m.unit)
        df = df.select([YEAR_COLUMN, 'vehicle_type', FORECAST_COLUMN, pl.col('Mileage').alias(m.column_id)])
        return df


class TransportFuelFactor(AdditiveNode):
    output_metrics = {
        EMISSION_FACTOR_QUANTITY: NodeMetric(unit='kg/vkm', quantity=EMISSION_FACTOR_QUANTITY)  # FIXME Not really emission but fuel
    }
    output_dimension_ids = [
        'energy_carrier', 'vehicle_type',
    ]
    input_dimension_ids = [
        'energy_carrier', 'vehicle_type',
    ]

    def compute(self) -> ppl.PathsDataFrame:
        df = self.get_input_dataset_pl()
        df = df.select_metrics(['fuel']).drop_nulls()
        if 'vehicle' not in df.get_unit('fuel').dimensionality:
            df = df.set_unit('fuel', 'kg/vkm', force=True)
        m = self.get_default_output_metric()
        df = df.rename(dict(fuel=m.column_id)).ensure_unit(m.column_id, m.unit)
        df = extend_last_historical_value_pl(df, self.get_end_year())
        df = self.add_nodes_pl(df, self.input_nodes)
        return df


class TransportEmissionFactor(Node):
    output_dimension_ids = [
        'emission_scope', 'vehicle_type',
    ]

    def compute(self) -> ppl.PathsDataFrame:
        ef_node = self.get_input_node(tag='general_electricity')
        efdf = ef_node.get_output_pl(self)
        efdf = efdf.rename({efdf.metric_cols[0]: 'EEF'})

        ec_node = self.get_input_node(tag='electricity_consumption_factor')
        ecdf = ec_node.get_output_pl(self)
        ecdf = ecdf.rename({ecdf.metric_cols[0]: 'EC'})

        m = self.get_default_output_metric()
        edf = ecdf.paths.join_over_index(efdf, index_from='union')
        edf = edf.multiply_cols(['EC', 'EEF'], 'EF', m.unit)
        # We only have CO2e for electricity, so pretend that it's just CO2 for now
        edf = edf.with_columns([pl.lit('co2').alias('greenhouse_gases')]).add_to_index('greenhouse_gases')
        edf = edf.select([YEAR_COLUMN, 'vehicle_type', 'emission_scope', 'greenhouse_gases', 'EF', FORECAST_COLUMN])

        fef_node = self.get_input_node(tag='fuel_emission_factor')
        fdf = fef_node.get_output_pl(target_node=self)
        fdf = fdf.rename({VALUE_COLUMN: 'EF'})

        ef_expr = pl.col('EF').map_dict({0.0: None}, default=pl.col('EF'))
        fdf = fdf.with_columns([ef_expr]).filter(~pl.col('EF').is_null())
        fdf = fdf.ensure_unit('EF', m.unit)
        fdf = extend_last_historical_value_pl(fdf, self.get_end_year())
        fdf = fdf.select([YEAR_COLUMN, 'vehicle_type', 'emission_scope', 'greenhouse_gases', 'EF', FORECAST_COLUMN])

        df = edf.paths.add_with_dims(fdf, how='outer')
        meta = df.get_meta()
        df = df.sort([YEAR_COLUMN, *df.dim_ids]).replace_meta(meta)
        df = df.rename({'EF': m.column_id})

        df = convert_to_co2e(df, 'greenhouse_gases')
        return df


class TransportEmissionsForFuel(AdditiveNode):
    def compute(self) -> ppl.PathsDataFrame:
        ff_node = self.get_input_node(tag='fuel_factor')
        ffdf = ff_node.get_output_pl(target_node=self)
        ffdf = ffdf.rename({VALUE_COLUMN: 'fuel'})

        efdf = self.get_input_dataset_pl()
        eunit = efdf.get_unit('emission_factor')
        if 'vehicle' not in eunit.dimensionality:
            efdf = efdf.set_unit('emission_factor', 'kg/vkm', force=True)
        df = efdf.paths.join_over_index(ffdf, index_from='union').drop_nulls()

        df = df.filter(pl.col('fuel').gt(0))
        df = df.divide_cols(['emission_factor', 'fuel'], 'EFFuel')
        m = self.get_default_output_metric()
        df = df.filter(pl.col('EFFuel').gt(0)).select_metrics(['EFFuel']).rename(dict(EFFuel=m.column_id))
        df = df.paths.sum_over_dims(['energy_carrier'])
        df = extend_last_historical_value_pl(df, self.get_end_year())
        return df


class TransportElectricity(AdditiveNode):
    def compute(self) -> ppl.PathsDataFrame:
        df = self.get_input_dataset_pl()
        m = self.get_default_output_metric()
        # Replace 0 values with nulls
        el_expr = pl.col('electricity').map_dict({0.0: None}, default=pl.col('electricity'))
        df = df.select([YEAR_COLUMN, *df.dim_ids, el_expr])
        df = df.set_unit('electricity', 'kWh/vkm', force=True)
        df = df.rename(dict(electricity=m.column_id)).ensure_unit(m.column_id, m.unit)
        # choose only electricity energy carrier and drop nulls
        filter_expr = pl.col('energy_carrier').eq('electricity') & ~pl.col(m.column_id).is_null()
        df = df.filter(filter_expr).drop('energy_carrier')
        df = extend_last_historical_value_pl(df, self.get_end_year())
        df = self.add_nodes_pl(df, self.input_nodes)
        return df


class TransportEmissions(MultiplicativeNode):
    input_dimension_ids = [
        'emission_scope', 'vehicle_type',
    ]
    output_dimension_ids = [
        'emission_scope', 'vehicle_type'
    ]
    default_unit = 'kt/a'
    quantity = 'emissions'

    def compute(self) -> ppl.PathsDataFrame:
        df = super().compute()
        df = df.filter(~pl.col(self.get_default_output_metric().column_id).is_null())
        return df


class NonroadMachineryEmissions(Node):
    def compute(self) -> ppl.PathsDataFrame:
        node = self.get_input_node()
        df = node.get_output_pl(target_node=self)
        ef = dict({
            'co2': 3.15,
            'n2o': 1.43e-04,
            'ch4': 1.65e-04,
        })
        zdf = pl.DataFrame([{YEAR_COLUMN: 2022, 'greenhouse_gases': key, 'ef': val} for key, val in ef.items()])
        years = pl.DataFrame(range(1990, self.get_end_year() + 1), schema=[YEAR_COLUMN])
        ef_unit = self.context.unit_registry.parse_units('kg/kg')
        efdf = ppl.to_ppdf(zdf, meta=ppl.DataFrameMeta(
            units={'ef': ef_unit},
            primary_keys=[YEAR_COLUMN, 'greenhouse_gases']
        ))
        efdf = efdf.paths.to_wide()
        meta = efdf.get_meta()
        zdf = efdf.join(years, on=YEAR_COLUMN, how='outer')
        zdf = zdf.fill_null(strategy='forward')
        zdf = zdf.fill_null(strategy='backward')
        efdf = ppl.to_ppdf(zdf, meta=meta)
        efdf = efdf.paths.to_narrow()
        efdf = efdf.with_columns(pl.when(pl.col(YEAR_COLUMN) > 2022).then(True).otherwise(False).alias(FORECAST_COLUMN))

        df = df.with_columns(pl.when(pl.col(YEAR_COLUMN) > 2022).then(pl.lit(True)).otherwise(False).alias(FORECAST_COLUMN))

        df = df.paths.join_over_index(efdf, how='outer', index_from='union')
        df = df.drop_nulls()
        m = self.get_default_output_metric()
        df = df.with_columns(pl.lit('scope1').alias('emission_scope')).add_to_index('emission_scope')
        df = df.multiply_cols(['Value', 'ef'], out_col=m.column_id, out_unit=m.unit).select_metrics([m.column_id])
        df = convert_to_co2e(df, 'greenhouse_gases')
        df = df.paths.sum_over_dims(['energy_carrier', 'non_road_machinery'])
        return df


class WasteIncinerationEmissions(SimpleNode):
    def compute(self) -> ppl.PathsDataFrame:
        dfs = self.get_input_datasets_pl()
        fdf = efdf = None
        for df in dfs:
            if 'share_of_fossil_co2' in df:
                fdf = df
                continue
            if 'emission_factor' in df:
                efdf = df
                continue
        if fdf is None:
            raise NodeError(self, "Dataset with 'share_of_fossil_co2' not found")
        if efdf is None:
            raise NodeError(self, "Dataset with emission factors not found")

        ccs_node = self.get_input_node(tag='ccs_share')
        cdf = ccs_node.get_output_pl(target_node=self)

        amount_node = self.get_input_node(tag='amount')
        adf = amount_node.get_output_pl(target_node=self)

        efdf = extend_last_historical_value_pl(efdf, self.get_end_year())
        df = (
            adf.paths.join_over_index(efdf, how='left', index_from='union')
        )
        df = (
            df.multiply_cols([VALUE_COLUMN, 'emission_factor'], 'Emissions')
            .select([*df.get_meta().primary_keys, FORECAST_COLUMN, 'Emissions'])
        )
        df = convert_to_co2e(df, 'greenhouse_gases')

        fdf = extend_last_historical_value_pl(fdf, self.get_end_year())
        df = df.paths.join_over_index(fdf.select_metrics(['share_of_fossil_co2']))
        df = df.multiply_cols(['Emissions', 'share_of_fossil_co2'], 'FossilEmissions').ensure_unit('FossilEmissions', df.get_unit('Emissions'))
        df = (
            df.with_columns((pl.col('Emissions') - pl.col('FossilEmissions')).alias('BioEmissions'))
            .set_unit('BioEmissions', df.get_unit('Emissions'))
        )

        scope_dim = self.context.dimensions['emission_scope']
        cdf = cdf.rename({VALUE_COLUMN: 'CCSShare'})
        cdf = cdf.ensure_unit('CCSShare', self.context.unit_registry.parse_units('dimensionless'))
        df = df.paths.join_over_index(cdf)
        df = df.with_columns([
            (pl.col('FossilEmissions') * (1 - pl.col('CCSShare'))).alias('FossilLeft'),
            (pl.lit(0) - pl.col('BioEmissions') * pl.col('CCSShare')).alias('BioCaptured'),
        ])
        df = (
            df.set_unit('FossilLeft', df.get_unit('FossilEmissions'))
            .set_unit('BioCaptured', df.get_unit('BioEmissions'))
            .select_metrics(['FossilLeft', 'BioCaptured'])
            .paths.sum_over_dims(['waste_incineration_plants'])
        )

        s1df = (df
            .select_metrics(['FossilLeft'])
            .rename({'FossilLeft': VALUE_COLUMN})
            .with_columns(pl.lit('scope1').alias(scope_dim.id))
            .add_to_index(scope_dim.id)
        )

        ndf = (df
            .select_metrics(['BioCaptured'])
            .rename({'BioCaptured': VALUE_COLUMN})
            .with_columns(pl.lit('negative_emissions').alias(scope_dim.id))
            .add_to_index(scope_dim.id)
        )

        df = ppl.to_ppdf(pl.concat([s1df, ndf]), ndf.get_meta())
        df = df.ensure_unit(VALUE_COLUMN, self.get_default_output_metric().unit)
        return df


class SewageSludgeProcessingEmissions(SimpleNode):
    def compute(self) -> ppl.PathsDataFrame:
        df = self.get_input_dataset_pl()
        df = df.with_columns(pl.lit(False).alias(FORECAST_COLUMN))
        df = extend_last_historical_value_pl(df, self.get_end_year())

        ccs_node = self.get_input_node(tag='ccs_share')
        cdf = ccs_node.get_output_pl(target_node=self)
        cdf = cdf.rename({VALUE_COLUMN: 'CCSShare'})

        df = df.paths.join_over_index(cdf).drop_nulls()
        df = df.filter(pl.col('greenhouse_gases') == 'co2_biogen')
        m = self.get_default_output_metric()
        df = df.multiply_cols(['emissions', 'CCSShare'], 'Captured', out_unit=m.unit)
        df = df.with_columns(pl.lit('negative_emissions').alias('emission_scope')).add_to_index('emission_scope')
        df = df.drop('greenhouse_gases').select_metrics(['Captured']).rename({'Captured': m.column_id})
        df = df.with_columns((pl.lit(0) - pl.col(m.column_id)).alias(m.column_id))
        df = df.paths.to_wide()
        ncol = df.metric_cols[0]
        s1col = ncol.replace('negative_emissions', 'scope1')
        df = df.with_columns(pl.lit(0.0).alias(s1col)).set_unit(s1col, df.get_unit(ncol))
        df = df.paths.to_narrow()
        return df
