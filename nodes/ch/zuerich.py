from typing import cast
import polars as pl
from common import polars as ppl

from nodes.calc import convert_to_co2e, extend_last_historical_value_pl
from nodes.node import NodeMetric, NodeError, Node
from nodes.simple import AdditiveNode, MultiplicativeNode, SimpleNode, MixNode
from nodes.constants import CONSUMPTION_FACTOR_QUANTITY, DEFAULT_METRIC, EMISSION_FACTOR_QUANTITY, EMISSION_QUANTITY, ENERGY_QUANTITY, FORECAST_COLUMN, MIX_QUANTITY, POPULATION_QUANTITY, VALUE_COLUMN, YEAR_COLUMN, MILEAGE_QUANTITY
from nodes.units import Unit
from params.param import BoolParameter


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
        df = self.get_input_dataset_pl(tag='energy')
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

        odf = self.get_input_dataset_pl('other_fuel_use')
        assert len(odf.metric_cols) == 1
        odf = odf.with_columns(pl.col(odf.metric_cols[0]) * -1)

        df = df.paths.add_df(odf, how='left')

        tedf = (
            self.get_input_node(tag='transport_electricity').get_output_pl(target_node=self)
        )
        tedf = tedf.with_columns(pl.col(tedf.metric_cols[0]) * -1)
        df = df.paths.add_df(tedf, how='left')
        return df


class BuildingFloorAreaHistorical(Node):
    def compute(self) -> ppl.PathsDataFrame:
        df = self.get_input_dataset_pl()
        df = df.with_columns(
            pl.col('building_use_extended').replace(
                'residential', 'residential', default='nonresidential'
            ).alias('building_use')
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
        gas_cols = [col for col in ('natural_gas', 'biogas', 'biogas_import') if col in edf.columns]
        edf = edf.with_columns([
            pl.sum_horizontal(gas_cols).alias('natural_gas'),
            (pl.col('environmental_heat') / (1 - 1/pl.col('HeatPumpCOP'))).alias('heat_pumps'),
        ])
        edf = edf.set_unit('heat_pumps', edf.get_unit('environmental_heat'))
        gas_cols.remove('natural_gas')
        edf = edf.drop(['HeatPumpCOP', *gas_cols, 'environmental_heat'])
        renames = {col: 'Value@heating_system:%s' % col for col in edf.metric_cols}
        edf = edf.rename(renames).paths.to_narrow()
        return edf


class BuildingUsefulHeat(Node):
    def compute(self) -> ppl.PathsDataFrame:
        en = self.get_input_node(tag='energy')
        copn = self.get_input_node(tag='cop')
        df = en.get_output_pl(target_node=self)
        cdf = copn.get_output_pl(target_node=self)
        cdf = cdf.rename({VALUE_COLUMN: 'COP'})
        df = df.paths.join_over_index(cdf)
        # Heat pump COP is already taken into account, so replace theh multiplier
        # with 1.0.
        df = df.with_columns(pl.when(pl.col('heating_system').eq('heat_pumps')).then(1.0).otherwise(pl.col('COP')).alias('COP'))
        m = self.get_default_output_metric()
        df = df.multiply_cols([VALUE_COLUMN, 'COP'], VALUE_COLUMN, m.unit).select_metrics([VALUE_COLUMN])
        return df


class BuildingHeatPerArea(Node):
    def compute(self):
        e_node = self.get_input_node(tag='consumption')
        f_node = self.get_input_node(tag='floor_area')
        edf = e_node.get_output_pl(target_node=self)
        adf = f_node.get_output_pl(target_node=self)
        adf = adf.rename({VALUE_COLUMN: 'Area'})
        adf = adf.paths.to_wide().drop_nulls().paths.to_narrow()
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

        df = df.paths.join_over_index(idf).drop_nulls()
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


# class BiogasShare(AdditiveNode):
#     def compute(self):
#         cnode = self.get_input_node(tag='consumption')
#         df = cnode.get_output_pl(target_node=self)
#         df = df.filter(pl.col('energy_carrier').is_in(['natural_gas', 'biogas', 'biogas_import']))
#         df = df.paths.calculate_shares(VALUE_COLUMN, 'Share', over_dims=['energy_carrier'])

#         output_unit = self.get_default_output_metric().unit
#         df = df.select_metrics(['Share']).ensure_unit('Share', output_unit).rename(dict(Share=VALUE_COLUMN))
#         df = extend_last_historical_value_pl(df, self.get_end_year())
#         input_nodes = list(self.input_nodes)
#         input_nodes.remove(cnode)
#         df = self.add_nodes_pl(df, input_nodes)

#         max_val = (1.0 * self.context.unit_registry.parse_units('dimensionless')).to(output_unit)
#         df = df.with_columns(pl.col(VALUE_COLUMN).clip(0, max_val.m))
#         return df


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
        edf = edf.rename({'natural_gas': 'natural_gas_heat'})
        sdf = sdf.paths.to_wide(only_category_names=True)
        sdf = sdf.rename({col: 'Share:%s' % col for col in sdf.metric_cols})
        edf = edf.paths.join_over_index(sdf)
        edf = edf.paths.join_over_index(cop_df)
        drop_cols = []
        for col in ('natural_gas', 'biogas', 'biogas_import'):
            edf = edf.multiply_cols(['natural_gas_heat', 'Share:%s' % col], col, edf.get_unit('natural_gas_heat'))
            drop_cols.append('Share:%s' % col)
        edf = edf.drop([*drop_cols, 'natural_gas_heat'])
        edf = edf.divide_cols(['heat_pumps', 'HeatPumpCOP'], 'electricity', out_unit=edf.get_unit('heat_pumps'))
        edf = edf.with_columns([
            (pl.col('heat_pumps') - pl.col('electricity')).alias('environmental_heat'),
        ])
        edf = edf.set_unit('environmental_heat', edf.get_unit('heat_pumps'))
        edf = edf.drop(['HeatPumpCOP', 'heat_pumps'])
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
        sum_df = gdf.group_by([YEAR_COLUMN]).agg(pl.sum('TotalEnergy').alias('YearSum')).sort(YEAR_COLUMN)
        sum_df = ppl.to_ppdf(sum_df, meta=ppl.DataFrameMeta(units={'YearSum': energy_unit}, primary_keys=[YEAR_COLUMN]))
        gdf = gdf.paths.join_over_index(sum_df)

        m = self.get_default_output_metric()
        gdf = gdf.divide_cols(['TotalEnergy', 'YearSum'], m.column_id, m.unit)
        dim_id = list(self.output_dimensions.keys())[0]
        df = gdf.select([YEAR_COLUMN, pl.col(es_dim).alias(dim_id), m.column_id])

        df = df.filter(pl.col(m.column_id).is_not_null() & pl.col(m.column_id).is_not_nan())

        df = extend_last_historical_value_pl(df, self.get_end_year())

        input_nodes = list(self.input_nodes)
        input_nodes.remove(energy_node)
        df = self.add_mix_normalized(df, input_nodes)

        return df


class GasGridMixin(Node):
    def use_gas_grid(self, df: ppl.PathsDataFrame):
        df = df.paths.to_wide(only_category_names=True)
        df = df.with_columns([pl.col(col).fill_nan(0.0) for col in df.metric_cols])
        df = df.sum_cols(['natural_gas', 'biogas', 'biogas_import'], out_col='all_gas', skip_missing=True)

        snode = self.get_input_node(tag='grid_share')
        sdf = snode.get_output_pl(target_node=self)
        sdf = sdf.select_metrics(sdf.metric_cols[0], rename='GridShare').ensure_unit('GridShare', '')

        mnode = self.get_input_node(tag='gas_mix')
        mdf = mnode.get_output_pl(target_node=self)
        mdf = mdf.ensure_unit(mdf.metric_cols[0], '')
        mdf = mdf.paths.to_wide(only_category_names=True)

        zdf = df.select(YEAR_COLUMN).join(mdf, on=YEAR_COLUMN, how='left').join(sdf, on=YEAR_COLUMN, how='left')
        zdf = zdf.with_columns(pl.col('GridShare').fill_null(0.0))

        def fc_only(col: str):
            own_supply = (1 - zdf['GridShare']) * pl.col(col)
            grid_supply = zdf['GridShare'] * zdf[col] * pl.col('all_gas')
            return (
                pl.when(pl.col(FORECAST_COLUMN))
                .then(own_supply + grid_supply)
                .otherwise(pl.col(col))
                .fill_nan(0.0).alias(col)
            )

        cols = ('natural_gas', 'biogas', 'biogas_import')
        for col in cols:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias(col)).set_unit(col, df.get_unit('all_gas'))

        df = df.with_columns([
            fc_only('natural_gas'),
            fc_only('biogas'),
            fc_only('biogas_import'),
        ])
        df = df.drop('all_gas')

        m = self.get_default_output_metric()
        df = df.paths.to_narrow(assign_dimension='energy_carrier', assign_metric=m.column_id)
        return df


class DistrictHeatProductionMix(MixNode, GasGridMixin):
    allowed_parameters = MixNode.allowed_parameters + [
        BoolParameter('use_gas_network', label='District heat uses gas grid mix')
    ]

    def compute(self) -> ppl.PathsDataFrame:
        mix_df = self.get_input_dataset_pl()
        assert len(mix_df.metric_cols) == 1
        assert len(mix_df.dim_ids) == 1
        m = self.get_default_output_metric()
        ec_dim_id, ec_dim = list(self.input_dimensions.items())[0]
        ec_s = ec_dim.series_to_ids_pl(mix_df[mix_df.dim_ids[0]])
        df = mix_df.select([pl.col(YEAR_COLUMN), ec_s.alias(ec_dim_id), pl.col(mix_df.metric_cols[0]).alias(m.column_id)])
        df = extend_last_historical_value_pl(df, self.get_end_year())

        add_nodes = list(self.input_nodes)
        snode = self.get_input_node(tag='grid_share', required=False)
        if snode is not None:
            add_nodes.remove(snode)
        mnode = self.get_input_node(tag='gas_mix', required=False)
        if mnode is not None:
            add_nodes.remove(mnode)

        df = self.add_mix_normalized(df, add_nodes)

        use_grid = self.get_parameter_value('use_gas_network', required=False)
        if use_grid:
            df = self.use_gas_grid(df)

        return df


class GasGridNode(AdditiveNode, GasGridMixin):
    def compute(self) -> ppl.PathsDataFrame:
        df = super().compute()
        meta = df.get_meta()
        other_dims = df.dim_ids
        other_dims.remove('energy_carrier')
        other_dim_cats = df.select(other_dims).unique()
        dfs = []
        for row in other_dim_cats.iter_rows():
            filters = [pl.col(dim).eq(cat) for dim, cat in zip(other_dims, row)]
            fdf = df.filter(pl.all_horizontal(filters)).drop(other_dims)
            fdf = self.use_gas_grid(fdf).with_columns([
                pl.lit(cat).alias(dim) for dim, cat in zip(other_dims, row)
            ])
            dfs.append(fdf)
        df = ppl.to_ppdf(pl.concat(dfs), meta=meta)
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

        ccs_node = self.get_input_node(tag='ccs', required=False)
        ccs_df = None
        if ccs_node is not None:
            ccs_df = ccs_node.get_output_pl(target_node=self)
            ccs_df = ccs_df.rename({VALUE_COLUMN: 'CCS'}).ensure_unit('CCS', 'dimensionless')

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
            ef_df = ef_df.paths.join_over_index(node_df, how='outer')
            ef_df = ef_df.with_columns([pl.col('EF').fill_null(pl.col('NodeEF'))]).drop('NodeEF')

        df = extend_last_historical_value_pl(ef_df, self.get_end_year())

        if ccs_df is not None:
            df = df.paths.join_over_index(ccs_df).with_columns(pl.col('CCS').fill_null(0.0))
            #df = df.multiply_cols(['EF', 'CCS'], 'EFRemaining', out_unit=df.get_unit('EF'))
            df = df.with_columns(
                pl.when(pl.col('energy_carrier').eq('natural_gas') & pl.col('emission_scope').eq('scope1'))
                    .then(pl.col('EF') * (1 - pl.col('CCS'))).otherwise(pl.col('EF')).alias('EF')
            )

        df = mix_df.paths.join_over_index(df, index_from='union')
        m = self.output_metrics[EMISSION_FACTOR_QUANTITY]
        df = df.multiply_cols(['Share', 'EF'], 'EF', out_unit=m.unit)
        df = df.with_columns([pl.col('EF').fill_null(0).fill_nan(0)])
        df = df.drop_nulls()

        meta = df.get_meta()
        other_dims = df.dim_ids
        other_dims.remove(es_dim_id)
        zdf = df.group_by([YEAR_COLUMN, *other_dims]).agg([pl.sum('EF'), pl.first(FORECAST_COLUMN)]).sort(YEAR_COLUMN)
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

        df = extend_last_historical_value_pl(df, self.get_end_year())

        for node in self.input_nodes:
            ndf = node.get_output_pl(self).ensure_unit(VALUE_COLUMN, meta.units[VALUE_COLUMN])
            ndf = ndf.rename({VALUE_COLUMN: '_Right'})
            df = df.paths.join_over_index(ndf, how='outer')
            df = df.with_columns(pl.col(VALUE_COLUMN).fill_null(0) + pl.col('_Right').fill_null(0)).drop('_Right')

        if df.paths.index_has_duplicates():
            dupes = df.group_by(df._primary_keys).agg(pl.count()).filter(pl.col('count') > 1)
            self.print(dupes)
            raise NodeError(self, "Duplicate rows detected")
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
            df.group_by([YEAR_COLUMN, *self.output_dimensions.keys()])
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

        pc_unit = cast(Unit, act_df.get_unit('Value') / pop_df.get_unit('Pop'))
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
        'Fuel': NodeMetric(unit='kg/vkm', quantity=CONSUMPTION_FACTOR_QUANTITY),
        'Electricity': NodeMetric(unit='kWh/vkm', quantity=CONSUMPTION_FACTOR_QUANTITY),
    }
    output_dimension_ids = [
        'energy_carrier', 'vehicle_type',
    ]
    input_dimension_ids = [
        'energy_carrier', 'vehicle_type',
    ]

    def compute(self) -> ppl.PathsDataFrame:
        df = self.get_input_dataset_pl()

        v_unit = self.context.unit_registry.parse_units('vehicle')

        df = df.select_metrics(['fuel', 'electricity'])
        e_m = self.output_metrics['Electricity']
        f_m = self.output_metrics['Fuel']

        exprs = []
        for col, m in (('electricity', e_m), ('fuel', f_m)):
            u = df.get_unit(col)
            if 'vehicle' not in u.dimensionality:
                df = df.set_unit(col, cast(Unit, u / v_unit), force=True)
            df = df.ensure_unit(col, m.unit).rename({col: m.column_id})
            df = df.with_columns(pl.col(m.column_id).fill_nan(None))
            exprs.append(pl.col(m.column_id).is_null() | pl.col(m.column_id).eq(0.0))

        df = df.filter(~pl.all_horizontal(exprs))
        df = df.with_columns(pl.lit(False).alias(FORECAST_COLUMN))
        df = extend_last_historical_value_pl(df, self.get_end_year())
        return df


class TransportEmissionFactor(Node):
    output_dimension_ids = [
        'emission_scope', 'vehicle_type', 'energy_carrier'
    ]

    def compute(self) -> ppl.PathsDataFrame:
        ef_node = self.get_input_node(tag='general_electricity_ef')
        efdf = ef_node.get_output_pl(self)
        efdf = efdf.rename({efdf.metric_cols[0]: 'EEF'})

        ec_node = self.get_input_node(tag='electricity_consumption_factor')
        ecdf = ec_node.get_output_pl(self)
        ecdf = ecdf.rename({ecdf.metric_cols[0]: 'EC'})

        m = self.get_default_output_metric()
        edf = ecdf.paths.join_over_index(efdf, index_from='union')
        edf = edf.multiply_cols(['EC', 'EEF'], 'EF', m.unit)
        # We only have CO2e for electricity, so pretend that it's just CO2 for now
        edf = edf.with_columns([
            pl.lit('co2').alias('greenhouse_gases'), pl.lit('electricity').alias('energy_carrier'),
        ]).add_to_index(['greenhouse_gases', 'energy_carrier'])
        edf = edf.select_metrics(['EF'])

        fef_node = self.get_input_node(tag='fuel_emission_factor')
        fdf = fef_node.get_output_pl(target_node=self)
        fdf = fdf.rename({VALUE_COLUMN: 'EF'})

        ef_expr = pl.col('EF').replace(0.0, None, default=pl.col('EF'))
        fdf = fdf.with_columns([ef_expr]).filter(~pl.col('EF').is_null())
        fdf = fdf.ensure_unit('EF', m.unit)
        fdf = extend_last_historical_value_pl(fdf, self.get_end_year())
        fdf = fdf.select_metrics(['EF'])

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

        efdf = self.get_input_dataset_pl(tag='emission_factor')
        eunit = efdf.get_unit('emission_factor')
        if 'vehicle' not in eunit.dimensionality:
            efdf = efdf.set_unit('emission_factor', 'kg/vkm', force=True)
        df = efdf.paths.join_over_index(ffdf, index_from='union').drop_nulls()

        df = df.filter(pl.col('fuel').gt(0))
        df = df.divide_cols(['emission_factor', 'fuel'], 'EFFuel')
        m = self.get_default_output_metric()
        df = df.filter(pl.col('EFFuel').gt(0)).select_metrics(['EFFuel']).rename(dict(EFFuel=m.column_id))
        df = df.paths.sum_over_dims(['energy_carrier'])
        tr_node = self.get_input_node(tag='tank_respiration', required=False)
        if tr_node is not None:
            trdf = (
                tr_node.get_output_pl(target_node=self)
                .rename({VALUE_COLUMN: 'TR'})
                .ensure_unit('TR', df.get_unit(m.column_id))
            )
            df = df.paths.join_over_index(trdf, how='outer')
            df = df.with_columns(pl.col(m.column_id).fill_null(0) + pl.col('TR').fill_null(0)).drop('TR')

        df = extend_last_historical_value_pl(df, self.get_end_year())

        anodes = self.get_input_nodes(tag='additive')
        if anodes:
            df = self.add_nodes_pl(df, anodes)

        return df


class TransportElectricity(AdditiveNode):
    def compute(self) -> ppl.PathsDataFrame:
        df = self.get_input_dataset_pl()
        m = self.get_default_output_metric()
        # Replace 0 values with nulls
        el_expr = pl.col('electricity').replace(0.0, None, default=pl.col('electricity'))
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
        'emission_scope', 'vehicle_type', 'energy_carrier',
    ]
    output_dimension_ids = [
        'emission_scope', 'vehicle_type', 'energy_carrier',
    ]
    default_unit = 'kt/a'
    quantity = 'emissions'

    def compute(self) -> ppl.PathsDataFrame:
        df = super().compute()
        df = df.filter(~pl.col(self.get_default_output_metric().column_id).is_null())
        return df


class TransportEmissions2kW(Node):
    def compute(self):
        enode = self.get_input_node(tag='emissions')
        edf = enode.get_output_pl(target_node=self)
        edf = edf.with_columns(emission_scope = pl.col('emission_scope').cast(pl.Categorical))
        edf = edf.rename({VALUE_COLUMN: 'emissions'})

        cnode = self.get_input_node(tag='consumption')
        cdf = cnode.get_output_pl(target_node=self)
        cdf = cdf.filter(pl.col('energy_carrier') != 'electricity')
        cdf = cdf.rename({VALUE_COLUMN: 'consumption'})

        fnode = self.get_input_node(tag='emission_factors')
        fdf = fnode.get_output_pl(target_node=self)
        fdf = fdf.rename({VALUE_COLUMN: 'factor'})

        cdf = cdf.paths.join_over_index(fdf)
        cdf = cdf.multiply_cols(['consumption', 'factor'], 'emissions_total')

        df = edf.filter((pl.col('energy_carrier') == 'electricity') |
                        (pl.col('emission_scope') == 'scope1'))

        edf = edf.filter(pl.col('emission_scope') == 'scope1')
        edf = edf.paths.join_over_index(cdf)

        edf = edf.subtract_cols(['emissions_total', 'emissions'], 'emissions_2kw')
        edf = edf.with_columns(emissions_2kw = pl.when(pl.col('emissions_2kw') < 0).then(0)
                                                 .when(pl.col('emissions_2kw').is_null()).then(0)
                                                 .otherwise(pl.col('emissions_2kw')))

        # print('emissions: %s' % edf._units['emissions'])
        # print('consumption: %s' % edf._units['consumption'])
        # print('factor: %s' % edf._units['factor'])
        # print('emissions_total: %s' % edf._units['emissions_total'])
        # print('emissions_2kw: %s' % edf._units['emissions_2kw'])

        edf = edf.drop(['consumption', 'factor', 'emissions_total', 'emissions'])
        edf = edf.with_columns(emission_scope = pl.lit('scope3').cast(pl.Categorical))

        edf = edf.rename({'emissions_2kw': VALUE_COLUMN})
        df = df.rename({'emissions': VALUE_COLUMN})
        df.extend(edf[df.columns])
        return df


class NonroadMachineryEmissions(Node):
    def compute(self) -> ppl.PathsDataFrame:
        nodes = list(self.input_nodes)
        efn = self.get_input_node(tag='emission_factor')
        efdf = efn.get_output_pl(target_node=self)
        fn = self.get_input_node(tag='fuel')
        fdf = fn.get_output_pl(target_node=self)
        nodes.remove(efn)
        nodes.remove(fn)
        efdf = efdf.rename({VALUE_COLUMN: 'EF'})
        fdf = fdf.rename({VALUE_COLUMN: 'Fuel'})

        df = fdf.paths.join_over_index(efdf, how='outer', index_from='union')
        df = df.multiply_cols(['Fuel', 'EF'], VALUE_COLUMN).drop_nulls().select_metrics(VALUE_COLUMN)
        df = convert_to_co2e(df, 'greenhouse_gases')
        df = df.ensure_unit(VALUE_COLUMN, self.get_default_output_metric().unit)

        df = self.add_nodes_pl(df, nodes)

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

        amount_node = self.get_input_node(tag='amount')
        adf = amount_node.get_output_pl(target_node=self)

        if not efdf.has_unit('emission_factor'):
            efdf = efdf.set_unit('emission_factor', 'dimensionless')
        efdf = extend_last_historical_value_pl(efdf, self.get_end_year())
        df = (
            adf.paths.join_over_index(efdf, how='left', index_from='union')
        )
        df = (
            df.multiply_cols([VALUE_COLUMN, 'emission_factor'], 'Emissions')
            .select([*df.get_meta().primary_keys, FORECAST_COLUMN, 'Emissions'])
        )

        fdf = extend_last_historical_value_pl(fdf, self.get_end_year())
        df = df.paths.join_over_index(fdf.select_metrics(['share_of_fossil_co2']))

        zdf = (
            df.filter(pl.col('greenhouse_gases').eq('co2'))
            .multiply_cols(['Emissions', 'share_of_fossil_co2'], 'fossil', df.get_unit('Emissions'))
            .with_columns((pl.col('Emissions') - pl.col('fossil')).alias('biogen'))
            .set_unit('biogen', df.get_unit('Emissions'))
        )

        fossil = zdf.select_metrics(['fossil']).rename({'fossil': 'Emissions'})
        biogen = (
            zdf.select_metrics(['biogen']).rename({'biogen': 'Emissions'})
            .with_columns(pl.lit('co2_biogen', dtype=pl.Categorical).alias('greenhouse_gases'))
        )

        df = df.select_metrics('Emissions').filter(~pl.col('greenhouse_gases').eq('co2'))
        meta = df.get_meta()
        df = ppl.to_ppdf(pl.concat([df, fossil, biogen]), meta=meta).rename({'Emissions': VALUE_COLUMN})
        return df


class SewageSludgeProcessingEmissions(SimpleNode):
    def compute(self) -> ppl.PathsDataFrame:
        df = self.get_input_dataset_pl()
        df = df.with_columns(pl.lit(False).alias(FORECAST_COLUMN))
        df = extend_last_historical_value_pl(df, self.get_end_year())
        ccs_node = self.get_input_node(tag='ccs_share')
        cdf = ccs_node.get_output_pl(target_node=self)
        cdf = cdf.rename({VALUE_COLUMN: 'CCSShare'}).ensure_unit('CCSShare', 'dimensionless')

        df = df.paths.join_over_index(cdf)
        # df = df.with_columns(pl.lit('scope1').alias('emission_scope')).add_to_index('emission_scope')

        df = df.with_columns([
            pl.when(
                pl.col('greenhouse_gases').eq('co2_biogen')
            ).then(pl.col('emissions') * pl.col('CCSShare') * -1).otherwise(pl.col('emissions')),
            pl.col('greenhouse_gases').cast(pl.String).replace({'co2_biogen': 'co2'}),
        ]).drop('CCSShare')

        df = df.with_columns([
            pl.col('greenhouse_gases').replace('co2', 'negative_emissions', default='scope1').alias('emission_scope'),
        ]).add_to_index('emission_scope')
        df = convert_to_co2e(df, 'greenhouse_gases')

        m = self.get_default_output_metric()
        df = df.rename({'emissions': m.column_id}).ensure_unit(m.column_id, m.unit)
        return df


class WastewaterTreatmentEmissions(Node):
    def compute(self) -> ppl.PathsDataFrame:
        pop_df = self.get_input_node(quantity='population').get_output_pl(self)
        cpop_df = self.get_input_datasets_pl(tag='population')[0]
        cpop_df = cpop_df.rename({cpop_df.metric_cols[0]: 'CatchmentPop'})
        efdf = self.get_input_datasets_pl(tag='emission_factor')[0]
        efdf = efdf.rename({efdf.metric_cols[0]: 'EF'})
        df = pop_df.paths.join_over_index(cpop_df)
        df = df.divide_cols(['CatchmentPop', VALUE_COLUMN], 'CPerPop')
        df = df.with_columns(pl.col('CPerPop').fill_null(strategy='forward'))
        df = df.multiply_cols(['CPerPop', VALUE_COLUMN], 'Pop').select_metrics('Pop')

        efdf = extend_last_historical_value_pl(efdf, self.get_end_year())
        df = efdf.paths.join_over_index(df, how='left', index_from='union')
        df = df.multiply_cols(['Pop', 'EF'], 'Emissions', out_unit=self.get_default_output_metric().unit)
        df = df.select_metrics('Emissions').rename({'Emissions': VALUE_COLUMN})
        df = convert_to_co2e(df, 'greenhouse_gases')
        df = df.with_columns(pl.lit('scope1').alias('emission_scope')).add_to_index('emission_scope')
        return df
