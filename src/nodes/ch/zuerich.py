from typing import TYPE_CHECKING, ClassVar, cast

from django.utils.translation import gettext_lazy as _

import polars as pl

from common import polars as ppl
from nodes.calc import convert_to_co2e, extend_last_historical_value_pl
from nodes.constants import (
    CONSUMPTION_FACTOR_QUANTITY,
    DEFAULT_METRIC,
    EMISSION_FACTOR_QUANTITY,
    EMISSION_QUANTITY,
    ENERGY_QUANTITY,
    FORECAST_COLUMN,
    MILEAGE_QUANTITY,
    POPULATION_QUANTITY,
    VALUE_COLUMN,
    YEAR_COLUMN,
)
from nodes.constraints.port_roles import PortRoleInferenceResult
from nodes.constraints.rules import AnyShapeRule, SameShapeRule
from nodes.defs.binding_def import DatasetBindingDef
from nodes.defs.port_def import InputPort, InputPortDeclaration, InputPortDef
from nodes.exceptions import NodeError
from nodes.node import Node, NodeMetric
from nodes.simple import AdditiveNode, MixNode, MultiplicativeNode, SimpleNode
from params.param import BoolParameter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polars.expr.expr import Expr

    from nodes.instance_graph import NodeMeta
    from nodes.units import Unit


class BuildingEnergy(AdditiveNode):
    energy_port = InputPort.one('energy', label=_('Building energy'))
    other_fuel_use_port = InputPort.one('other_fuel_use', label=_('Other fuel use'))
    input_port_declarations: ClassVar[tuple[InputPortDeclaration, ...]] = (energy_port, other_fuel_use_port)
    output_metrics = {ENERGY_QUANTITY: NodeMetric(unit='GWh/a', quantity=ENERGY_QUANTITY)}
    output_dimension_ids = [
        'energy_carrier',
    ]
    input_dimension_ids = [
        'energy_carrier',
    ]

    @classmethod
    def shape_rules(cls, meta: NodeMeta) -> tuple[AnyShapeRule, ...]:
        inputs = meta.input_port_ids_for_roles('energy', 'other_fuel_use')
        if not inputs:
            return ()
        return (SameShapeRule(inputs=inputs, output=meta.require_output_port('output').id),)

    @classmethod
    def infer_legacy_port_roles(cls, meta: NodeMeta, candidates: Sequence[InputPortDef]) -> PortRoleInferenceResult:
        result = PortRoleInferenceResult()
        for port in candidates:
            tags = {tag for binding in meta.bindings_for_port(port.id) for tag in binding.tags}
            if 'energy' in tags:
                result.classify(port, 'energy', "binding tag 'energy'")
            elif 'other_fuel_use' in tags:
                result.classify(port, 'other_fuel_use', "binding tag 'other_fuel_use'")
            else:
                result.refuse(port, 'not one of the two dataset inputs consumed by BuildingEnergy')
        return result

    def compute(self) -> ppl.PathsDataFrame:
        df = self.require_input(self.energy_port)
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
        df = df.with_columns([pl.col(col).alias(VALUE_COLUMN), pl.lit(value=False).alias(FORECAST_COLUMN)])
        df = df.select([YEAR_COLUMN, *meta.dim_ids, VALUE_COLUMN, FORECAST_COLUMN])

        odf = self.require_input(self.other_fuel_use_port)
        assert len(odf.metric_cols) == 1
        odf = odf.with_columns(pl.col(odf.metric_cols[0]) * -1)

        df = df.paths.add_df(odf, how='left')

        # tedf = (
        #     self.get_input_node(tag='transport_electricity').get_output_pl(target_node=self)
        # )
        # tedf = tedf.with_columns(pl.col(tedf.metric_cols[0]) * -1)
        # df = df.paths.add_df(tedf, how='left')
        return df


class BuildingFloorAreaHistorical(Node):
    input_port = InputPort.one('input', label=_('Building floor area'))
    input_port_declarations: ClassVar[tuple[InputPortDeclaration, ...]] = (input_port,)
    legacy_fixed_dataset_input_role = 'input'
    legacy_untagged_dataset_input_role = 'input'

    def compute(self) -> ppl.PathsDataFrame:
        df = self.require_input(self.input_port)
        df = df.with_columns(
            pl.col('building_use_extended').replace('residential', 'residential', default='nonresidential').alias('building_use')
        )
        df = df.add_to_index('building_use')
        df = df.paths.sum_over_dims(['building_use_extended'])
        df = df.rename({df.metric_cols[0]: self.get_default_output_metric().column_id})
        df = df.with_columns(pl.lit(value=False).alias(FORECAST_COLUMN))
        return df


class BuildingHeatHistorical(Node):
    cop_port = InputPort.one('heat_pump_cop', label=_('Heat pump COP'))
    consumption_port = InputPort.one('consumption', label=_('Energy consumption'))
    input_port_declarations = (cop_port, consumption_port)

    @classmethod
    def infer_legacy_port_roles(cls, meta: NodeMeta, candidates: Sequence[InputPortDef]) -> PortRoleInferenceResult:
        result = PortRoleInferenceResult()
        for port in candidates:
            tags = {tag for binding in meta.bindings_for_port(port.id) for tag in binding.tags}
            if 'heat_pump_cop' in tags:
                result.classify(port, 'heat_pump_cop', "binding tag 'heat_pump_cop'")
            elif 'consumption' in tags:
                result.classify(port, 'consumption', "binding tag 'consumption'")
            else:
                result.refuse(port, 'not a recognized BuildingHeatHistorical input')
        return result

    def compute(self) -> ppl.PathsDataFrame:
        cop_df = self.require_input(self.cop_port)
        cop_df = cop_df.rename({VALUE_COLUMN: 'HeatPumpCOP'})

        edf = self.require_input(self.consumption_port)
        edf = edf.filter(pl.col('energy_carrier') != 'electricity')
        edf = edf.paths.to_wide(only_category_names=True)
        edf = edf.paths.join_over_index(cop_df)
        gas_cols = [col for col in ('natural_gas', 'biogas', 'biogas_import') if col in edf.columns]
        edf = edf.with_columns([
            pl.sum_horizontal(gas_cols).alias('natural_gas'),
            (pl.col('environmental_heat') / (1 - 1 / pl.col('HeatPumpCOP'))).alias('heat_pumps'),
        ])
        edf = edf.set_unit('heat_pumps', edf.get_unit('environmental_heat'))
        gas_cols.remove('natural_gas')
        edf = edf.drop(['HeatPumpCOP', *gas_cols, 'environmental_heat'])
        renames = {col: 'Value@heating_system:%s' % col for col in edf.metric_cols}
        edf = edf.rename(renames).paths.to_narrow()
        return edf


class BuildingUsefulHeat(Node):
    energy_port = InputPort.one('energy', label=_('Energy'))
    cop_port = InputPort.one('cop', label=_('Coefficient of performance'))
    input_port_declarations = (energy_port, cop_port)
    legacy_input_port_roles_by_tag = {'energy': 'energy', 'cop': 'cop'}

    def compute(self) -> ppl.PathsDataFrame:
        df = self.require_input(self.energy_port)
        cdf = self.require_input(self.cop_port)
        cdf = cdf.rename({VALUE_COLUMN: 'COP'})
        df = df.paths.join_over_index(cdf)
        # Heat pump COP is already taken into account, so replace theh multiplier
        # with 1.0.
        df = df.with_columns(pl.when(pl.col('heating_system').eq('heat_pumps')).then(1.0).otherwise(pl.col('COP')).alias('COP'))
        m = self.get_default_output_metric()
        df = df.multiply_cols([VALUE_COLUMN, 'COP'], VALUE_COLUMN, m.unit).select_metrics([VALUE_COLUMN])
        return df


class BuildingHeatPerArea(Node):
    consumption_port = InputPort.one('consumption', label=_('Energy consumption'))
    floor_area_port = InputPort.one('floor_area', label=_('Floor area'))
    additive_port = InputPort.multi('additive', required=False, aggregation='sum', label=_('Additive inputs'))
    input_port_declarations = (consumption_port, floor_area_port, additive_port)
    legacy_input_port_roles_by_tag = {'consumption': 'consumption', 'floor_area': 'floor_area'}
    legacy_untagged_input_role = 'additive'

    def compute(self):
        edf = self.require_input(self.consumption_port)
        adf = self.require_input(self.floor_area_port)
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
        additive_df = self.get_input(self.additive_port)
        if additive_df is not None:
            df = df.paths.add_with_dims(additive_df, how='outer')
        return df


class BuildingGeneralElectricityEfficiency(AdditiveNode):
    dataset_port = InputPort.one('dataset', label=_('Electricity use distribution'))
    consumption_port = InputPort.one('consumption', label=_('Energy consumption'))
    heat_consumption_port = InputPort.one('heat_consumption', label=_('Heat consumption'))
    floor_area_port = InputPort.one('floor_area', label=_('Floor area'))
    additive_port = InputPort.multi('additive', required=False, aggregation='sum', label=_('Additive inputs'))
    input_port_declarations = (
        dataset_port,
        consumption_port,
        heat_consumption_port,
        floor_area_port,
        additive_port,
    )
    legacy_input_port_roles_by_tag = {
        'consumption': 'consumption',
        'heat_consumption': 'heat_consumption',
        'floor_area': 'floor_area',
    }
    legacy_untagged_dataset_input_role = 'dataset'
    legacy_untagged_input_role = 'additive'

    def compute(self):
        idf = self.require_input(self.dataset_port)
        edf = self.require_input(self.consumption_port)
        hdf = self.require_input(self.heat_consumption_port)
        adf = self.require_input(self.floor_area_port)

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
        additive_df = self.get_input(self.additive_port)
        if additive_df is not None:
            df = df.paths.add_with_dims(additive_df, how='outer')
        return df


class BuildingHeatUseMix(MixNode):
    consumption_port = InputPort.one('consumption', label=_('Energy consumption'))
    additive_port = InputPort.multi('additive', required=False, aggregation='sum', label=_('Additive inputs'))
    input_port_declarations = (consumption_port, additive_port)
    legacy_input_port_roles_by_tag = {'consumption': 'consumption'}
    legacy_untagged_input_role = 'additive'

    def compute(self):
        edf = self.require_input(self.consumption_port)

        sdf = edf.paths.sum_over_dims(['heating_system']).rename({VALUE_COLUMN: 'Total'})
        edf = edf.paths.join_over_index(sdf)
        edf = edf.divide_cols([VALUE_COLUMN, 'Total'], 'Share').select_metrics(['Share']).rename(dict(Share=VALUE_COLUMN))

        df = extend_last_historical_value_pl(edf, self.get_end_year())
        additive_df = self.get_input(self.additive_port)
        if additive_df is not None:
            df = df.paths.add_with_dims(additive_df, how='outer')
        df = self.normalize_mix(df)
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
    heat_pump_cop_port = InputPort.one('heat_pump_cop', label=_('Heat pump COP'))
    consumption_port = InputPort.one('consumption', label=_('Energy consumption'))
    biogas_share_port = InputPort.one('biogas_share', label=_('Biogas share'))
    input_port_declarations = (heat_pump_cop_port, consumption_port, biogas_share_port)
    legacy_input_port_roles_by_tag = {
        'heat_pump_cop': 'heat_pump_cop',
        'consumption': 'consumption',
        'biogas_share': 'biogas_share',
    }

    def compute(self):
        cop_df = self.require_input(self.heat_pump_cop_port)
        cop_df = cop_df.rename({VALUE_COLUMN: 'HeatPumpCOP'})

        edf = self.require_input(self.consumption_port)
        sdf = self.require_input(self.biogas_share_port)
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
    external_supply_port = InputPort.one('external_supply', label=_('External electricity supply'))
    consumption_port = InputPort.one('consumption', label=_('Electricity consumption'))
    general_mix_port = InputPort.one('general_mix', label=_('General production mix'))
    subsidized_mix_port = InputPort.one('subsidized_mix', label=_('Subsidized production mix'))
    external_mix_port = InputPort.one('external_mix', label=_('External production mix'))
    input_port_declarations = (
        external_supply_port,
        consumption_port,
        general_mix_port,
        subsidized_mix_port,
        external_mix_port,
    )
    legacy_input_port_roles_by_tag = {
        'external_supply': 'external_supply',
        'consumption': 'consumption',
        'general_mix': 'general_mix',
        'subsidized_mix': 'subsidized_mix',
        'external_mix': 'external_mix',
    }

    def compute(self) -> ppl.PathsDataFrame:
        external_df = self.require_input(self.external_supply_port).rename({'energy': 'ExternalEnergy'})

        energy_df = self.require_input(self.consumption_port)
        energy_metric = energy_df.metric_cols[0]
        energy_df = (
            energy_df.filter(~pl.col(FORECAST_COLUMN)).rename({energy_metric: 'TotalEnergy'}).paths.join_over_index(external_df)
        )

        energy_unit = energy_df.get_unit('TotalEnergy')
        energy_df = (
            energy_df
            .ensure_unit('ExternalEnergy', energy_unit)
            .with_columns((pl.col('TotalEnergy') - pl.col('ExternalEnergy')).alias('InternalEnergy'))
            .with_columns((pl.col('ExternalEnergy') / pl.col('TotalEnergy')).alias('ExternalTotal'))
            .with_columns((pl.col('InternalEnergy') / pl.col('TotalEnergy')).alias('InternalTotal'))
            .select([YEAR_COLUMN, FORECAST_COLUMN, 'ExternalTotal', 'InternalTotal'])
        )

        mix_df = self.require_input(self.general_mix_port)

        subsidized_df = (
            mix_df
            .filter(pl.col('electricity_source') == pl.lit('subsidized'))
            .select([YEAR_COLUMN, 'share'])
            .rename({'share': 'SubsidizedTotal'})
        )

        mix_df = (
            mix_df
            .filter(pl.col('electricity_source') != pl.lit('subsidized'))
            .rename({'share': 'InternalPercent'})
            .paths.join_over_index(subsidized_df)
            .paths.join_over_index(self.require_input(self.subsidized_mix_port))
            .rename({'share': 'SubsidizedPercent'})
            .paths.join_over_index(self.require_input(self.external_mix_port))
            .rename({'share': 'ExternalPercent'})
            .paths.join_over_index(energy_df)
            .with_columns(
                (
                    (
                        (pl.col('InternalPercent') + (pl.col('SubsidizedPercent') * (pl.col('SubsidizedTotal') / 100)))
                        * pl.col('InternalTotal')
                    )
                    + (pl.col('ExternalPercent') * pl.col('ExternalTotal'))
                ).alias(VALUE_COLUMN)
            )
            .set_unit(VALUE_COLUMN, '%')
            .select([YEAR_COLUMN, FORECAST_COLUMN, 'electricity_source', VALUE_COLUMN])
        )

        mix_df = extend_last_historical_value_pl(mix_df, self.get_end_year())
        return mix_df


class ElectricityProductionMixLegacy(MixNode):
    general_mix_port = InputPort.one('general_mix', label=_('General production mix'))
    subsidized_mix_port = InputPort.one('subsidized_mix', label=_('Subsidized production mix'))
    external_energy_port = InputPort.one('external_energy', label=_('Externally supplied energy'))
    consumption_port = InputPort.one('consumption', label=_('Electricity consumption'))
    additive_port = InputPort.multi('additive', required=False, aggregation='sum', label=_('Additive inputs'))
    input_port_declarations = (
        general_mix_port,
        subsidized_mix_port,
        external_energy_port,
        consumption_port,
        additive_port,
    )

    @classmethod
    def infer_legacy_port_roles(cls, meta: NodeMeta, candidates: Sequence[InputPortDef]) -> PortRoleInferenceResult:
        result = PortRoleInferenceResult()
        dataset_roles = iter(('general_mix', 'subsidized_mix', 'external_energy'))
        for port in candidates:
            bindings = meta.bindings_for_port(port.id)
            tags = {tag for binding in bindings for tag in binding.tags}
            if 'consumption' in tags:
                result.classify(port, 'consumption', "binding tag 'consumption'")
            elif bindings and all(isinstance(binding, DatasetBindingDef) for binding in bindings):
                role = next(dataset_roles, None)
                if role is None:
                    result.refuse(port, 'more than three positional datasets')
                else:
                    result.classify(port, role, f'the positional {role} dataset')
            else:
                result.classify(port, 'additive', 'an additional node input')
        return result

    def compute(self) -> ppl.PathsDataFrame:
        gen_mix_df = self.require_input(self.general_mix_port)
        sub_mix_df = self.require_input(self.subsidized_mix_port)
        ext_energy_df = self.require_input(self.external_energy_port)

        df = self.require_input(self.consumption_port)
        energy_metric = df.metric_cols[0]
        energy_unit = df.get_unit(energy_metric)

        df = df.filter(~pl.col(FORECAST_COLUMN))

        df = df.rename({energy_metric: 'TotalEnergy'}).ensure_unit('TotalEnergy', energy_unit)
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

        idf = df.select([YEAR_COLUMN, 'ExtEnergy', pl.lit('import').alias(es_dim)]).replace_meta(
            ppl.DataFrameMeta(units={'ExtEnergy': energy_unit}, primary_keys=[YEAR_COLUMN, es_dim])
        )

        gdf = gdf.paths.join_over_index(idf)
        gdf = gdf.select([YEAR_COLUMN, es_dim, pl.col('TotalEnergy') + pl.col('ExtEnergy').fill_null(0)])
        sum_df = gdf.group_by([YEAR_COLUMN]).agg(pl.sum('TotalEnergy').alias('YearSum')).sort(YEAR_COLUMN)
        sum_df = ppl.to_ppdf(sum_df, meta=ppl.DataFrameMeta(units={'YearSum': energy_unit}, primary_keys=[YEAR_COLUMN]))
        gdf = gdf.paths.join_over_index(sum_df)

        m = self.get_default_output_metric()
        gdf = gdf.divide_cols(['TotalEnergy', 'YearSum'], m.column_id, m.unit)
        dim_id = next(iter(self.output_dimensions.keys()))
        df = gdf.select([YEAR_COLUMN, pl.col(es_dim).alias(dim_id), m.column_id])

        df = df.filter(pl.col(m.column_id).is_not_null() & pl.col(m.column_id).is_not_nan())

        df = extend_last_historical_value_pl(df, self.get_end_year())

        additive_df = self.get_input(self.additive_port)
        if additive_df is not None:
            df = df.paths.add_with_dims(additive_df, how='outer')
        df = self.normalize_mix(df)

        return df


class GasGridMixin(Node):
    gas_mix_port: ClassVar[InputPortDeclaration]
    grid_share_port: ClassVar[InputPortDeclaration]

    def use_gas_grid(
        self,
        df: ppl.PathsDataFrame,
        *,
        gas_mix_df: ppl.PathsDataFrame | None = None,
        grid_share_df: ppl.PathsDataFrame | None = None,
    ) -> ppl.PathsDataFrame:
        df = df.paths.to_wide(only_category_names=True)
        df = df.with_columns([pl.col(col).fill_nan(0.0) for col in df.metric_cols])
        df = df.sum_cols(['natural_gas', 'biogas', 'biogas_import'], out_col='all_gas', skip_missing=True)

        if grid_share_df is None:
            grid_share_df = self.require_input(self.grid_share_port)
        sdf = grid_share_df
        sdf = sdf.select_metrics(sdf.metric_cols[0], rename='GridShare').ensure_unit('GridShare', '')

        if gas_mix_df is None:
            gas_mix_df = self.require_input(self.gas_mix_port)
        mdf = gas_mix_df
        mdf = mdf.ensure_unit(mdf.metric_cols[0], '')
        mdf = mdf.paths.to_wide(only_category_names=True)

        zdf = df.select(YEAR_COLUMN).join(mdf, on=YEAR_COLUMN, how='left').join(sdf, on=YEAR_COLUMN, how='left')
        zdf = zdf.with_columns(pl.col('GridShare').fill_null(0.0))

        def fc_only(col: str) -> Expr:
            own_supply = (1 - zdf['GridShare']) * pl.col(col)
            grid_supply = zdf['GridShare'] * zdf[col] * pl.col('all_gas')
            return pl.when(pl.col(FORECAST_COLUMN)).then(own_supply + grid_supply).otherwise(pl.col(col)).fill_nan(0.0).alias(col)

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
    base_mix_port = InputPort.one('base_mix', label=_('Base mix'))
    additive_port = InputPort.multi('additive', required=False, aggregation='sum', label=_('Additive inputs'))
    gas_mix_port = InputPort.optional('gas_mix', label=_('Gas mix'))
    grid_share_port = InputPort.optional('grid_share', label=_('Gas grid share'))
    input_port_declarations: ClassVar[tuple[InputPortDeclaration, ...]] = (
        base_mix_port,
        additive_port,
        gas_mix_port,
        grid_share_port,
    )
    export_additive_input_ports_as_multi = True
    additive_multi_input_excluded_tags = frozenset({'gas_mix', 'grid_share', 'non_additive'})

    allowed_parameters = [
        *MixNode.allowed_parameters,
        BoolParameter(local_id='use_gas_network', label=_('District heat uses gas grid mix')),
    ]

    @classmethod
    def shape_rules(cls, meta: NodeMeta) -> tuple[AnyShapeRule, ...]:
        inputs = meta.input_port_ids_for_roles('base_mix', 'additive')
        if not inputs:
            return ()
        return (SameShapeRule(inputs=inputs, output=meta.require_output_port('output').id),)

    @classmethod
    def infer_legacy_port_roles(cls, meta: NodeMeta, candidates: Sequence[InputPortDef]) -> PortRoleInferenceResult:
        result = PortRoleInferenceResult()
        for port in candidates:
            bindings = meta.bindings_for_port(port.id)
            tags = {tag for binding in bindings for tag in binding.tags}
            if 'gas_mix' in tags:
                result.classify(port, 'gas_mix', "binding tag 'gas_mix'")
            elif 'grid_share' in tags:
                result.classify(port, 'grid_share', "binding tag 'grid_share'")
            elif any(isinstance(binding, DatasetBindingDef) for binding in bindings):
                result.classify(port, 'base_mix', 'dataset binding')
            else:
                result.classify(port, 'additive', 'ordinary node binding')
        return result

    def compute(self) -> ppl.PathsDataFrame:
        mix_df = self.require_input(self.base_mix_port)
        assert len(mix_df.metric_cols) == 1
        assert len(mix_df.dim_ids) == 1
        m = self.get_default_output_metric()
        ec_dim_id, ec_dim = next(iter(self.input_dimensions.items()))
        ec_s = ec_dim.series_to_ids_pl(mix_df[mix_df.dim_ids[0]])
        df = mix_df.select([pl.col(YEAR_COLUMN), ec_s.alias(ec_dim_id), pl.col(mix_df.metric_cols[0]).alias(m.column_id)])
        df = extend_last_historical_value_pl(df, self.get_end_year())

        additive_df = self.get_input(self.additive_port)
        if additive_df is not None:
            df = df.paths.add_with_dims(additive_df, how='outer')
        df = self.normalize_mix(df)

        use_grid = self.get_parameter_value('use_gas_network', required=False)
        if use_grid:
            gas_mix_df = self.require_input(self.gas_mix_port)
            grid_share_df = self.require_input(self.grid_share_port)
            df = self.use_gas_grid(df, gas_mix_df=gas_mix_df, grid_share_df=grid_share_df)

        return df


class GasGridNode(AdditiveNode, GasGridMixin):
    additive_port = InputPort.multi('additive', required=True, aggregation='sum', label=_('Additive inputs'))
    gas_mix_port = InputPort.one('gas_mix', label=_('Gas mix'))
    grid_share_port = InputPort.one('grid_share', label=_('Gas grid share'))
    input_port_declarations = (additive_port, AdditiveNode.impute_port, gas_mix_port, grid_share_port)
    legacy_input_port_roles_by_tag = {'impute': 'impute', 'gas_mix': 'gas_mix', 'grid_share': 'grid_share'}
    legacy_untagged_input_role = 'additive'

    def compute(self) -> ppl.PathsDataFrame:
        df = super().compute()
        meta = df.get_meta()
        other_dims = df.dim_ids
        other_dims.remove('energy_carrier')
        other_dim_cats = df.select(other_dims).unique()
        dfs = []
        for row in other_dim_cats.iter_rows():
            filters = [pl.col(dim).eq(cat) for dim, cat in zip(other_dims, row, strict=False)]
            fdf = df.filter(pl.all_horizontal(filters)).drop(other_dims)
            fdf = self.use_gas_grid(fdf).with_columns([pl.lit(cat).alias(dim) for dim, cat in zip(other_dims, row, strict=False)])
            dfs.append(fdf)
        df = ppl.to_ppdf(pl.concat(dfs), meta=meta)
        return df


class EnergyProductionEmissionFactor(AdditiveNode):
    mix_port = InputPort.one('mix', label=_('Production mix'))
    ccs_port = InputPort.optional('ccs', label=_('Carbon capture share'))
    dataset_port = InputPort.one('dataset', label=_('Emission factor dataset'))
    emission_factor_port = InputPort.multi('emission_factor', required=False, label=_('Emission factor overrides'))
    input_port_declarations = (mix_port, ccs_port, dataset_port, emission_factor_port)
    legacy_input_port_roles_by_tag = {
        'mix': 'mix',
        'ccs': 'ccs',
        'emission_factor': 'emission_factor',
    }
    legacy_untagged_dataset_input_role = 'dataset'
    output_metrics = {EMISSION_FACTOR_QUANTITY: NodeMetric(unit='g/kWh', quantity=EMISSION_FACTOR_QUANTITY)}
    default_unit = 'g/kWh'

    def compute(self) -> ppl.PathsDataFrame:
        mix_df = self.require_input(self.mix_port)
        mix_df = mix_df.rename({mix_df.metric_cols[0]: 'Share'})

        ccs_df = self.get_input(self.ccs_port)
        if ccs_df is not None:
            ccs_df = ccs_df.rename({VALUE_COLUMN: 'CCS'}).ensure_unit('CCS', 'dimensionless')

        ef_df = self.require_input(self.dataset_port)
        if len(self.input_dimensions) != 1:
            raise NodeError(self, 'Must have exactly 1 input dimensions (%d given)' % len(self.input_dimensions))

        es_dim_id, es_dim = next(iter(self.input_dimensions.items()))
        ef_df = ef_df.with_columns([es_dim.series_to_ids_pl(ef_df[es_dim_id])])
        ef_df = ef_df.rename({ef_df.metric_cols[0]: 'EF'})

        for override_df in self.iter_inputs(self.emission_factor_port):
            node_df = override_df.select([
                YEAR_COLUMN,
                *override_df.dim_ids,
                pl.col(override_df.metric_cols[0]).alias('NodeEF'),
            ])
            assert set(ef_df.dim_ids) == set(node_df.dim_ids)
            ef_df = ef_df.paths.join_over_index(node_df, how='outer')
            ef_df = ef_df.with_columns([pl.col('EF').fill_null(pl.col('NodeEF'))]).drop('NodeEF')

        df = extend_last_historical_value_pl(ef_df, self.get_end_year())

        if ccs_df is not None:
            df = df.paths.join_over_index(ccs_df).with_columns(pl.col('CCS').fill_null(0.0))
            # df = df.multiply_cols(['EF', 'CCS'], 'EFRemaining', out_unit=df.get_unit('EF'))
            df = df.with_columns(
                pl
                .when(pl.col('energy_carrier').eq('natural_gas') & pl.col('emission_scope').eq('scope1'))
                .then(pl.col('EF') * (1 - pl.col('CCS')))
                .otherwise(pl.col('EF'))
                .alias('EF')
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
    dataset_port = InputPort.one('dataset', label=_('Emission factor dataset'))
    additive_port = InputPort.multi('additive', required=False, aggregation='sum', label=_('Additive inputs'))
    input_port_declarations = (dataset_port, additive_port)
    legacy_untagged_dataset_input_role = 'dataset'
    legacy_untagged_input_role = 'additive'
    input_dimension_ids = ['energy_carrier', 'emission_scope']
    output_dimension_ids = ['energy_carrier', 'emission_scope']

    def compute(self) -> ppl.PathsDataFrame:
        df = self.require_input(self.dataset_port)
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
            pl.lit(value=False).alias(FORECAST_COLUMN),
        ])

        df = df.rename({metric_col: VALUE_COLUMN}).drop_nulls()
        meta = df.get_meta()
        if dim.id not in meta.primary_keys:
            meta.primary_keys.append(dim.id)
        if YEAR_COLUMN not in meta.primary_keys:
            meta.primary_keys.append(YEAR_COLUMN)

        df = extend_last_historical_value_pl(df, self.get_end_year())

        additive = self.get_input(self.additive_port)
        if additive is not None:
            ndf = additive.ensure_unit(VALUE_COLUMN, meta.units[VALUE_COLUMN])
            ndf = ndf.rename({VALUE_COLUMN: '_Right'})
            df = df.paths.join_over_index(ndf, how='outer')
            df = df.with_columns(pl.col(VALUE_COLUMN).fill_null(0) + pl.col('_Right').fill_null(0)).drop('_Right')

        if df.paths.index_has_duplicates():
            dupes = df.group_by(df._primary_keys).agg(pl.count()).filter(pl.col('count') > 1)
            self.print(dupes)
            raise NodeError(self, 'Duplicate rows detected')
        return df


class EmissionFactorActivity(Node):
    energy_port = InputPort.one('energy', label=_('Energy'))
    emission_factor_port = InputPort.one('emission_factor', label=_('Emission factor'))
    input_port_declarations = (energy_port, emission_factor_port)
    legacy_input_port_roles_by_quantity = {
        ENERGY_QUANTITY: 'energy',
        EMISSION_FACTOR_QUANTITY: 'emission_factor',
    }
    output_metrics = {
        DEFAULT_METRIC: NodeMetric('kt/a', quantity=EMISSION_QUANTITY, column_id=VALUE_COLUMN),
    }
    # input_dimension_ids = ['energy_carrier']

    def compute(self) -> ppl.PathsDataFrame:
        edf = self.require_input(self.energy_port)
        edf = edf.rename({VALUE_COLUMN: 'Energy'})
        fdf = self.require_input(self.emission_factor_port)
        fdf = fdf.rename({VALUE_COLUMN: 'EF'})
        df = edf.paths.join_over_index(fdf, index_from='union')
        if df['EF'].has_nulls():
            self.print(df.filter(pl.col('EF').is_null()))
            raise NodeError(self, 'Emission factor not found for some categories')

        m = self.get_default_output_metric()
        df = df.multiply_cols(['Energy', 'EF'], m.column_id)
        df = df.ensure_unit(m.column_id, m.unit)
        meta = df.get_meta()
        zdf = (
            df
            .group_by([YEAR_COLUMN, *self.output_dimensions.keys()])
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

        pc_unit = cast('Unit', act_df.get_unit('Value') / pop_df.get_unit('Pop'))
        df = df.with_columns([
            (pl.col(VALUE_COLUMN) / pl.col('Pop')).alias('PerCapita'),
            (pl.col(FORECAST_COLUMN) | pl.col(FORECAST_COLUMN + '_right')).alias(FORECAST_COLUMN),
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
                pl.col(FORECAST_COLUMN) | pl.col(FORECAST_COLUMN + '_right').fill_null(value=False),
            ])
            df = df.select([YEAR_COLUMN, VALUE_COLUMN, FORECAST_COLUMN])
        df = ppl.to_ppdf(df, meta=meta)
        return df


class VehicleDatasetNode(AdditiveNode):  # Based on BuildingEnergy.
    output_metrics = {MILEAGE_QUANTITY: NodeMetric(unit='km/a', quantity=MILEAGE_QUANTITY)}
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
        df = df.with_columns([pl.col(col).alias(VALUE_COLUMN), pl.lit(value=False).alias(FORECAST_COLUMN)]).drop_nulls()
        df = df.select([YEAR_COLUMN, *meta.dim_ids, VALUE_COLUMN, FORECAST_COLUMN])
        # df = df.set_unit(VALUE_COLUMN, output_unit)

        df = extend_last_historical_value_pl(df, self.get_end_year())
        return df


class VehicleMileageHistorical(Node):
    input_port = InputPort.one('input', label=_('Mileage dataset'))
    input_port_declarations = (input_port,)
    legacy_untagged_dataset_input_role = 'input'
    output_dimension_ids = [
        'vehicle_type',
    ]
    input_dimension_ids = [
        'vehicle_type',
    ]

    def compute(self) -> ppl.PathsDataFrame:
        df = self.require_input(self.input_port)
        m = self.get_default_output_metric()
        unit = df.get_unit('mileage')
        if '[vehicle]' not in unit.dimensionality:
            unit = unit * self.context.unit_registry.parse_units('vehicle')
            df = df.set_unit('mileage', unit, force=True)
        df = df.rename({'mileage': m.column_id}).ensure_unit(m.column_id, m.unit)
        df = df.with_columns(pl.lit(value=False).alias(FORECAST_COLUMN))
        return df


class PassengerKilometers(Node):
    vehicle_mileage_port = InputPort.one('vehicle_mileage', label=_('Vehicle mileage'))
    occupancy_factor_port = InputPort.one('occupancy_factor', label=_('Occupancy factor'))
    input_port_declarations = (vehicle_mileage_port, occupancy_factor_port)
    legacy_input_port_roles_by_tag = {
        'vehicle_mileage': 'vehicle_mileage',
        'occupancy_factor': 'occupancy_factor',
    }
    input_dimension_ids = [
        'vehicle_type',
    ]
    output_dimension_ids = ['transport_mode']

    def compute(self) -> ppl.PathsDataFrame:
        vdf = self.require_input(self.vehicle_mileage_port)
        odf = self.require_input(self.occupancy_factor_port)

        tm_dim = self.output_dimensions[self.output_dimension_ids[0]]
        vt_dim = self.input_dimensions[self.input_dimension_ids[0]]
        vdf = vdf.with_columns([vt_dim.ids_to_groups(pl.col(vt_dim.id)).alias('vehicle_group')])
        vdf = (
            vdf
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
    passenger_kilometers_port = InputPort.one('passenger_kilometers')
    occupancy_factor_port = InputPort.one('occupancy_factor')
    mileage_historical_port = InputPort.one('mileage_historical')
    population_port = InputPort.one('population')
    additive_port = InputPort.multi('additive', required=False, aggregation='sum')
    input_port_declarations = (
        passenger_kilometers_port,
        occupancy_factor_port,
        mileage_historical_port,
        population_port,
        additive_port,
    )
    legacy_input_port_roles_by_tag = {
        'passenger_kilometers': 'passenger_kilometers',
        'occupancy_factor': 'occupancy_factor',
        'mileage_historical': 'mileage_historical',
        'population': 'population',
    }
    legacy_untagged_input_role = 'additive'

    def compute(self) -> ppl.PathsDataFrame:
        pdf = self.require_input(self.passenger_kilometers_port)
        odf = self.require_input(self.occupancy_factor_port)
        mdf = self.require_input(self.mileage_historical_port)
        popdf = self.require_input(self.population_port)
        popdf = popdf.rename({VALUE_COLUMN: 'Pop'})

        m = self.get_default_output_metric()
        pdf = pdf.rename({VALUE_COLUMN: 'Pkm'})
        odf = odf.rename({VALUE_COLUMN: 'OF'})
        pdf = pdf.paths.join_over_index(odf)
        pdf = pdf.divide_cols(['Pkm', 'OF'], 'Vkm')
        pdf = pdf.paths.join_over_index(popdf)
        pdf = pdf.divide_cols(['Vkm', 'Pop'], 'LocalTransportVkm', out_unit=m.unit).select_metrics(['LocalTransportVkm'])

        tm_dim = self.context.dimensions['transport_mode']
        vt_dim = self.context.dimensions['vehicle_type']

        mdf = mdf.with_columns([vt_dim.ids_to_groups(pl.col(vt_dim.id)).alias('vehicle_group')])
        mdf = (
            mdf
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
        additive = self.get_input(self.additive_port)
        return mdf if additive is None else mdf.paths.add_with_dims(additive, how='outer')


class VehicleEngineTypeSplit(MixNode):
    mileage_port = InputPort.one('mileage')
    additive_port = InputPort.multi('additive', required=False, aggregation='sum')
    input_port_declarations = (mileage_port, additive_port)
    legacy_input_port_roles_by_tag = {'mileage': 'mileage'}
    legacy_untagged_input_role = 'additive'

    def compute(self) -> ppl.PathsDataFrame:
        mdf = self.require_input(self.mileage_port)
        dim = self.input_dimensions['vehicle_type']
        mdf = mdf.with_columns(dim.ids_to_groups(pl.col('vehicle_type')).alias('group')).add_to_index('group')
        mdf = mdf.paths.calculate_shares(VALUE_COLUMN, 'Share', over_dims=['vehicle_type'])
        m = self.get_default_output_metric()
        mdf = mdf.select_metrics(['Share']).rename(dict(Share=m.column_id)).ensure_unit(m.column_id, m.unit)
        df = mdf.with_columns(pl.lit(value=False).alias(FORECAST_COLUMN))

        gdf = df.select(['vehicle_type', 'group']).unique()
        df = df.drop('group')
        df = extend_last_historical_value_pl(df, self.get_end_year())
        additive = self.get_input(self.additive_port)
        if additive is not None:
            df = df.paths.add_with_dims(additive, how='outer')

        df = ppl.to_ppdf(df.join(gdf, on='vehicle_type', how='left'), df.get_meta()).sort(YEAR_COLUMN).add_to_index('group')
        df = self.add_mix_normalized(df, [], over_dims=['vehicle_type'])
        df = df.drop('group')
        return df


class VehicleMileage(Node):
    population_port = InputPort.one('population')
    engine_type_split_port = InputPort.one('engine_type_split')
    mileage_per_inhabitant_port = InputPort.one('mileage_per_inhabitant')
    input_port_declarations = (population_port, engine_type_split_port, mileage_per_inhabitant_port)
    legacy_input_port_roles_by_tag = {
        'population': 'population',
        'engine_type_split': 'engine_type_split',
        'mileage_per_inhabitant': 'mileage_per_inhabitant',
    }

    def compute(self) -> ppl.PathsDataFrame:
        popdf = self.require_input(self.population_port)
        popdf = popdf.rename({VALUE_COLUMN: 'Pop'})

        etdf = self.require_input(self.engine_type_split_port)
        etdf = etdf.rename({VALUE_COLUMN: 'EngineTypeShare'})

        mdf = self.require_input(self.mileage_per_inhabitant_port)
        mdf = mdf.rename({VALUE_COLUMN: 'MileagePerPop'})

        m = self.get_default_output_metric()
        mdf = mdf.paths.join_over_index(popdf)
        mdf = mdf.multiply_cols(['MileagePerPop', 'Pop'], 'TotalMileage', out_unit=m.unit)

        vt_dim = self.context.dimensions['vehicle_type']
        tm_dim = self.context.dimensions['transport_mode']
        etdf = etdf.with_columns(vt_dim.ids_to_groups(pl.col('vehicle_type')).alias('vehicle_group'))
        etdf = etdf.with_columns(tm_dim.series_to_ids_pl(etdf['vehicle_group']).alias('transport_mode')).add_to_index(
            'transport_mode'
        )
        df = etdf.paths.join_over_index(mdf)
        df = df.multiply_cols(['TotalMileage', 'EngineTypeShare'], 'Mileage', out_unit=m.unit)
        df = df.select([YEAR_COLUMN, 'vehicle_type', FORECAST_COLUMN, pl.col('Mileage').alias(m.column_id)])
        return df


class TransportFuelFactor(AdditiveNode):
    input_port = InputPort.repeatable('input', min_count=2, default_count=2, label=_('Fuel factor metric'))
    input_port_declarations = (input_port,)
    legacy_untagged_dataset_input_role = 'input'
    output_metrics = {
        'Fuel': NodeMetric(unit='kg/vkm', quantity=CONSUMPTION_FACTOR_QUANTITY),
        'Electricity': NodeMetric(unit='kWh/vkm', quantity=CONSUMPTION_FACTOR_QUANTITY),
    }
    output_dimension_ids = [
        'energy_carrier',
        'vehicle_type',
    ]
    input_dimension_ids = [
        'energy_carrier',
        'vehicle_type',
    ]

    def compute(self) -> ppl.PathsDataFrame:
        df = None
        for port in self.iter_input_ports(self.input_port):
            value = self.require_input_port(port)
            if {'fuel', 'electricity'} <= set(value.metric_cols):
                df = value
                break
            identifier = port.definition.identifier if port.definition is not None else None
            if identifier is not None and len(value.metric_cols) == 1 and value.metric_cols[0] != identifier:
                value = value.rename({value.metric_cols[0]: str(identifier).lower()})
            df = value if df is None else df.paths.join_over_index(value, how='outer', index_from='union')
        if df is None:
            raise NodeError(self, 'TransportFuelFactor requires fuel and electricity input metrics')

        v_unit = self.context.unit_registry.parse_units('vehicle')

        df = df.select_metrics(['fuel', 'electricity'])
        e_m = self.output_metrics['Electricity']
        f_m = self.output_metrics['Fuel']

        exprs = []
        for col, m in (('electricity', e_m), ('fuel', f_m)):
            u = df.get_unit(col)
            if 'vehicle' not in u.dimensionality:
                df = df.set_unit(col, cast('Unit', u / v_unit), force=True)
            df = df.ensure_unit(col, m.unit).rename({col: m.column_id})
            df = df.with_columns(pl.col(m.column_id).fill_nan(None))
            exprs.append(pl.col(m.column_id).is_null() | pl.col(m.column_id).eq(0.0))

        df = df.filter(~pl.all_horizontal(exprs))
        df = df.with_columns(pl.lit(value=False).alias(FORECAST_COLUMN))
        df = extend_last_historical_value_pl(df, self.get_end_year())
        return df


class TransportEmissionFactor(Node):
    general_electricity_ef_port = InputPort.one('general_electricity_ef')
    electricity_consumption_factor_port = InputPort.one('electricity_consumption_factor')
    fuel_emission_factor_port = InputPort.one('fuel_emission_factor')
    input_port_declarations = (
        general_electricity_ef_port,
        electricity_consumption_factor_port,
        fuel_emission_factor_port,
    )
    legacy_input_port_roles_by_tag = {
        'general_electricity_ef': 'general_electricity_ef',
        'electricity_consumption_factor': 'electricity_consumption_factor',
        'fuel_emission_factor': 'fuel_emission_factor',
    }
    output_dimension_ids = ['emission_scope', 'vehicle_type', 'energy_carrier']

    def compute(self) -> ppl.PathsDataFrame:
        efdf = self.require_input(self.general_electricity_ef_port)
        efdf = efdf.rename({efdf.metric_cols[0]: 'EEF'})

        ecdf = self.require_input(self.electricity_consumption_factor_port)
        ecdf = ecdf.rename({ecdf.metric_cols[0]: 'EC'})

        m = self.get_default_output_metric()
        edf = ecdf.paths.join_over_index(efdf, index_from='union')
        edf = edf.multiply_cols(['EC', 'EEF'], 'EF', m.unit)
        # We only have CO2e for electricity, so pretend that it's just CO2 for now
        edf = edf.with_columns([
            pl.lit('co2').alias('greenhouse_gases'),
            pl.lit('electricity').alias('energy_carrier'),
        ]).add_to_index(['greenhouse_gases', 'energy_carrier'])
        edf = edf.select_metrics(['EF'])

        fdf = self.require_input(self.fuel_emission_factor_port)
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
    fuel_factor_port = InputPort.one('fuel_factor')
    emission_factor_port = InputPort.one('emission_factor')
    tank_respiration_port = InputPort.optional('tank_respiration')
    additive_port = InputPort.multi('additive', required=False, aggregation='sum')
    input_port_declarations = (fuel_factor_port, emission_factor_port, tank_respiration_port, additive_port)
    legacy_input_port_roles_by_tag = {
        'fuel_factor': 'fuel_factor',
        'emission_factor': 'emission_factor',
        'tank_respiration': 'tank_respiration',
        'additive': 'additive',
    }

    def compute(self) -> ppl.PathsDataFrame:
        # Read DF from the transport_fuel_factor node; contains interpolated and extended values to 2040.
        ffdf = self.require_input(self.fuel_factor_port)
        ffdf = ffdf.rename({VALUE_COLUMN: 'fuel'})

        # Read DF from the transport_emission_factors dataset; contains interpolated values, which are then extended to 2024.
        efdf = self.require_input(self.emission_factor_port)
        efdf = extend_last_historical_value_pl(efdf, efdf[YEAR_COLUMN].max())  # type: ignore
        eunit = efdf.get_unit('emission_factor')
        if 'vehicle' not in eunit.dimensionality:
            efdf = efdf.set_unit('emission_factor', 'kg/vkm', force=True)

        # Join the two DFs. By dropping nulls, DF is truncated to 2024. EFDF doesn't have energy carrier dimension.
        df = efdf.paths.join_over_index(ffdf, index_from='union').drop_nulls()

        # Divide EF (kg[ghg]/km) by FF (kg[fuel]/km), to obtain kg[ghg]/kg[fuel]. Filter to rows in which this result > 0. Sum
        # over energy carriers.
        df = df.filter(pl.col('fuel').gt(0))
        df = df.divide_cols(['emission_factor', 'fuel'], 'EFFuel')
        m = self.get_default_output_metric()
        df = df.filter(pl.col('EFFuel').gt(0)).select_metrics(['EFFuel']).rename(dict(EFFuel=m.column_id))
        df = df.paths.sum_over_dims(['energy_carrier'])

        # Read DF from transport_tank_respiration_for_fuel node; contains extended values to 2040.
        trdf = self.get_input(self.tank_respiration_port)
        if trdf is not None:
            trdf = trdf.rename({VALUE_COLUMN: 'TR'}).ensure_unit('TR', df.get_unit(m.column_id)).filter(~pl.col(FORECAST_COLUMN))

            # Join the DF, add TR to the metric column, and drop TR. DF contains dimension combos with differing last years
            # (historical or forecast).
            df = df.paths.join_over_index(trdf, how='outer')
            df = df.with_columns(pl.col(m.column_id).fill_null(0) + pl.col('TR').fill_null(0)).drop('TR')

        df = extend_last_historical_value_pl(df, self.get_end_year())

        additive = self.get_input(self.additive_port)
        if additive is not None:
            df = df.paths.add_with_dims(additive, how='outer')

        return df


class TransportElectricity(AdditiveNode):
    dataset_port = InputPort.one('dataset')
    additive_port = InputPort.multi('additive', required=False, aggregation='sum')
    input_port_declarations = (dataset_port, additive_port)
    legacy_untagged_dataset_input_role = 'dataset'
    legacy_untagged_input_role = 'additive'

    def compute(self) -> ppl.PathsDataFrame:
        df = self.require_input(self.dataset_port)
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
        additive = self.get_input(self.additive_port)
        if additive is not None:
            df = df.paths.add_with_dims(additive, how='outer')
        return df


class TransportEmissions(MultiplicativeNode):
    input_dimension_ids = [
        'emission_scope',
        'vehicle_type',
        'energy_carrier',
    ]
    output_dimension_ids = [
        'emission_scope',
        'vehicle_type',
        'energy_carrier',
    ]
    default_unit = 'kt/a'
    quantity = 'emissions'

    def compute(self) -> ppl.PathsDataFrame:
        df = super().compute()
        df = df.filter(~pl.col(self.get_default_output_metric().column_id).is_null())
        return df


class TransportEmissions2kW(Node):
    emissions_port = InputPort.one('emissions')
    consumption_port = InputPort.one('consumption')
    emission_factors_port = InputPort.one('emission_factors')
    input_port_declarations = (emissions_port, consumption_port, emission_factors_port)
    legacy_input_port_roles_by_tag = {
        'emissions': 'emissions',
        'consumption': 'consumption',
        'emission_factors': 'emission_factors',
    }

    def compute(self):
        edf = self.require_input(self.emissions_port)
        edf = edf.with_columns(emission_scope=pl.col('emission_scope').cast(pl.Categorical))
        edf = edf.rename({VALUE_COLUMN: 'emissions'})

        cdf = self.require_input(self.consumption_port)
        cdf = cdf.filter(pl.col('energy_carrier') != 'electricity')
        cdf = cdf.rename({VALUE_COLUMN: 'consumption'})

        fdf = self.require_input(self.emission_factors_port)
        fdf = fdf.rename({VALUE_COLUMN: 'factor'})

        cdf = cdf.paths.join_over_index(fdf)
        cdf = cdf.multiply_cols(['consumption', 'factor'], 'emissions_total')

        df = edf.filter((pl.col('energy_carrier') == 'electricity') | (pl.col('emission_scope') == 'scope1'))

        edf = edf.filter(pl.col('emission_scope') == 'scope1')
        edf = edf.paths.join_over_index(cdf)

        edf = edf.subtract_cols(['emissions_total', 'emissions'], 'emissions_2kw')
        edf = edf.with_columns(
            emissions_2kw=pl
            .when(pl.col('emissions_2kw') < 0)
            .then(0)
            .when(pl.col('emissions_2kw').is_null())
            .then(0)
            .otherwise(pl.col('emissions_2kw'))
        )

        # print('emissions: %s' % edf._units['emissions'])
        # print('consumption: %s' % edf._units['consumption'])
        # print('factor: %s' % edf._units['factor'])
        # print('emissions_total: %s' % edf._units['emissions_total'])
        # print('emissions_2kw: %s' % edf._units['emissions_2kw'])

        edf = edf.drop(['consumption', 'factor', 'emissions_total', 'emissions'])
        edf = edf.with_columns(emission_scope=pl.lit('scope3').cast(pl.Categorical))

        edf = edf.rename({'emissions_2kw': VALUE_COLUMN})
        df = df.rename({'emissions': VALUE_COLUMN})
        df.extend(edf[df.columns])
        return df


class NonroadMachineryEmissions(Node):
    emission_factor_port = InputPort.one('emission_factor')
    fuel_port = InputPort.one('fuel')
    additive_port = InputPort.multi('additive', required=False, aggregation='sum')
    input_port_declarations = (emission_factor_port, fuel_port, additive_port)
    legacy_input_port_roles_by_tag = {'emission_factor': 'emission_factor', 'fuel': 'fuel'}
    legacy_untagged_input_role = 'additive'
    quantity = 'emissions'

    def compute(self) -> ppl.PathsDataFrame:
        efdf = self.require_input(self.emission_factor_port)
        fdf = self.require_input(self.fuel_port)
        efdf = efdf.rename({VALUE_COLUMN: 'EF'})
        fdf = fdf.rename({VALUE_COLUMN: 'Fuel'})

        df = fdf.paths.join_over_index(efdf, how='outer', index_from='union')
        df = df.multiply_cols(['Fuel', 'EF'], VALUE_COLUMN).drop_nulls().select_metrics(VALUE_COLUMN)
        df = convert_to_co2e(df, 'greenhouse_gases')
        df = df.ensure_unit(VALUE_COLUMN, self.get_default_output_metric().unit)

        additive = self.get_input(self.additive_port)
        if additive is not None:
            df = df.paths.add_with_dims(additive, how='outer')

        return df


class WasteIncinerationEmissions(SimpleNode):
    fossil_share_port = InputPort.one('fossil_share')
    emission_factor_port = InputPort.one('emission_factor')
    amount_port = InputPort.one('amount')
    input_port_declarations = (fossil_share_port, emission_factor_port, amount_port)
    legacy_input_port_roles_by_tag = {
        'emission_factor': 'emission_factor',
        'amount': 'amount',
    }

    @classmethod
    def infer_legacy_port_roles(cls, meta: NodeMeta, candidates: Sequence[InputPortDef]) -> PortRoleInferenceResult:
        result = PortRoleInferenceResult()
        for port in candidates:
            bindings = meta.bindings_for_port(port.id)
            tags = {str(tag) for binding in bindings for tag in binding.tags}
            if 'amount' in tags:
                result.classify(port, 'amount', "binding tag 'amount'")
                continue
            dataset_ids = {
                str(binding.external_dataset_id)
                for binding in bindings
                if isinstance(binding, DatasetBindingDef) and binding.external_dataset_id is not None
            }
            if any('emission_factor' in dataset_id for dataset_id in dataset_ids):
                result.classify(port, 'emission_factor', 'the emission-factor dataset identifier')
            elif dataset_ids:
                result.classify(port, 'fossil_share', 'the waste-incineration composition dataset')
            else:
                result.refuse(port, 'not a recognized waste-incineration input')
        return result

    def compute(self) -> ppl.PathsDataFrame:
        fdf = self.require_input(self.fossil_share_port)
        efdf = self.require_input(self.emission_factor_port)
        adf = self.require_input(self.amount_port)

        if not efdf.has_unit('emission_factor'):
            efdf = efdf.set_unit('emission_factor', 'dimensionless')
        efdf = extend_last_historical_value_pl(efdf, self.get_end_year())
        df = adf.paths.join_over_index(efdf, how='left', index_from='union')
        df = df.multiply_cols([VALUE_COLUMN, 'emission_factor'], 'Emissions').select([
            *df.get_meta().primary_keys,
            FORECAST_COLUMN,
            'Emissions',
        ])

        fdf = extend_last_historical_value_pl(fdf, self.get_end_year())
        df = df.paths.join_over_index(fdf.select_metrics(['share_of_fossil_co2']))

        zdf = (
            df
            .filter(pl.col('greenhouse_gases').eq('co2'))
            .multiply_cols(['Emissions', 'share_of_fossil_co2'], 'fossil', df.get_unit('Emissions'))
            .with_columns((pl.col('Emissions') - pl.col('fossil')).alias('biogen'))
            .set_unit('biogen', df.get_unit('Emissions'))
        )

        fossil = zdf.select_metrics(['fossil']).rename({'fossil': 'Emissions'})
        biogen = (
            zdf
            .select_metrics(['biogen'])
            .rename({'biogen': 'Emissions'})
            .with_columns(pl.lit('co2_biogen', dtype=pl.Categorical).alias('greenhouse_gases'))
        )

        df = df.select_metrics('Emissions').filter(~pl.col('greenhouse_gases').eq('co2'))
        meta = df.get_meta()
        df = ppl.to_ppdf(pl.concat([df, fossil, biogen]), meta=meta).rename({'Emissions': VALUE_COLUMN})
        return df


class SewageSludgeProcessingEmissions(SimpleNode):
    dataset_port = InputPort.one('dataset')
    ccs_share_port = InputPort.one('ccs_share')
    input_port_declarations = (dataset_port, ccs_share_port)
    legacy_input_port_roles_by_tag = {'ccs_share': 'ccs_share'}
    legacy_untagged_dataset_input_role = 'dataset'

    def compute(self) -> ppl.PathsDataFrame:
        df = self.require_input(self.dataset_port)
        df = df.with_columns(pl.lit(value=False).alias(FORECAST_COLUMN))
        df = extend_last_historical_value_pl(df, self.get_end_year())
        cdf = self.require_input(self.ccs_share_port)
        cdf = cdf.rename({VALUE_COLUMN: 'CCSShare'}).ensure_unit('CCSShare', 'dimensionless')

        df = df.paths.join_over_index(cdf)
        # df = df.with_columns(pl.lit('scope1').alias('emission_scope')).add_to_index('emission_scope')

        df = df.with_columns([
            pl
            .when(pl.col('greenhouse_gases').eq('co2_biogen'))
            .then(pl.col('emissions') * pl.col('CCSShare') * -1)
            .otherwise(pl.col('emissions')),
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
    population_port = InputPort.one('population')
    catchment_population_port = InputPort.one('catchment_population')
    emission_factor_port = InputPort.one('emission_factor')
    input_port_declarations = (population_port, catchment_population_port, emission_factor_port)
    legacy_input_port_roles_by_tag = {'emission_factor': 'emission_factor'}

    @classmethod
    def infer_legacy_port_roles(cls, meta: NodeMeta, candidates: Sequence[InputPortDef]) -> PortRoleInferenceResult:
        result = PortRoleInferenceResult()
        for port in candidates:
            bindings = meta.bindings_for_port(port.id)
            tags = {str(tag) for binding in bindings for tag in binding.tags}
            if 'emission_factor' in tags:
                result.classify(port, 'emission_factor', "binding tag 'emission_factor'")
            elif 'population' in tags and any(isinstance(binding, DatasetBindingDef) for binding in bindings):
                result.classify(port, 'catchment_population', "dataset binding tag 'population'")
            else:
                result.classify(port, 'population', 'the population node input')
        return result

    def compute(self) -> ppl.PathsDataFrame:
        pop_df = self.require_input(self.population_port)
        cpop_df = self.require_input(self.catchment_population_port)
        cpop_df = cpop_df.rename({cpop_df.metric_cols[0]: 'CatchmentPop'})
        efdf = self.require_input(self.emission_factor_port)
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
