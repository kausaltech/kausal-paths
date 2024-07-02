from nodes.calc import convert_to_co2e
from params.param import NumberParameter, StringParameter
import polars as pl

from common import polars as ppl
from .constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN, TIME_INTERVAL
from .node import Node
from .exceptions import NodeError
from nodes.actions.energy_saving import CfFloorAreaAction
from nodes.simple import MultiplicativeNode, AdditiveNode, SimpleNode
from .units import unit_registry


class FloorAreaNode(MultiplicativeNode):  # FIXME Rebuild this with modern tools
    explanation = 'Floor area node takes in actions and calculates the floor area impacted.'
    output_dimension_ids = ['action', 'building_energy_class', 'emission_sectors']  # FIXME Generalise and remove emission_sectors
    input_dimension_ids = ['building_energy_class', 'emission_sectors']


    def include_custom_dimension(self, df: ppl.PathsDataFrame):  # Dimension must be explained in column name in the right syntax
        df = df.paths.to_wide()  # Make column names consistent
        for s in df.columns:
            arr = s.split('@')
            if len(arr) > 2:
                s2 = '@'.join(arr[:2]) + '/' + ''.join(arr[2:])
                df = df.rename({s: s2})
        df = df.paths.to_narrow()

        return df

    def compute(self):
        nodes: list(Node) = []
        actions: list(CfFloorAreaAction) = []
        for node in self.get_input_nodes():
            if isinstance(node, CfFloorAreaAction):
                actions += [node]
            else:
                nodes += [node]
        if len(nodes) == 1:
            df: ppl.PathsDataFrame = nodes.pop(0).get_output_pl(target_node=self)
        else:
            raise NodeError(self, 'Must have exactly one upstream node for floor area')

        # Existing (old) and new floor area in baseline
        flhv = df.get_last_historical_values()
        flhv = flhv.rename({flhv.metric_cols[0]: 'floor_old'})
        df_bau = df.paths.join_over_index(flhv.drop([YEAR_COLUMN, FORECAST_COLUMN]))
        df_bau = df_bau.with_columns(
            pl.when(pl.col(FORECAST_COLUMN)).then(pl.col('floor_old'))
            .otherwise(pl.col(VALUE_COLUMN)).alias('floor_old')
            )
        df_bau = df_bau.with_columns((pl.col(VALUE_COLUMN) - pl.col('floor_old')).alias('floor_new'))
        # FIXME Bubblegum fix for wrong unit treatment in diff:
        df_bau = df_bau.set_unit('floor_new', df_bau.get_unit('floor_old') * unit_registry(TIME_INTERVAL))
        df_bau = df_bau.diff('floor_new').with_columns(pl.col('floor_new').fill_null(0))
        df_bau = df_bau.drop(VALUE_COLUMN)

        # Add or update dimension building_energy_class
        if 'building_energy_class' in df.dim_ids:
            df_bau = df_bau.with_columns((
                pl.when(pl.col('building_energy_class')
                        .eq('existing')).then(pl.col('floor_old'))
                        .otherwise(pl.col('floor_new')).alias('floor_old')))
            df_bau = df_bau.drop('floor_new')
            df_bau = df_bau.rename({'floor_old': 'floor_area'})
        else:
            col = 'floor_area@building_energy_class:'
            df_bau = df_bau.rename({
                'floor_old': col + 'existing',
                'floor_new': col + 'new'})
            df_bau = self.include_custom_dimension(df_bau)

        df_out = None
        for action in actions:
            df = action.get_output_pl(target_node=self)
            df = df.ensure_unit('triggered', 'dimensionless')
            df = df.ensure_unit('compliant', 'dimensionless')

            df = df.paths.join_over_index(df_bau)
            df = df.with_columns((
                pl.when(pl.col('building_energy_class').eq(pl.lit('new')))
                .then(pl.lit(1.0)).otherwise(pl.col('triggered')).alias('triggered')))

            df = df.multiply_cols(['floor_area', 'triggered', 'compliant'], 'floor_area')

            df = df.rename({'floor_area': 'floor_area@action:' + action.id})
            df = df.drop(['triggered', 'compliant'])
            df = self.include_custom_dimension(df)

            if df_out is None:
                df_out = df
            else:
                meta = df.get_meta()
                df_out = pl.concat([df_out, df], rechunk=True)
                df_out = ppl.to_ppdf(df_out, meta)

        df_out = df_out.ensure_unit('floor_area', self.unit)
        df_out = df_out.with_columns(pl.col('floor_area').alias(VALUE_COLUMN))

        m = self.get_default_output_metric()
        df_out = df_out.select_metrics(['floor_area']).rename({'floor_area': m.column_id})

        return df_out


class CfNode(FloorAreaNode):
    '''
    Consumption factor (CF) describes the energy saving caused by the action.
    There must be at least one action of type energy_saving.CfFloorAreaAction.
    '''
    output_dimension_ids = ['action', 'building_energy_class', 'emission_sectors']
    input_dimension_ids = ['building_energy_class', 'emission_sectors']

    def compute(self):
        nodes: list(Node) = []
        actions: list(CfFloorAreaAction) = []
        for node in self.get_input_nodes():
            if isinstance(node, CfFloorAreaAction):
                actions += [node]
            else:
                nodes += [node]

        assert len(actions) > 0

        df = None
        for action in actions:
            df_a = action.get_output_pl(target_node=self)

            if df is None:
                df = df_a
            else:
                df = df_a.paths.join_over_index(df, index_from='union')
                df = df.with_columns(pl.col(VALUE_COLUMN).fill_null(pl.lit(0)))

            col = VALUE_COLUMN + '@action:' + action.id
            df = df.with_columns(pl.col(VALUE_COLUMN).alias(col))
            df = df.drop(VALUE_COLUMN)

        df = self.include_custom_dimension(df)

        # Inputs nodes are baseline but not required.
        # If actions are not in the same units as the baseline, they are assumed to be relative values.
        if len(nodes) > 0:
            df_bau = self.add_nodes_pl(None, nodes=nodes)
            df = df.paths.join_over_index(df_bau)
            sub = self.is_compatible_unit(df.get_unit(VALUE_COLUMN), df.get_unit(VALUE_COLUMN + '_right'))
            if sub:
                assert sub is False  # Because the use case is unclear
                df = df.subtract_cols([VALUE_COLUMN + '_right', VALUE_COLUMN], VALUE_COLUMN)
            else:
                df = df.multiply_cols([VALUE_COLUMN + '_right', VALUE_COLUMN], VALUE_COLUMN)
            df = df.drop(VALUE_COLUMN + '_right')

        df = df.ensure_unit(VALUE_COLUMN, self.unit)

        return df


class EnergyNode(MultiplicativeNode):
    '''
    Takes the floor area and consumption factor categorized by building energy class and action.

    This energy saving is accumulated over time to reflect the situation that
    the energy use of a building stays constant after renovation.
    However, accumulation can be prevented by using the parameter not_cumulated.
    '''

    allowed_parameters = MultiplicativeNode.allowed_parameters + [
        StringParameter(
            local_id='not_cumulated',
            description='Action that is not cumulated',
            is_customizable=False,
        )]

    input_dimension_ids = ['action', 'building_energy_class', 'emission_sectors']
    output_dimension_ids = ['action', 'building_energy_class', 'emission_sectors']

    def compute(self):
        df = super().compute()
        df = df.with_columns([pl.col(VALUE_COLUMN).alias('cumulated')])
        df = df.cumulate('cumulated')

        not_cumulated = self.get_parameter_value('not_cumulated', required=False)
        if not_cumulated is not None:
            df = df.with_columns(  # FIXME fails to pick non-cumulated action
                pl.when(pl.col('action') == not_cumulated)
                .then(pl.col(VALUE_COLUMN))
                .otherwise(pl.col('cumulated')).alias(VALUE_COLUMN)
                )

        df = df.drop(['cumulated'])

        df = df.ensure_unit(VALUE_COLUMN, self.unit)
        return df


class HistoricalNode(AdditiveNode):
    def compute(self) -> ppl.PathsDataFrame:
        df = super().compute()
        df = df.filter(pl.col(FORECAST_COLUMN) == False)
        return df


class CCSNode(SimpleNode):
    allowed_parameters = [
        NumberParameter('capture_efficiency', unit_str='%', is_customizable=True),
        NumberParameter('storage_efficiency', unit_str='%', is_customizable=True),
    ]
    def compute(self) -> ppl.PathsDataFrame:
        df = self.get_input_node(tag='emissions').get_output_pl(target_node=self)
        df = df.rename({VALUE_COLUMN: 'Emissions'})

        sdf = self.get_input_node(tag='ccs_share').get_output_pl(target_node=self)
        sdf = sdf.rename({VALUE_COLUMN: 'CCSShare'}).ensure_unit('CCSShare', 'dimensionless')

        df = df.paths.join_over_index(sdf)
        df = df.with_columns(pl.col('CCSShare').fill_null(0.0))

        capt_eff = self.get_parameter_value('capture_efficiency', units=True).to('dimensionless').m
        df = df.multiply_cols(['Emissions', 'CCSShare'], 'Captured')
        df = df.with_columns(
            pl.when(pl.col('greenhouse_gases').is_in(('co2', 'co2_biogen')))
                .then(pl.col('Captured') * capt_eff).otherwise(0.0)
        )

        storage_eff = self.get_parameter_value('storage_efficiency', units=True).to('dimensionless').m
        u = df.get_unit('Captured')
        df = df.with_columns([
            (pl.col('Captured') * storage_eff).alias('Stored'),
            (pl.col('Emissions') - pl.col('Captured')).alias('Remaining'),
        ]).with_columns([
            (pl.col('Captured') - pl.col('Stored')).alias('StorageLoss')
        ]).set_unit('Remaining', u).set_unit('StorageLoss', u)

        m = self.get_default_output_metric()
        rdf = (
            df.select_metrics('Remaining', rename=m.column_id)
            .with_columns(pl.lit('scope1').alias('emission_scope'))
        )
        sdf = (
            df.select_metrics('Stored', rename=m.column_id).filter(
                pl.col('greenhouse_gases').eq('co2_biogen')
            ).with_columns([
                pl.lit('negative_emissions').alias('emission_scope'),
                # use co2 to be able to convert to GWP
                pl.lit('co2', dtype=pl.Categorical).alias('greenhouse_gases'),
                (-pl.col(m.column_id)).alias(m.column_id)
            ])
        )
        ldf = (
            df.select_metrics('StorageLoss', rename=m.column_id)
            .with_columns(pl.lit('scope3').alias('emission_scope'))
        )

        df = ppl.to_ppdf(pl.concat([rdf, sdf, ldf]), rdf.get_meta()).add_to_index('emission_scope')
        df = convert_to_co2e(df, 'greenhouse_gases')
        df = df.ensure_unit(m.column_id, m.unit)
        return df
