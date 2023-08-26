import functools
from common.perf import PerfCounter
from nodes.calc import convert_to_co2e, extend_last_historical_value_pl, nafill_all_forecast_years
from params.param import Parameter, BoolParameter, NumberParameter, ParameterWithUnit, StringParameter
from typing import List, ClassVar, Tuple
import polars as pl
import pandas as pd
import pint

from common.i18n import TranslatedString
from common import polars as ppl
from .constants import FORECAST_COLUMN, MIX_QUANTITY, NODE_COLUMN, VALUE_COLUMN, YEAR_COLUMN, DEFAULT_METRIC
from .node import Node, NodeMetric
from .exceptions import NodeError
from nodes.actions.energy_saving import UsBuildingAction
from nodes.simple import MultiplicativeNode


class FloorAreaNode(MultiplicativeNode):
    '''
    Floor area splits into 1+2+4 categories based on building energy class:
    # all: all floor area
    ## floor_old: existing floor area built by the last historical year
    ### renovated: floor area that is triggered to renovation (increases yearly)
    ### regular: the remaining existing floor area (stays constant in future years)
    ## floor_new: floor area that is built after the last historical year (cumulative)
    ### compliant: share of new floor area that follows the stricter energy efficiency
    ### non_compliant: share that does not follow the stricter energy efficiency
    '''
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
        actions: list(UsBuildingAction) = []
        for node in self.get_input_nodes():
            if isinstance(node, UsBuildingAction):
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
        df_bau = df_bau.set_unit('floor_new', df_bau.get_unit('floor_old'))
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


class EuiNode(FloorAreaNode):
    '''
    Consumption factor has 2 + 3 * i combined categories for action * building_energy class:
    # none * all: the BAU CF
    # none * regular: the difference between BAU CF and CF for regular old buildings (0 by definition)
    # action_i * renovated: the difference between BAU CF and CF of action_i
    # action_i * compliant: same as action_i * renovated
    # action_i * non_compliant: same as none * regular
    '''
    output_dimension_ids = ['building_energy_class', 'action', 'emission_sectors']
    input_dimension_ids = ['building_energy_class', 'emission_sectors']

    def compute(self):
        nodes: list(Node) = []
        actions: list(UsBuildingAction) = []
        for node in self.get_input_nodes():
            if isinstance(node, UsBuildingAction):
                actions += [node]
            else:
                nodes += [node]

        assert len(nodes) == 0

        df = self.get_input_dataset_pl(required=True)
        df = extend_last_historical_value_pl(df, self.get_end_year())
        df = df.rename({VALUE_COLUMN: 'bau'})

        for action in actions:
            df_a = action.get_output_pl(target_node=self)
            df_a = df_a.rename({VALUE_COLUMN: 'improvement'})
            df_a = df_a.ensure_unit('improvement', 'dimensionless')

            df = df_a.paths.join_over_index(df, index_from='union')
            df = df.with_columns(pl.col('improvement').fill_null(pl.lit(0)))
            df = df.multiply_cols(['improvement', 'bau'], 'improvement')

            col = 'consumption_factor@action:' + action.id
            df = df.with_columns(pl.col('improvement').alias(col))
            df = df.drop('improvement')

        df = df.drop('bau')

        df = self.include_custom_dimension(df)

        df = df.rename({'consumption_factor': VALUE_COLUMN})
        df = df.ensure_unit(VALUE_COLUMN, self.unit)
        print(df)
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


