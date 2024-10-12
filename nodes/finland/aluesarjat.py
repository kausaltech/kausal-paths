from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import pandas as pd
import pint_pandas
import polars as pl

from nodes.calc import extend_last_historical_value, extend_last_historical_value_pl
from nodes.constants import CONSUMPTION_FACTOR_QUANTITY, ENERGY_QUANTITY, FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.exceptions import NodeError
from nodes.node import Node, NodeMetric
from nodes.simple import AdditiveNode, DivisiveNode, SimpleNode

if TYPE_CHECKING:
    from common import polars as ppl

INDUSTRY_USES = '''
Teollisuuden ja kaivannaistoiminnan rakennukset
Varastorakennukset
Muut rakennukset
'''.strip().splitlines()

RESIDENTIAL_USES = '''
Omakoti- ja paritalot
Rivitalot
Kerrostalot
'''.strip().splitlines()

SERVICE_USES = '''
Asuntolarakennukset ja erityisryhmien asuinrakennukset
Liikerakennukset
Toimistorakennukset
Liikenteen rakennukset
Hoitoalan rakennukset
Kokoontumisrakennukset
Opetusrakennukset
Energiahuoltorakennukset
Yhdyskuntatekniikan rakennukset
Pelastustoimen rakennukset
'''.strip().splitlines()

BUILDING_USE_MAP = {}
for use in RESIDENTIAL_USES:
    BUILDING_USE_MAP[use] = 'Asuminen'
for use in SERVICE_USES:
    BUILDING_USE_MAP[use] = 'Palvelut'
for use in INDUSTRY_USES:
    BUILDING_USE_MAP[use] = 'Teollisuus'


HEAT_SOURCE_MAP = {
    'Kauko- tai aluelämpö': 'Kaukolämpö',
    'Kivihiili': 'Öljylämmitys',
    'Maalämpö': 'Maalämpö',
    'Muu tai tuntematon': 'Muu',
    'Puu, turve': 'Muu',
    'Sähkö': 'Sähkölämmitys',
    'Öljy, kaasu': 'Öljylämmitys',
}


FLOOR_AREA = 'floor_area'
HEAT_CONSUMPTION = 'heat_consumption'


class BuildingStock(AdditiveNode):
    output_metrics = {
        FLOOR_AREA: NodeMetric(unit='m**2', quantity='floor_area')
    }
    input_datasets = [
        'helsinki/aluesarjat/02um_rakennukset_lammitys'
    ]
    output_dimension_ids = [
        'building_heat_source',
        'building_use',
    ]
    input_dimension_ids = [
        'building_heat_source',
        'building_use',
    ]
    global_parameters = ['municipality_name']

    def compute(self) -> pd.DataFrame:
        df = self.get_input_dataset()
        muni = self.get_global_parameter_value('municipality_name')
        if 'Alue' in df.columns:
            df = df.loc[df.Alue == muni]
        df = df.loc[
            (df.Tiedot == 'Kerrosala (m2)') &
            (df['Rakennuksen käyttötarkoitus'] != 'Asuinrakennukset yhteensä')
        ]

        hs_dim = self.output_dimensions['building_heat_source']
        df[hs_dim.id] = hs_dim.series_to_ids(
            df['Rakennuksen lämmitysaine'].map(HEAT_SOURCE_MAP)
        )
        use_dim = self.output_dimensions['building_use']
        df[use_dim.id] = use_dim.series_to_ids(
            df['Rakennuksen käyttötarkoitus'].map(BUILDING_USE_MAP)
        )

        df[YEAR_COLUMN] = df['Vuosi'].astype(int)
        m = self.output_metrics[FLOOR_AREA]
        s = df.groupby(['building_heat_source', 'building_use', YEAR_COLUMN])['value'].sum()
        s = m.ensure_output_unit(self, s)
        df = pd.DataFrame(data=s.values, index=s.index, columns=[VALUE_COLUMN])
        df[VALUE_COLUMN] = df[VALUE_COLUMN].astype(s.dtype)
        df[FORECAST_COLUMN] = False

        df = extend_last_historical_value(df, self.context.model_end_year)
        df = self.add_nodes(df, self.input_nodes)
        return df


class FutureBuildingStock(SimpleNode):
    """Calculate the new floor area of future buildings based on the 10-year average
    per respective new population; calculate this separately for different building types
    (unless floor area decreases) and assume the same ratio will hold in the future.
    """
    output_metrics = {
        FLOOR_AREA: NodeMetric(unit='m**2', quantity='floor_area')
    }
    input_dimension_ids = [
        'building_heat_source',
        'building_use',
    ]
    output_dimension_ids = [
        'building_heat_source',
        'building_use',
    ]

    def compute(self) -> pd.DataFrame:
        # First compute m2 / capita
        hist_node = self.get_input_node(tag='existing_floor_area')
        hist_df = hist_node.get_output()

        pop_node = self.get_input_node(quantity='population')
        pop_df = pop_node.get_output()

        df: pd.DataFrame = hist_df
        df = df.rename(columns={VALUE_COLUMN: 'Area'})
        area_dt = df.dtypes['Area']
        pop_dt = pop_df.dtypes[VALUE_COLUMN]

        last_hist_year = df.loc[~df.Forecast].index.get_level_values(YEAR_COLUMN).max()
        df = df.drop(columns=FORECAST_COLUMN)
        df = df.unstack(self.output_dimension_ids)  # type: ignore
        df['PopDiff'] = pop_df[VALUE_COLUMN]
        # Count the average from the last 10 years
        df = df.loc[(df.index >= last_hist_year - 10) & (df.index <= last_hist_year)]
        df = df.diff().dropna().cumsum()
        pop = df.pop('PopDiff').astype(pop_dt)
        pc_dt = pint_pandas.PintType(area_dt.units / pop_dt.units)
        for col in list(df.columns):
            df[col] = (df[col].astype(area_dt) / pop)

        total_sum = df.sum(axis=1).astype(pc_dt)
        # Remove the building types that have been decreasing
        df = df.clip(lower=0)
        area_sum = df.sum(axis=1).astype(pc_dt)
        # Count the ratios of new area per building type
        df = df.div(area_sum, axis=0)
        # Scale back to total sum -> and there it is
        area_per_new_cap_df = df.mul(total_sum, axis=0)

        # Now look into the future
        pop_diff = pop_df.loc[pop_df.index >= last_hist_year, VALUE_COLUMN].diff().dropna()
        df = area_per_new_cap_df
        future_index = pd.RangeIndex(last_hist_year + 1, self.context.model_end_year + 1, name=YEAR_COLUMN)
        df = df.reindex(df.index.append(future_index))
        df = df.ffill()
        df = df.mul(pop_diff, axis=0).dropna().cumsum()
        df = df.astype(area_dt)

        df = cast(pd.DataFrame, df.stack(cast(Sequence[str], self.output_dimension_ids), future_stack=True))  # noqa: PD013
        assert isinstance(df, pd.DataFrame)
        df = df.rename(columns={'Area': VALUE_COLUMN})
        df[FORECAST_COLUMN] = True

        nodes = self.input_nodes.copy()
        nodes.remove(hist_node)
        nodes.remove(pop_node)
        assert isinstance(df, pd.DataFrame)
        # self.add_nodes(df, nodes)  FIXME

        return df


class BuildingHeatPredict(Node):
    output_metrics = {
        HEAT_CONSUMPTION: NodeMetric(unit='GWh/a', quantity=ENERGY_QUANTITY)
    }
    output_dimension_ids = [
        'building_heat_source',
        'building_use',
    ]

    def compute(self) -> pd.DataFrame:
        # First compute GWh / m2
        area_node = self.get_input_node(quantity=FLOOR_AREA)
        area_df = area_node.get_output()
        area_df = area_df.sort_index()

        heat_node = self.get_input_node(quantity=ENERGY_QUANTITY)
        heat_df = heat_node.get_output()
        assert isinstance(heat_df.index, pd.MultiIndex)
        heat_df.index = heat_df.index.reorder_levels(area_df.index.names)
        df = heat_df.sort_index()
        df = df.rename(columns={VALUE_COLUMN: 'HeatConsumption'})
        heat_dt = df.dtypes['HeatConsumption']

        df['Area'] = area_df[VALUE_COLUMN]
        area_dt = area_df.dtypes[VALUE_COLUMN]
        df[FORECAST_COLUMN] |= area_df[FORECAST_COLUMN]
        last_hist_year = df.loc[~df[FORECAST_COLUMN]].index.get_level_values(YEAR_COLUMN).max()

        # Calculate heat consumption per area
        rows = ~df[FORECAST_COLUMN]

        s = (df.loc[rows, 'HeatConsumption'] / df.loc[rows, 'Area']).dropna()
        dt = s.dtype
        pa_df = s.unstack(self.output_dimension_ids)  # type: ignore
        pa_df = pa_df.rolling(window=5).mean().astype(dt)

        """
            df = df.loc[rows].dropna()
            df = df.drop(columns=FORECAST_COLUMN)
            df = df.unstack(self.output_dimension_ids)
            df = df.loc[(df.index >= last_hist_year - 12) & (df.index <= last_hist_year)]
            df = df.diff().cumsum().dropna()
            df = df.stack(self.output_dimension_ids)
            df['PerArea'] = df['HeatConsumption'].astype(heat_dt) / df['Area'].astype(area_dt)
            self.print(df['PerArea'].unstack(self.output_dimension_ids).astype('pint[kWh/a/m**2]'))
            exit()
        """

        s = pa_df.stack(self.output_dimension_ids)  # type: ignore
        s = s.astype(dt)
        s.index = s.index.reorder_levels(heat_df.index.names)  # type: ignore
        df['PerArea'] = s
        df['PerArea'] = df['PerArea'].ffill()

        rows = df[FORECAST_COLUMN]
        df.loc[rows, 'HeatConsumption'] = df.loc[rows, 'Area'] * df.loc[rows, 'PerArea']
        df = df[['HeatConsumption', FORECAST_COLUMN]].rename(columns=dict(HeatConsumption=VALUE_COLUMN))
        return df


class BuildingHeatPerAreaOld(Node):
    output_metrics = {
        HEAT_CONSUMPTION: NodeMetric(unit='kWh/a/m**2', quantity=ENERGY_QUANTITY)
    }
    output_dimension_ids = [
        'building_heat_source',
        'building_use',
    ]

    def compute(self) -> pd.DataFrame:
        # First compute GWh / m2
        area_node = self.get_input_node(quantity=FLOOR_AREA)
        area_df = area_node.get_output()
        area_df = area_df.sort_index()

        heat_node = self.get_input_node(quantity=ENERGY_QUANTITY)
        heat_df = heat_node.get_output()
        assert isinstance(heat_df.index, pd.MultiIndex)
        heat_df.index = heat_df.index.reorder_levels(area_df.index.names)
        df = heat_df.sort_index()
        df = df.rename(columns={VALUE_COLUMN: 'HeatConsumption'})
        heat_dt = df.dtypes['HeatConsumption']

        df['Area'] = area_df[VALUE_COLUMN]
        area_dt = area_df.dtypes[VALUE_COLUMN]
        df[FORECAST_COLUMN] |= area_df[FORECAST_COLUMN]
        last_hist_year = df.loc[~df[FORECAST_COLUMN]].index.get_level_values(YEAR_COLUMN).max()

        # Calculate heat consumption per area
        rows = ~df[FORECAST_COLUMN]

        s = (df.loc[rows, 'HeatConsumption'] / df.loc[rows, 'Area']).dropna()
        dt = s.dtype
        pa_df = s.unstack(self.output_dimension_ids)  # type: ignore
        pa_df = pa_df.rolling(window=5).mean().astype(dt)

        """
            df = df.loc[rows].dropna()
            df = df.drop(columns=FORECAST_COLUMN)
            df = df.unstack(self.output_dimension_ids)
            df = df.loc[(df.index >= last_hist_year - 12) & (df.index <= last_hist_year)]
            df = df.diff().cumsum().dropna()
            df = df.stack(self.output_dimension_ids)
            df['PerArea'] = df['HeatConsumption'].astype(heat_dt) / df['Area'].astype(area_dt)
            self.print(df['PerArea'].unstack(self.output_dimension_ids).astype('pint[kWh/a/m**2]'))
            exit()
        """

        s = pa_df.stack(self.output_dimension_ids)  # type: ignore
        s = s.astype(dt)
        s.index = s.index.reorder_levels(heat_df.index.names)  # type: ignore
        df['PerArea'] = s
        df['PerArea'] = df['PerArea'].ffill()
        # FIXME Oil heating (services) gets wrongly copied to all future years. PathsDataFrames should be used.
        df['PerArea'] = self.ensure_output_unit(df['PerArea'])

#        rows = df[FORECAST_COLUMN]
#        df.loc[rows, 'PerArea'] = df.loc[rows, 'PerArea']
        df = df[['PerArea', FORECAST_COLUMN]].rename(columns=dict(PerArea=VALUE_COLUMN))
        return df


class BuildingHeatPerArea(DivisiveNode):
    output_metrics = {
        HEAT_CONSUMPTION: NodeMetric(unit='kWh/a/m**2', quantity=CONSUMPTION_FACTOR_QUANTITY)
    }
    output_dimension_ids = [
        'building_heat_source',
        'building_use',
    ]

    # FIXME compute() forked from MultiplicativeNode. Maybe generalize?
    def compute(self) -> ppl.PathsDataFrame:
        additive_nodes: list[Node] = []
        operation_nodes: list[Node] = []
        assert self.unit is not None
        non_additive_nodes = self.get_input_nodes(tag='non_additive')
        if len(non_additive_nodes) == 1:
            non_additive_node = non_additive_nodes[0].id
        else:
            non_additive_node = ''
        for node in self.input_nodes:
            if node.unit is None:
                raise NodeError(self, "Input node %s does not have a unit" % str(node))
            if node.id == non_additive_node:
                operation_nodes.append(node)
            elif self.is_compatible_unit(node.unit, self.unit):
                additive_nodes.append(node)
            else:
                operation_nodes.append(node)

        if len(operation_nodes) != 2:
            raise NodeError(self, "Must receive exactly two inputs to operate %s on" % self.operation_label)

        n1, n2 = operation_nodes
        df1 = n1.get_output_pl(target_node=self)
        df2 = n2.get_output_pl(target_node=self)

        if self.debug:
            print('%s: %s input from node 1 (%s):' % (self.operation_label, self.id, n1.id))
            self.print(df1)
            print('%s: %s input from node 2 (%s):' % (self.operation_label, self.id, n2.id))
            self.print(df2)

        df = self.perform_operation([n1, n2], [df1, df2])

        df = df.with_columns(pl.col(VALUE_COLUMN).fill_nan(None)).drop_nulls()
        df = df.filter(~pl.col(FORECAST_COLUMN))
        df = extend_last_historical_value_pl(df, end_year=self.get_end_year())
        df = df.paths.to_narrow()

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

    def compute_old(self) -> pd.DataFrame:
        df = super().compute()
        df = df.with_columns(pl.col(VALUE_COLUMN).fill_nan(None)).drop_nulls()
        df = df.filter(~pl.col(FORECAST_COLUMN))  # FIXME forgets added nodes
        df = extend_last_historical_value_pl(df, end_year=self.get_end_year())
        df = df.paths.to_narrow()

        # FIXME A thing to consider: should the energy efficiency be a long-term average?
        return df
