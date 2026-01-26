from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas as pd
import pint_pandas

import common.polars as ppl
from nodes.calc import extend_last_historical_value
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.node import NodeMetric
from nodes.simple import AdditiveNode, SimpleNode

if TYPE_CHECKING:
    from collections.abc import Sequence

INDUSTRY_USES = """
Teollisuuden ja kaivannaistoiminnan rakennukset
Varastorakennukset
Muut rakennukset
""".strip().splitlines()

RESIDENTIAL_USES = """
Omakoti- ja paritalot
Rivitalot
Kerrostalot
""".strip().splitlines()

SERVICE_USES = """
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
""".strip().splitlines()

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
    # 'Yhteensä': 'Muu',
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

    def compute(self) -> ppl.PathsDataFrame:
        df = self.get_input_dataset()
        muni = self.get_global_parameter_value('municipality_name')
        if 'Alue' in df.columns:
            df = df.loc[df.Alue == muni]
        df = df.loc[
            (df.Tiedot == 'Kerrosala (m2)') &
            (df['Rakennuksen käyttötarkoitus'] != 'Asuinrakennukset yhteensä') &
            (df['Rakennuksen käyttötarkoitus'] != 'Rakennukset yhteensä') &
            (df['Rakennuksen lämmitysaine'] != 'Yhteensä')
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
        s = df.groupby(['building_heat_source', 'building_use', YEAR_COLUMN])['Value'].sum()
        s = m.ensure_output_unit(self, s)
        df = pd.DataFrame(data=s.values, index=s.index, columns=[VALUE_COLUMN])
        df[VALUE_COLUMN] = df[VALUE_COLUMN].astype(s.dtype)
        df[FORECAST_COLUMN] = False

        df = extend_last_historical_value(df, self.context.model_end_year)
        df = self.add_nodes(df, self.input_nodes)
        dfpl = ppl.from_pandas(df)
        return dfpl


class FutureBuildingStock(SimpleNode):
    """
    Calculate the new floor area of future buildings.

    This is based on the 10-year average
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
        df = df.unstack(self.output_dimension_ids)  # type: ignore  # noqa: PD010
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
        pop_diff = pop_df.loc[pop_df.index >= last_hist_year, VALUE_COLUMN].diff().dropna() # type: ignore
        df = area_per_new_cap_df
        future_index = pd.RangeIndex(last_hist_year + 1, self.context.model_end_year + 1, name=YEAR_COLUMN)
        df = df.reindex(df.index.append(future_index))
        df = df.ffill()
        df = df.mul(pop_diff, axis=0).dropna().cumsum()
        df = df.astype(area_dt)

        df = cast('pd.DataFrame', df.stack(cast('Sequence[str]', self.output_dimension_ids), future_stack=True))  # noqa: PD013
        assert isinstance(df, pd.DataFrame)
        df = df.rename(columns={'Area': VALUE_COLUMN})
        df[FORECAST_COLUMN] = True

        nodes = self.input_nodes.copy()
        nodes.remove(hist_node)
        nodes.remove(pop_node)
        assert isinstance(df, pd.DataFrame)
        # self.add_nodes(df, nodes)  FIXME

        return df


