import logging
from typing import ClassVar, Tuple, Union

import numpy as np
import pandas as pd
from nodes.calc import extend_last_historical_value
from nodes.dimensions import Dimension
from params import StringParameter, Parameter
from nodes import Node, NodeMetric
from nodes.constants import (
    FORECAST_COLUMN, PER_CAPITA_QUANTITY, VALUE_COLUMN, YEAR_COLUMN, FORECAST_COLUMN, 
    EMISSION_FACTOR_QUANTITY, EMISSION_QUANTITY, ENERGY_QUANTITY
)
from nodes.simple import AdditiveNode
from nodes.exceptions import NodeError


BELOW_ZERO_WARNED = False


class AluesarjatNode(Node):
    input_datasets = [
        'helsinki/aluesarjat/02um_rakennukset_lammitys'
    ]
    global_parameters = ['municipality_name']
    output_metrics = {
        VALUE_COLUMN: NodeMetric(unit='m**2', quantity='area'),
    }

    def compute(self) -> pd.DataFrame:
        df = self.get_input_dataset()
        muni_name = self.get_global_parameter_value('municipality_name')

        df = self.get_input_dataset()
        self.print_pint_df(df)
        todrop = ['Alue', 'Tiedot']
        if 'index' in df.columns:
            todrop += ['index']
        df = df[df['Alue'] == muni_name]
        df = df[df['Tiedot'] == 'Kerrosala (m2)'].drop(columns=todrop)
        df = df.rename(columns={'Vuosi': YEAR_COLUMN, 'value': VALUE_COLUMN})
        for metric_id, metric in self.output_metrics.items():
            if hasattr(df[metric_id], 'pint'):
                df[metric_id] = self.convert_to_unit(df[metric_id], metric.unit)
            else:
                df[metric_id] = df[metric_id].astype('pint[' + str(metric.unit) + ']')

        dimensions = ['Rakennuksen käyttötarkoitus', 'Rakennuksen lämmitystapa', 'Rakennuksen lämmitysaine']
        df['Dimension'] = ''
        for i in range(3):
            if i > 0:
                df['Dimension'] += '|'
            df['Dimension'] += df[dimensions[i]].astype(str)
        df[FORECAST_COLUMN] = False
        keeps = list(set(df.columns) - set(dimensions))
        df = df[keeps]
        df = df.set_index([YEAR_COLUMN, 'Dimension'])
        return df


class HsyNode(Node):
    input_datasets = [
        'hsy/pks_khk_paastot',
    ]
    global_parameters = ['municipality_name']
    output_metrics = {
        EMISSION_QUANTITY: NodeMetric(unit='kt/a', quantity=EMISSION_QUANTITY),
        ENERGY_QUANTITY: NodeMetric(unit='GWh/a', quantity=ENERGY_QUANTITY),
    }
    output_dimensions = {
        'sector': Dimension(id='hsy_sector', label=dict(en='HSY emission sector'), is_internal=True)
    }

    def compute(self) -> pd.DataFrame:
        muni_name = self.get_global_parameter_value('municipality_name')

        df = self.get_input_dataset()
        todrop = ['Kaupunki']
        if 'index' in df.columns:
            todrop += ['index']
        df = df.loc[df['Kaupunki'] == muni_name].drop(columns=todrop)
        df = df.rename(columns={
            'Vuosi': YEAR_COLUMN,
            'Päästöt': EMISSION_QUANTITY,
            'Energiankulutus': ENERGY_QUANTITY,
        })
        below_zero = (df[EMISSION_QUANTITY] < 0) | (df[ENERGY_QUANTITY] < 0)
        if len(below_zero):
            global BELOW_ZERO_WARNED

            if not BELOW_ZERO_WARNED:
                self.logger.warn('HSY dataset has negative emissions, filling with zero')
                BELOW_ZERO_WARNED = True
            df.loc[below_zero, [EMISSION_QUANTITY, ENERGY_QUANTITY]] = 0

        # df[EMISSION_FACTOR_QUANTITY] = df[EMISSION_QUANTITY] / df[ENERGY_QUANTITY].replace(0, np.nan)
        df['Sector'] = ''
        for i in range(1, 5):
            if i > 1:
                df['Sector'] += '|'
            df['Sector'] += df['Sektori%d' % i].astype(str)

        df = df[[YEAR_COLUMN, EMISSION_QUANTITY, ENERGY_QUANTITY, 'Sector']]
        df = df.rename(columns=dict(Sector='sector'))
        df = df.set_index([YEAR_COLUMN, 'sector'])
        if len(df) == 0:
            raise NodeError(self, "Municipality %s not found in data" % muni_name)
        for metric_id, metric in self.output_metrics.items():
            if hasattr(df[metric_id], 'pint'):
                df[metric_id] = self.convert_to_unit(df[metric_id], metric.unit)
            else:
                df[metric_id] = df[metric_id].astype('pint[' + str(metric.unit) + ']')
        return df


class HsyNodeMixin:
    allowed_parameters: ClassVar[list[Parameter]] = [
        StringParameter(
            local_id='sector',
            label='Sector path in HSY emission database',
            is_customizable=False
        ),
    ]

    def get_sector(self: Union[Node, 'HsyNodeMixin'], columns: str | list[str], sector: str | None = None, multi_index: bool = False) -> Tuple[pd.DataFrame, list[Node]]:
        assert isinstance(self, Node)
        nodes = list(self.input_nodes)
        for node in nodes:
            if isinstance(node, HsyNode):
                break
        else:
            raise NodeError(self, "HsyNode not configured as an input node")

        # Remove the HsyNode from the list of nodes to be added together
        nodes.remove(node)
        df = node.get_output()

        sector_name: str
        if sector is None:
            sector_name = self.get_parameter_value('sector')
        else:
            sector_name = sector

        matching_sectors = df.index.get_level_values('sector').str.startswith(sector_name)
        if not matching_sectors.any():
            raise NodeError(self, "Sector level '%s' not found in input" % sector_name)

        df = df.loc[matching_sectors]
        if multi_index:
            df_xs = df.groupby(['Year', 'sector']).sum()
        else:
            df_xs = df.groupby('Year').sum()
        assert isinstance(df_xs, pd.DataFrame)
        df = df_xs

        if isinstance(columns, str):
            columns = [columns]
        df = df[columns].copy()
        df['Forecast'] = False
        df = extend_last_historical_value(df, end_year=self.context.model_end_year)

        return df, nodes


class HsyEnergyConsumption(AdditiveNode, HsyNodeMixin):
    default_unit = 'GWh/a'
    quantity = ENERGY_QUANTITY
    allowed_parameters: ClassVar[list[Parameter]] = HsyNodeMixin.allowed_parameters

    def compute(self) -> pd.DataFrame:
        df, other_nodes = self.get_sector(ENERGY_QUANTITY)
        df = df.rename(columns={ENERGY_QUANTITY: VALUE_COLUMN})
        assert VALUE_COLUMN in df

        # If there are other input nodes connected, add them with this one.
        if len(other_nodes):
            df = self.add_nodes(df, other_nodes)
        return df


class HsyEmissions(AdditiveNode, HsyNodeMixin):
    default_unit = 'kt/a'
    quantity = EMISSION_QUANTITY
    allowed_parameters: ClassVar[list[Parameter]] = HsyNodeMixin.allowed_parameters

    def compute(self) -> pd.DataFrame:
        df, other_nodes = self.get_sector(EMISSION_QUANTITY)
        df = df.rename(columns={EMISSION_QUANTITY: VALUE_COLUMN})
        assert VALUE_COLUMN in df
        if len(other_nodes):
            df = self.add_nodes(df, other_nodes)
        return df


class HsyEmissionFactor(AdditiveNode, HsyNodeMixin):
    default_unit = 'g/kWh'
    quantity = EMISSION_FACTOR_QUANTITY
    allowed_parameters: ClassVar[list[Parameter]] = HsyNodeMixin.allowed_parameters

    def compute(self) -> pd.DataFrame:
        df, other_nodes = self.get_sector([ENERGY_QUANTITY, EMISSION_QUANTITY])
        df[VALUE_COLUMN] = df[EMISSION_QUANTITY] / df[ENERGY_QUANTITY].replace(0, np.nan)
        df = df.drop(columns=[ENERGY_QUANTITY, EMISSION_QUANTITY])

        # If there are other input nodes connected, add them with this one.
        if len(other_nodes):
            df = self.add_nodes(df, other_nodes)
        return df


class HsyBuildingHeatConsumption(Node, HsyNodeMixin):
    default_unit = 'GWh/a'
    quantity = ENERGY_QUANTITY

    output_dimension_ids = [
        'building_heat_source',
        'building_use',
    ]

    def compute(self) -> pd.DataFrame:
        df, _ = self.get_sector(ENERGY_QUANTITY, sector='Lämmitys', multi_index=True)
        df = df.reset_index().set_index(YEAR_COLUMN)
        df['building_use'] = df['sector'].apply(lambda x: x.split('|')[-1])
        df['building_heat_source'] = df['sector'].apply(lambda x: x.split('|')[-2])

        use_dim = self.output_dimensions['building_use']
        df['building_use'] = use_dim.series_to_ids(df['building_use'])
        heat_dim = self.output_dimensions['building_heat_source']
        df['building_heat_source'] = heat_dim.series_to_ids(df['building_heat_source'])

        df[VALUE_COLUMN] = self.ensure_output_unit(df[ENERGY_QUANTITY])
        df = df.reset_index()
        df = df.set_index([YEAR_COLUMN, 'building_use', 'building_heat_source'])[[VALUE_COLUMN, FORECAST_COLUMN]]
        return df


class HsyPerCapitaEnergyConsumption(AdditiveNode, HsyNodeMixin):
    default_unit = 'kWh/cap/a'
    quantity = PER_CAPITA_QUANTITY
    input_datasets = ['population']

    def compute(self) -> pd.DataFrame:
        df, other_nodes = self.get_sector(ENERGY_QUANTITY)
        pop_df = self.get_input_dataset()
        df[VALUE_COLUMN] = df[VALUE_COLUMN].div(pop_df[VALUE_COLUMN], axis='index')
        print(df)
        exit()
