import logging
from typing import ClassVar, Tuple, Union

import numpy as np
import pandas as pd
from nodes.calc import extend_last_historical_value
from params import StringParameter, BoolParameter, Parameter
from nodes import Node
from nodes.constants import (
    VALUE_COLUMN, YEAR_COLUMN, EMISSION_FACTOR_QUANTITY, EMISSION_QUANTITY, ENERGY_QUANTITY
)
from nodes.simple import AdditiveNode
from nodes.exceptions import NodeError
from nodes.node import NodeMetric


BELOW_ZERO_WARNED = False


class HsyNode(Node):
    input_datasets = [
        'hsy/pks_khk_paastot',
    ]
    global_parameters = ['municipality_name']
    metrics = {
        EMISSION_QUANTITY: NodeMetric(unit='kt/a', quantity=EMISSION_QUANTITY),
        ENERGY_QUANTITY: NodeMetric(unit='GWh/a', quantity=ENERGY_QUANTITY),
        EMISSION_FACTOR_QUANTITY: NodeMetric(unit='g/kWh', quantity=EMISSION_FACTOR_QUANTITY)
    }

    def compute(self) -> pd.DataFrame:
        muni_name = self.get_global_parameter_value('municipality_name')

        df = self.get_input_dataset()
        todrop = ['Kaupunki']
        if 'index' in df.columns:
            todrop += ['index']
        df = df[df['Kaupunki'] == muni_name].drop(columns=todrop)
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

        df[EMISSION_FACTOR_QUANTITY] = df[EMISSION_QUANTITY] / df[ENERGY_QUANTITY].replace(0, np.nan)
        df['Sector'] = ''
        for i in range(1, 5):
            if i > 1:
                df['Sector'] += '|'
            df['Sector'] += df['Sektori%d' % i].astype(str)

        df = df[[YEAR_COLUMN, EMISSION_QUANTITY, ENERGY_QUANTITY, EMISSION_FACTOR_QUANTITY, 'Sector']]
        df = df.set_index(['Year', 'Sector'])
        if len(df) == 0:
            raise NodeError(self, "Municipality %s not found in data" % muni_name)
        for dim_id, dim in self.metrics.items():
            if hasattr(df[dim_id], 'pint'):
                df[dim_id] = self.convert_to_unit(df[dim_id], dim.unit)
            else:
                df[dim_id] = df[dim_id].astype('pint[' + str(dim.unit) + ']')
        return df


class HsyNodeMixin:
    allowed_parameters: ClassVar[list[Parameter]] = [
        StringParameter(
            local_id='sector',
            label='Sector path in ALaS',
            is_customizable=False
        ),
    ]

    def get_sector(self: Union[Node, 'HsyNodeMixin'], column: str) -> Tuple[pd.DataFrame, list[Node]]:
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
        sector = self.get_parameter_value('sector')
        try:
            df_xs = df.xs(sector, level='Sector')
            assert isinstance(df_xs, pd.DataFrame)
            df = df_xs
        except KeyError:
            raise NodeError(self, "'Sector' level not found in input")

        df = df[[column]]
        df = df.rename(columns={column: VALUE_COLUMN})
        df['Forecast'] = False
        df = extend_last_historical_value(df, self.context.target_year)
        return df, nodes


class HsyEnergyConsumption(AdditiveNode, HsyNodeMixin):
    default_unit = 'GWh/a'
    quantity = ENERGY_QUANTITY
    allowed_parameters: ClassVar[list[Parameter]] = HsyNodeMixin.allowed_parameters

    def compute(self) -> pd.DataFrame:
        df, other_nodes = self.get_sector(ENERGY_QUANTITY)
        # If there are other input nodes connected, add them with this one.
        if len(other_nodes):
            df = self.add_nodes(df, other_nodes)
        return df


class HsyEmissionFactor(AdditiveNode, HsyNodeMixin):
    default_unit = 'g/kWh'
    quantity = EMISSION_FACTOR_QUANTITY
    allowed_parameters: ClassVar[list[Parameter]] = HsyNodeMixin.allowed_parameters

    def compute(self) -> pd.DataFrame:
        df, other_nodes = self.get_sector(EMISSION_FACTOR_QUANTITY)
        if len(other_nodes):
            df = self.add_nodes(df, other_nodes)
        return df


class HsyEmissions(AdditiveNode, HsyNodeMixin):
    default_unit = 'kt/a'
    quantity = EMISSION_QUANTITY
    allowed_parameters: ClassVar[list[Parameter]] = HsyNodeMixin.allowed_parameters

    def compute(self) -> pd.DataFrame:
        df, other_nodes = self.get_sector(EMISSION_QUANTITY)
        if len(other_nodes):
            df = self.add_nodes(df, other_nodes)
        return df
