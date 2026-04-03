from __future__ import annotations

from typing import cast

from django.utils.translation import gettext_lazy as _

import numpy as np
import pandas as pd

from kausal_common.i18n.pydantic import TranslatedString

from nodes.calc import extend_last_historical_value
from nodes.constants import (
    EMISSION_FACTOR_QUANTITY,
    EMISSION_QUANTITY,
    ENERGY_QUANTITY,
    FORECAST_COLUMN,
    VALUE_COLUMN,
    YEAR_COLUMN,
)
from nodes.dimensions import Dimension
from nodes.exceptions import NodeError
from nodes.node import Node, NodeMetric
from params import BoolParameter, StringParameter


class AlasNode(Node):
    input_datasets = [
        'syke/alas_emissions',
    ]
    global_parameters = ['municipality_name', 'selected_framework']
    allowed_parameters = [
        StringParameter(local_id='region', label=_('Region to be included'), is_customizable=False),
    ]
    output_metrics = {
        EMISSION_QUANTITY: NodeMetric(unit='kt/a', quantity=EMISSION_QUANTITY),
        ENERGY_QUANTITY: NodeMetric(unit='GWh/a', quantity=ENERGY_QUANTITY),
        EMISSION_FACTOR_QUANTITY: NodeMetric(unit='g/kWh', quantity=EMISSION_FACTOR_QUANTITY),
    }
    output_dimensions = {
        'Sector': Dimension(id='syke_sector', label=TranslatedString(en='SYKE emission sector'), is_internal=True),
    }

    def compute(self) -> pd.DataFrame:  # noqa: C901, PLR0912
        df = self.get_input_dataset()
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        muni_name = self.get_global_parameter_value('municipality_name')
        region_name = self.get_parameter_value('region', required=False)
        if region_name:
            if region_name == 'Suomi':
                cats = ['taso_1', 'taso_2', 'taso_3', 'taso_4', 'taso_5', 'hinku-laskenta', 'päästökauppa', 'vuosi']
                df = df.groupby(cats, observed=True).sum(numeric_only=True).reset_index()
            else:
                raise Exception(self, 'Unknown region')
        else:
            df = df[df['kunta'] == muni_name].drop(columns=['kunta'])

        emission_field = 'ktCO2e'
        fw = self.get_global_parameter_value('selected_framework', required=False)
        if fw:
            frameworks = [
                'Hinku-laskenta ilman päästöhyvityksiä',
                'Hinku-laskenta päästöhyvityksillä',
                'Kaikki ALas-päästöt',
                'Taakanjakosektorin kaikki ALas-päästöt',
                'Päästökaupan alaiset ALas-päästöt',
            ]

            if fw in frameworks[0:2]:
                print('vain hinku')
                df = df[df['hinku-laskenta']]
                if fw == frameworks[0]:
                    print('ei kompensaatiota')
                    df = df[df['taso_1'] != 'Kompensaatiot']
                else:
                    emission_field += '_tuuli'

            elif fw == frameworks[3]:
                print('ei päästökauppaa')
                df = df[~df['päästökauppa']]
            elif fw == frameworks[4]:
                print('vain päästökauppa')
                df = df[df['päästökauppa']]

        df = df.rename(
            columns={
                'vuosi': YEAR_COLUMN,
                emission_field: EMISSION_QUANTITY,
                'energiankulutus': ENERGY_QUANTITY,
            }
        )
        df[EMISSION_FACTOR_QUANTITY] = df[EMISSION_QUANTITY] / df[ENERGY_QUANTITY].replace(0, np.nan)

        df['Sector'] = ''
        for i in range(1, 6):
            if i > 1:
                df['Sector'] += '|'
            df['Sector'] += df['taso_%d' % i].astype(str)
        df.loc[df['hinku-laskenta'], 'Sector'] += '+HINKU'
        df.loc[df['päästökauppa'], 'Sector'] += '+ETS'

        df = df[[YEAR_COLUMN, EMISSION_QUANTITY, ENERGY_QUANTITY, EMISSION_FACTOR_QUANTITY, 'Sector']]
        df = df.set_index([YEAR_COLUMN, 'Sector']).sort_index()
        if len(df) == 0:
            raise NodeError(self, 'Municipality %s not found in data' % muni_name)
        for metric_id, metric in self.output_metrics.items():
            if hasattr(df[metric_id], 'pint'):
                df[metric_id] = self.convert_to_unit(df[metric_id], metric.unit)
            else:
                df[metric_id] = df[metric_id].astype('pint[' + str(metric.unit) + ']')  # type: ignore[call-overload]

        df[FORECAST_COLUMN] = False

        return df

    def check(self):
        return


class AlasEmissions(Node):
    default_unit = 'kt/a'
    quantity = EMISSION_QUANTITY
    allowed_input_classes = [
        AlasNode,
    ]
    global_parameters = ['extend_historical_values']
    allowed_parameters = [
        StringParameter(
            local_id='sector',
            label='Sector path in ALaS',
            is_customizable=False,
        ),
        BoolParameter(
            local_id='required',
            label='Has to exist in data',
            is_customizable=False,
        ),
    ]

    def compute(self) -> pd.DataFrame:
        df = self.input_nodes[0].get_output()
        sector = self.get_parameter_value('sector')
        required = self.get_parameter_value('required', required=False)
        try:
            df = cast('pd.DataFrame', df.xs(sector, level='Sector'))
        except KeyError:
            if not required:
                years = df.index.get_level_values(YEAR_COLUMN).unique()
                dt = df.dtypes[EMISSION_QUANTITY]
                df = pd.DataFrame([0.0] * len(years), index=years, columns=[EMISSION_QUANTITY])
                df[EMISSION_QUANTITY] = df[EMISSION_QUANTITY].astype(dt)
            else:
                raise
        df = df[[EMISSION_QUANTITY]]
        if df[EMISSION_QUANTITY].isna().all():
            df = df.fillna(0.0)
        df = df.rename(columns={EMISSION_QUANTITY: VALUE_COLUMN})
        df[FORECAST_COLUMN] = False

        if self.get_global_parameter_value('extend_historical_values', required=False):
            df.index = df.index.astype(dtype='int64')  # TODO Could update this function to polars
            df = extend_last_historical_value(df, self.get_end_year())
        return df
