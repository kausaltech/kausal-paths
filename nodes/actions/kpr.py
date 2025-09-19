from __future__ import annotations

import pandas as pd

from common.i18n import TranslatedString
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN
from nodes.context import Context
from params import StringParameter
from params.param import NumberParameter

from . import ActionNode as BaseActionNode


class ActionNode(BaseActionNode):
    unit = 'kt'
    quantity = 'emissions'
    input_datasets = ['kpr/indicator_results']

    allowed_parameters = [
        StringParameter(local_id='panorama_id', label=TranslatedString(en='Node ID in Panorama'), is_customizable=False),
        NumberParameter(
            local_id='panorama_reduction_mton', label=TranslatedString(en='Panorama reduction potential'), is_customizable=False
        ),
    ]

    def compute_effect(self, context: Context) -> pd.DataFrame:
        df = self.get_input_dataset(context)
        sec_id = self.get_parameter_value('panorama_id')
        df = df.set_index(['NyckelID', 'År'])
        if sec_id not in df.index:
            print('WARNING: Node %s not found in KPR input data (%s)' % (self.id, sec_id))
            val = pd.Series([0, 1], index=[2020, 2045])
        else:
            val = df.loc[sec_id]['Panorama potential (kton utsläppsminskning)']

        df = pd.DataFrame(val.values, index=val.index.astype(int), columns=[VALUE_COLUMN])

        # Scale to match the Panorama reduction numbers
        first_val = df.loc[2020][VALUE_COLUMN]
        df[VALUE_COLUMN] -= first_val
        last_val = df.iloc[-1][VALUE_COLUMN]
        if last_val > 0:
            df[VALUE_COLUMN] /= -last_val
            df.loc[df.index < 2020, VALUE_COLUMN] = 0
        else:
            # Divide by zero or negative, just do a linear interpolation
            df.loc[df.index <= 2020, VALUE_COLUMN] = 0
            df.loc[df.index >= 2020, VALUE_COLUMN] = None
            df.loc[df.index.max(), VALUE_COLUMN] = -1.0

        df.loc[context.model_end_year, VALUE_COLUMN] = -1.0
        df[VALUE_COLUMN] = df[VALUE_COLUMN].interpolate()
        if not self.is_enabled():
            df.loc[df.index >= 2020, VALUE_COLUMN] = 0
        df *= self.get_parameter_value('panorama_reduction_mton') * 1000
        df[FORECAST_COLUMN] = True
        df.loc[df.index < 2020, FORECAST_COLUMN] = False

        return df
