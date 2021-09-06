from params.param import NumberParameter
import pandas as pd
from nodes.context import Context
from . import ActionNode as BaseActionNode
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN
from params import StringParameter
from common.i18n import TranslatedString


class ActionNode(BaseActionNode):
    unit = 'kt'
    quantity = 'emissions'
    input_datasets = ['kpr/indicator_results']

    allowed_parameters = [
        StringParameter(
            local_id='panorama_id',
            label=TranslatedString(en="Node ID in Panorama"),
            is_customizable=False
        ),
        NumberParameter(
            local_id='panorama_reduction_mton',
            label=TranslatedString(en="Panorama reduction potential"),
            is_customizable=False
        )
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

        df.loc[context.target_year, VALUE_COLUMN] = -1.0
        df[VALUE_COLUMN] = df[VALUE_COLUMN].interpolate()
        df *= self.get_parameter_value('panorama_reduction_mton') * 1000
        df[FORECAST_COLUMN] = True

        return df
