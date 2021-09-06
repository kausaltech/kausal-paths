from nodes.context import Context
import pandas as pd
from .simple import SectorEmissions as BaseSectorEmissions
from .constants import FORECAST_COLUMN, VALUE_COLUMN
from params import StringParameter
from common.i18n import TranslatedString


class SectorEmissions(BaseSectorEmissions):
    unit = 'Mt'
    input_datasets = [
        'kpr/emission_statistics',
    ]
    allowed_parameters = [
        StringParameter(
            local_id='panorama_id',
            label=TranslatedString(en="Node ID in Panorama"),
            is_customizable=False
        ),
    ]

    def compute(self, context: Context):
        # Hack to avoid double counting
        if self.output_nodes[0].id == 'net_emissions':
            return self.add_nodes(context, None, self.input_nodes)

        df = self.get_input_dataset(context).set_index('Utsläpp ID')

        sec_id = self.get_parameter_value('panorama_id')
        edf = df.loc[sec_id]
        df = pd.DataFrame(
            edf.values, index=edf.index.astype(int), columns=[VALUE_COLUMN]
        )
        df[FORECAST_COLUMN] = False
        last_year = df.index.max()
        target_year = context.target_year
        df = df.reindex(range(df.index.min(), target_year + 1))
        df.loc[target_year, VALUE_COLUMN] = df.loc[last_year, VALUE_COLUMN]
        df[VALUE_COLUMN] = df[VALUE_COLUMN].interpolate()
        df.loc[df.index > last_year, FORECAST_COLUMN] = True

        return self.add_nodes(context, df, self.input_nodes)
