from __future__ import annotations

import pandas as pd

from kausal_common.i18n.pydantic import TranslatedString

from params import StringParameter

from .constants import FORECAST_COLUMN, VALUE_COLUMN
from .simple import SectorEmissions as BaseSectorEmissions


class SectorEmissions(BaseSectorEmissions):
    default_unit = 'kt'
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

    def compute(self):
        context = self.context
        if self.output_nodes[0].id == 'net_emissions':
            return self.add_nodes(None, self.input_nodes)

        df = self.get_input_dataset().set_index('Utsläpp ID')

        sec_id = self.get_parameter_value_str('panorama_id')
        edf = df.loc[sec_id]
        df = pd.DataFrame(
            edf.values, index=edf.index.astype(int), columns=[VALUE_COLUMN]
        )
        df[FORECAST_COLUMN] = False
        last_year = df.index.max()
        model_end_year = context.model_end_year
        df = df.reindex(range(df.index.min(), model_end_year + 1))
        df.loc[model_end_year, VALUE_COLUMN] = df.loc[last_year, VALUE_COLUMN]
        df[VALUE_COLUMN] = df[VALUE_COLUMN].interpolate()
        df.loc[df.index > last_year, FORECAST_COLUMN] = True

        return self.add_nodes(df, self.input_nodes)
