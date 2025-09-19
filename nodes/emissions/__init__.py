from __future__ import annotations

from django.utils.translation import gettext_lazy as _

import pandas as pd
import polars as pl

from common import polars as ppl
from nodes.calc import AR5GWP100
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.node import Node


class GlobalWarmingPotential(Node):
    explanation = _("""Calculate CO2 equivalents based on 100-year global warming potential.""")
    default_unit = 'dimensionless'

    def compute(self) -> ppl.PathsDataFrame:
        start_year = self.context.instance.reference_year or 1990
        end_year = self.get_end_year()
        years = range(start_year, end_year + 1)
        ghg_dim = next(iter(self.output_dimensions.values()))
        idx = pd.MultiIndex.from_product([years, ghg_dim.get_cat_ids()]).to_list()
        df = pl.DataFrame(idx, orient='row', schema=[YEAR_COLUMN, ghg_dim.id])
        df = df.with_columns(
            [pl.col(ghg_dim.id).replace(AR5GWP100).cast(pl.Float32).alias(VALUE_COLUMN), pl.lit(False).alias(FORECAST_COLUMN)]
        )
        meta = ppl.DataFrameMeta(
            units={VALUE_COLUMN: self.context.unit_registry.parse_units('dimensionless')}, primary_keys=[YEAR_COLUMN, ghg_dim.id]
        )
        pdf = ppl.to_ppdf(df, meta=meta)
        return pdf
