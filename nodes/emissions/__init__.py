import pandas as pd
import polars as pl

from nodes.node import Node
from common import polars as ppl
from nodes.calc import AR5GWP100
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN


class GlobalWarmingPotential(Node):
    default_unit = 'dimensionless'

    def compute(self) -> ppl.PathsDataFrame:
        start_year = self.context.instance.reference_year or 1990
        end_year = self.get_end_year()
        years = range(start_year, end_year + 1)
        ghg_dim = list(self.output_dimensions.values())[0]
        idx = pd.MultiIndex.from_product([years, ghg_dim.get_cat_ids()]).to_list()
        df = pl.DataFrame(idx, orient='row', schema=[YEAR_COLUMN, ghg_dim.id])
        df = df.with_columns([
            pl.col(ghg_dim.id).map_dict(AR5GWP100).alias(VALUE_COLUMN),
            pl.lit(False).alias(FORECAST_COLUMN)
        ])
        meta = ppl.DataFrameMeta(
            units={VALUE_COLUMN: self.context.unit_registry.parse_units('dimensionless')},
            primary_keys=[YEAR_COLUMN, ghg_dim.id]
        )
        pdf = ppl.to_ppdf(df, meta=meta)
        return pdf
