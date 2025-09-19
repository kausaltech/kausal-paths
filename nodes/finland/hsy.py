from __future__ import annotations

import re
import typing
import warnings
from functools import wraps
from typing import ClassVar, ParamSpec, TypedDict, TypeVar

import pandas as pd
import polars as pl

from common import polars as ppl
from common.i18n import TranslatedString
from nodes.calc import extend_last_historical_value
from nodes.constants import (
    EMISSION_QUANTITY,
    ENERGY_QUANTITY,
    FORECAST_COLUMN,
    VALUE_COLUMN,
    YEAR_COLUMN,
)
from nodes.dimensions import Dimension
from nodes.exceptions import NodeError
from nodes.node import Node, NodeMetric
from params import Parameter, StringParameter

if typing.TYPE_CHECKING:
    from collections.abc import Callable

BELOW_ZERO_WARNED = False


P = ParamSpec('P')  # For parameters
R = TypeVar('R')  # For return type


def deprecated(alternative: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            warnings.warn(
                f'{func.__name__} is deprecated and will be removed in future versions. Use {alternative} instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


class AluesarjatNode(Node):
    input_datasets = ['helsinki/aluesarjat/02um_rakennukset_lammitys']
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
                # Convert to pint unit directly rather than using astype
                df[metric_id] = pd.Series(df[metric_id].values, dtype=f'pint[{metric.unit}]')

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
    global_parameters = ['municipality_name']
    output_metrics = {
        EMISSION_QUANTITY: NodeMetric(unit='kt/a', quantity=EMISSION_QUANTITY),
        ENERGY_QUANTITY: NodeMetric(unit='GWh/a', quantity=ENERGY_QUANTITY),
    }
    output_dimensions = {'sector': Dimension(id='hsy_sector', label=TranslatedString(en='HSY emission sector'), is_internal=True)}

    # TODO Put this in a better place. Is this needed by nodes in general?
    def _linear_interpolate(self, df: ppl.PathsDataFrame, year_col: str = YEAR_COLUMN) -> ppl.PathsDataFrame:
        years = df[year_col].unique().sort()
        min_year = years.min()
        assert isinstance(min_year, int)
        max_year = years.max()
        assert isinstance(max_year, int)
        df = df.paths.to_wide()
        years_df = pl.DataFrame(data=range(min_year, max_year + 1), schema=[year_col])
        meta = df.get_meta()
        zdf = years_df.join(df, on=year_col, how='left').sort(year_col)
        df = ppl.to_ppdf(zdf, meta=meta)
        cols = [pl.col(col).interpolate() for col in df.metric_cols]
        if FORECAST_COLUMN in df.columns:
            cols.append(pl.col(FORECAST_COLUMN).fill_null(strategy='forward'))
        df = df.with_columns(cols)
        df = df.paths.to_narrow()
        return df

    def compute(self) -> ppl.PathsDataFrame:
        muni_name = self.get_global_parameter_value('municipality_name')
        assert isinstance(muni_name, str)

        df = self.get_input_dataset_pl()
        if 'Kaupunki' in df.columns:
            df = df.filter(pl.col('Kaupunki') == muni_name).drop('Kaupunki')

        if 'index' in df.columns:
            df = df.drop('index')

        df = df.rename(
            {
                'Vuosi': YEAR_COLUMN,
                'Päästöt': EMISSION_QUANTITY,
                'Energiankulutus': ENERGY_QUANTITY,
            }
        )
        # Handle negative values
        has_negative = df.filter((pl.col(EMISSION_QUANTITY) < 0) | (pl.col(ENERGY_QUANTITY) < 0)).height > 0
        if has_negative:
            global BELOW_ZERO_WARNED  # noqa: PLW0603

            if not BELOW_ZERO_WARNED:
                self.logger.warning('HSY dataset has negative emissions, filling with zero')
                BELOW_ZERO_WARNED = True

            df = df.with_columns(
                [
                    pl.when(pl.col(EMISSION_QUANTITY) < 0).then(0).otherwise(pl.col(EMISSION_QUANTITY)).alias(EMISSION_QUANTITY),
                    pl.when(pl.col(ENERGY_QUANTITY) < 0).then(0).otherwise(pl.col(ENERGY_QUANTITY)).alias(ENERGY_QUANTITY),
                ]
            )

        # Create Sector column by concatenating Sektori1-4
        sector_cols = [f'Sektori{i}' for i in range(1, 6)]
        df = df.with_columns(
            [
                pl.concat_str([pl.col(col).cast(pl.Utf8) for col in sector_cols if col in df.columns], separator='|').alias(
                    'Sector'
                )
            ]
        )

        df = df.select([YEAR_COLUMN, EMISSION_QUANTITY, ENERGY_QUANTITY, 'Sector'])
        df = df.rename({'Sector': 'sector'})
        df = df.add_to_index('sector')

        if df.height == 0:
            raise NodeError(self, 'Municipality %s not found in data' % muni_name)

        df = df.with_columns([pl.lit(False).alias(FORECAST_COLUMN)])  # noqa: FBT003
        df = self._linear_interpolate(df)

        return df

    def check(self):
        return


class SectorParseResult(TypedDict):
    pattern: str
    dimensions: dict[int, str]


class HsyNodeMixin:
    allowed_parameters: ClassVar[list[Parameter]] = [
        StringParameter(local_id='sector', label='Sector path in HSY emission database', is_customizable=False),
    ]

    def parse_dimension_names_from_sector_string(self, sector_name: str) -> SectorParseResult:
        sector_levels = sector_name.split('|')
        dimension_map = {}  # Maps level index to dimension name
        filter_pattern = []  # Build regex pattern for filtering

        for i, level in enumerate(sector_levels):
            if level.startswith('_') and level.endswith('_'):
                # This is a dimension level
                dim_name = level.strip('_')
                dimension_map[i] = dim_name
                filter_pattern.append(r'[^|]+')  # Match any non-pipe characters
            elif level == '*':
                filter_pattern.append(r'[^|]+')
            else:
                # This is a fixed level
                filter_pattern.append(re.escape(level))

        # Create regex pattern for filtering
        full_pattern = r'\|'.join(filter_pattern)
        return {'pattern': full_pattern, 'dimensions': dimension_map}

    def get_sector2(
        self: Node | HsyNodeMixin, columns: str | list[str], sector: str | None = None, multi_index: bool = False
    ) -> tuple[pd.DataFrame, list[Node]]:
        """
        Get sector data with dimensional support.

        This method extends the original get_sector functionality with support
        for dimensional data parsing. Use this for new code.

        Args:
            columns: Columns to include in output
            sector: Sector string, can include dimension markers like:
                   'Liikenne|Tieliikenne|_vehicle_type_|_sector_type_'
                   where _dimension_name_ indicates which levels become dimensions.
                   In the string, * means that the level is ignored.
            multi_index: Whether to keep sector in index when no dimensions specified

        """
        assert isinstance(self, Node)
        assert isinstance(self, HsyNodeMixin)
        nodes = list(self.input_nodes)
        for node in nodes:
            if isinstance(node, HsyNode):
                break
        else:
            raise NodeError(self, 'HsyNode not configured as an input node')

        nodes.remove(node)
        df = node.get_output()

        sector_name: str
        if sector is None:
            sector_name = self.get_parameter_value_str('sector')
        else:
            sector_name = sector

        parsed_sectors = self.parse_dimension_names_from_sector_string(sector_name)
        full_pattern = parsed_sectors['pattern']
        dimension_map = parsed_sectors['dimensions']
        matching_sectors = df.index.get_level_values('sector').str.match(full_pattern)

        if not matching_sectors.any():
            raise NodeError(self, f"Sector pattern '{full_pattern}' not found in input")

        df = df.loc[matching_sectors]

        if dimension_map:
            # Split sectors into hierarchy levels
            df['sector_levels'] = df.index.get_level_values('sector').str.split('|')

            # Create new columns for each dimension with proper category IDs
            for level_idx, dim_name in dimension_map.items():
                dim = self.input_dimensions[dim_name]
                # Extract the level values as a series and convert to category IDs
                level_series = pl.Series(df['sector_levels'].str[level_idx])
                df[dim_name] = dim.series_to_ids_pl(level_series)

            # Group by dimensions and year
            group_cols = ['Year'] + list(dimension_map.values())
            df_xs = df.groupby(group_cols, observed=True).sum()

        elif multi_index:
            df_xs = df.groupby(['Year', 'sector'], observed=True).sum()
        else:
            df_xs = df.groupby('Year', observed=True).sum()

        assert isinstance(df_xs, pd.DataFrame)
        df = df_xs

        if isinstance(columns, str):
            columns = [columns]
        df = df[columns].copy()
        df['Forecast'] = False
        df = extend_last_historical_value(df, end_year=self.context.model_end_year)

        return df, nodes

    @deprecated(alternative='get_sector2')
    def get_sector(
        self, columns: str | list[str], sector: str | None = None, multi_index: bool = False
    ) -> tuple[pd.DataFrame, list[Node]]:
        """
        Get sector data using simple string matching (deprecated).

        This method is deprecated and will be removed in future versions.
        Use get_sector2() instead, which supports dimensional data.

        Args:
            columns: Columns to include in output
            sector: Sector string to match
            multi_index: Whether to keep sector in index

        """

        assert isinstance(self, Node)
        nodes = list(self.input_nodes)
        for node in nodes:
            if isinstance(node, HsyNode):
                break
        else:
            raise NodeError(self, 'HsyNode not configured as an input node')

        # Remove the HsyNode from the list of nodes to be added together
        nodes.remove(node)
        df = node.get_output()

        sector_name: str
        if sector is None:
            sector_name = self.get_parameter_value_str('sector')
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
