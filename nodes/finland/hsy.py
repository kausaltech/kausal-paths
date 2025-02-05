from __future__ import annotations

import re
import typing
import warnings
from functools import wraps
from typing import ClassVar, ParamSpec, TypedDict, TypeVar

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from common import polars as ppl
from common.i18n import TranslatedString
from nodes.calc import extend_last_historical_value
from nodes.constants import (
    EMISSION_FACTOR_QUANTITY,
    EMISSION_QUANTITY,
    ENERGY_QUANTITY,
    FORECAST_COLUMN,
    PER_CAPITA_QUANTITY,
    VALUE_COLUMN,
    YEAR_COLUMN,
)
from nodes.dimensions import Dimension
from nodes.exceptions import NodeError
from nodes.node import Node, NodeMetric
from nodes.simple import AdditiveNode, MultiplicativeNode
from params import NumberParameter, Parameter, StringParameter

if typing.TYPE_CHECKING:
    from collections.abc import Callable

BELOW_ZERO_WARNED = False


P = ParamSpec('P')  # For parameters
R = TypeVar('R')    # For return type

def deprecated(alternative: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            warnings.warn(
                f"{func.__name__} is deprecated and will be removed in future versions. "
                f"Use {alternative} instead.",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


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
                # Convert to pint unit directly rather than using astype
                df[metric_id] = pd.Series(df[metric_id].values, dtype=f"pint[{metric.unit}]")

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
    output_dimensions = {
        'sector': Dimension(id='hsy_sector', label=TranslatedString(en='HSY emission sector'), is_internal=True)
    }

    def compute(self) -> pd.DataFrame:
        muni_name = self.get_global_parameter_value('municipality_name')

        df = self.get_input_dataset()
        if 'Kaupunki' in df.columns:
            df = df.loc[df['Kaupunki'] == muni_name].drop(columns=['Kaupunki'])

        if 'index' in df.columns:
            df = df.drop(columns=['index'])

        df = df.rename(columns={
            'Vuosi': YEAR_COLUMN,
            'Päästöt': EMISSION_QUANTITY,
            'Energiankulutus': ENERGY_QUANTITY,
        })
        below_zero = (df[EMISSION_QUANTITY] < 0) | (df[ENERGY_QUANTITY] < 0)
        if len(below_zero):
            global BELOW_ZERO_WARNED  # noqa: PLW0603

            if not BELOW_ZERO_WARNED:
                self.logger.warning('HSY dataset has negative emissions, filling with zero')
                BELOW_ZERO_WARNED = True
            df.loc[below_zero, [EMISSION_QUANTITY, ENERGY_QUANTITY]] = 0

        # Emission factors are calculated later because they cannot be summed
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
                df[metric_id] = pd.Series(df[metric_id].values, dtype=f"pint[{metric.unit}]")

        df[FORECAST_COLUMN] = False
        return df

    def check(self):
        return


class SectorParseResult(TypedDict):
    pattern: str
    dimensions: dict[int, str]


class HsyNodeMixin:
    allowed_parameters: ClassVar[list[Parameter]] = [
        StringParameter(
            local_id='sector',
            label='Sector path in HSY emission database',
            is_customizable=False
        ),
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
        self: Node | HsyNodeMixin,
        columns: str | list[str],
        sector: str | None = None,
        multi_index: bool = False
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
            raise NodeError(self, "HsyNode not configured as an input node")

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
            df_xs = df.groupby(group_cols).sum()

        elif multi_index:
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

    @deprecated(alternative='get_sector2')
    def get_sector(
        self, columns: str | list[str],
        sector: str | None = None,
        multi_index: bool = False) -> tuple[pd.DataFrame, list[Node]]:
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
            raise NodeError(self, "HsyNode not configured as an input node")

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


class HsyEnergyConsumption(AdditiveNode, HsyNodeMixin):
    default_unit = 'GWh/a'
    quantity = ENERGY_QUANTITY
    allowed_parameters: ClassVar[list[Parameter]] = HsyNodeMixin.allowed_parameters

    def compute(self) -> ppl.PathsDataFrame:
        try:
            df, other_nodes = self.get_sector2(
                columns=[ENERGY_QUANTITY],
                # multi_index=True,
            )
        except (NodeError, ValueError, KeyError) as e:
            logger.warning(f"Dimensional parsing failed: {e}. Falling back to legacy sector parsing.")
            df, other_nodes = self.get_sector(ENERGY_QUANTITY)

        df = df.rename(columns={ENERGY_QUANTITY: VALUE_COLUMN})
        assert VALUE_COLUMN in df

        # If there are other input nodes connected, add them with this one.
        if len(other_nodes):
            df = self.add_nodes(df, other_nodes)
        pdf = ppl.from_pandas(df)
        # df = self.convert_names_to_ids(df)
        return pdf

class HsyEmissions(AdditiveNode, HsyNodeMixin):
    default_unit = 'kt/a'
    quantity = EMISSION_QUANTITY
    allowed_parameters: ClassVar[list[Parameter]] = HsyNodeMixin.allowed_parameters

    def compute(self) -> ppl.PathsDataFrame:
        try:
            df, other_nodes = self.get_sector2(
                columns=[EMISSION_QUANTITY],
                # multi_index=True,
            )
        except (NodeError, ValueError, KeyError) as e:
            logger.warning(f"Dimensional parsing failed: {e}. Falling back to legacy sector parsing.")
            df, other_nodes = self.get_sector(EMISSION_QUANTITY)

        df = df.rename(columns={EMISSION_QUANTITY: VALUE_COLUMN})
        assert VALUE_COLUMN in df

        # If there are other input nodes connected, add them with this one.
        if len(other_nodes):
            df = self.add_nodes(df, other_nodes)
        pdf = ppl.from_pandas(df)
        # df = self.convert_names_to_ids(df)
        return pdf

class HsyEmissionFactor(AdditiveNode, HsyNodeMixin):
    default_unit = 'g/kWh'
    quantity = EMISSION_FACTOR_QUANTITY
    allowed_parameters: ClassVar[list[Parameter]] = HsyNodeMixin.allowed_parameters

    def compute(self) -> ppl.PathsDataFrame:
        try:
            df, other_nodes = self.get_sector2(
                columns=[ENERGY_QUANTITY, EMISSION_QUANTITY],
                # multi_index=True,
            )
        except (NodeError, ValueError, KeyError) as e:
            logger.warning(f"Dimensional parsing failed: {e}. Falling back to legacy sector parsing.")
            df, other_nodes = self.get_sector(ENERGY_QUANTITY, EMISSION_QUANTITY)

        df[VALUE_COLUMN] = df[EMISSION_QUANTITY] / df[ENERGY_QUANTITY].replace(0, np.nan)
        df = df.drop(columns=[ENERGY_QUANTITY, EMISSION_QUANTITY])
        assert self.unit is not None
        df[VALUE_COLUMN] = self.convert_to_unit(df[VALUE_COLUMN], self.unit)

        # If there are other input nodes connected, add them with this one.
        if len(other_nodes):
            df = self.add_nodes(df, other_nodes)
        pdf = ppl.from_pandas(df)
        # df = self.convert_names_to_ids(df)
        return pdf


class HsyBuildingHeatConsumption(Node, HsyNodeMixin):
    default_unit = 'GWh/a'
    quantity = ENERGY_QUANTITY
    allowed_parameters: ClassVar[list[Parameter]] = HsyNodeMixin.allowed_parameters

    output_dimension_ids = [
        'building_heat_source',
        'building_use',
    ]

    def compute(self) -> ppl.PathsDataFrame:
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

        # There was a change in HSY statistics logic for geothermal energy between 2018-2019.
        # Fix geothermal values before 2019, if they are heat rather than electricity.
        pdf: ppl.PathsDataFrame = ppl.from_pandas(df)
        geo = pl.col('building_heat_source')==pl.lit('geothermal')
        tst = pdf.filter(geo)
        meta = tst.get_meta()
        tst = ppl.to_ppdf(tst.group_by(pl.col(YEAR_COLUMN)).sum(), meta=meta)
        tst1 = tst.filter(pl.col(YEAR_COLUMN) == 2018)[VALUE_COLUMN][0]
        tst = tst.filter(pl.col(YEAR_COLUMN) == 2019)[VALUE_COLUMN][0]
        if tst1 > tst * 2:
            cop = 3  # Ratio of heat energy produced per electricity consumed
            pdf = pdf.with_columns(
                pl.when((geo) & (pl.col(YEAR_COLUMN) < 2019))
                .then(pl.col(VALUE_COLUMN) / cop)
                .otherwise(pl.col(VALUE_COLUMN).alias(VALUE_COLUMN))
            )
        # df = df.to_pandas()
        return pdf


class HsyDataCollection(Node, HsyNodeMixin):
    default_unit = 'GWh/a'
    quantity = ENERGY_QUANTITY

    allowed_parameters: typing.ClassVar[list[Parameter]] = HsyNodeMixin.allowed_parameters + [
        NumberParameter(
            local_id='dimension1_column',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='dimension2_column',
            is_customizable=False,
        ),
    ]

    def compute(self) -> ppl.PathsDataFrame:
        assert len(self.output_dimensions) == 2
        dim1, dim2 = list(self.output_dimensions.keys())
        column = self.quantity
        dim1_level = int(self.get_typed_parameter_value('dimension1_column', float)) - 1
        dim2_level = int(self.get_typed_parameter_value('dimension2_column', float)) - 1

        df, _ = self.get_sector(column, multi_index=True)
        df = df.reset_index().set_index(YEAR_COLUMN)
        df[dim1] = df['sector'].apply(lambda x: x.split('|')[dim1_level])
        df[dim2] = df['sector'].apply(lambda x: x.split('|')[dim2_level])

        use_dim = self.output_dimensions[dim1]
        df[dim1] = use_dim.series_to_ids(df[dim1])
        heat_dim = self.output_dimensions[dim2]
        df[dim2] = heat_dim.series_to_ids(df[dim2])

        df[VALUE_COLUMN] = self.ensure_output_unit(df[column])
        df = df.reset_index()
        df = df.set_index([YEAR_COLUMN, dim1, dim2])[[VALUE_COLUMN, FORECAST_COLUMN]]
        pdf: ppl.PathsDataFrame = ppl.from_pandas(df)
        assert isinstance(pdf, ppl.PathsDataFrame)
        if self.debug:
            self.print(pdf.get_last_historical_values())
        return pdf


class HsyPerCapitaEnergyConsumption(AdditiveNode, HsyNodeMixin):
    default_unit = 'kWh/cap/a'
    quantity = PER_CAPITA_QUANTITY
    input_datasets = ['population']
    allowed_parameters: ClassVar[list[Parameter]] = HsyNodeMixin.allowed_parameters

    def compute(self) -> ppl.PathsDataFrame:
        df, other_nodes = self.get_sector(ENERGY_QUANTITY)
        pop_df = self.get_input_dataset()
        df[VALUE_COLUMN] = df[VALUE_COLUMN].div(pop_df[VALUE_COLUMN], axis='index')
        print(df)
        exit()


class MultiplicativeWithDataBackup(MultiplicativeNode):

    def compute(self) -> ppl.PathsDataFrame:
        pdf = super().compute()
        meta = pdf.get_meta()

        data_node = self.get_input_node(tag='data_node')
        df_data = data_node.get_output_pl()
        # FIXME dimensions in df are cat but in df_data str. Which one they should be and how to fix this in a clever way?
        df_data = df_data.with_columns([pl.col('building_heat_source').cast(pl.Categorical)])
        df_data = df_data.with_columns([pl.col('building_use').cast(pl.Categorical)])
        on = list(set(pdf.get_meta().primary_keys + df_data.get_meta().primary_keys))
        df = pdf.join(df_data, on=on, how='outer', coalesce=True)

        # FIXME If you add actions to years without calculated values, you get zero-counting rather than double-counting.
        df = df.with_columns([
            pl.when(pl.col(VALUE_COLUMN) != pl.col(VALUE_COLUMN + '_right'))
            .then(True).otherwise(False).alias('DoubleCounting')  # noqa: FBT003
        ])
        df = df.with_columns([
            pl.when(pl.col('DoubleCounting'))
            .then(pl.col(VALUE_COLUMN) - pl.col(VALUE_COLUMN + '_right'))
            .otherwise(pl.col(VALUE_COLUMN)).alias(VALUE_COLUMN)
        ])
        df = df.with_columns([
            pl.when(pl.col('DoubleCounting'))
            .then(pl.col(FORECAST_COLUMN))
            .otherwise(pl.col(FORECAST_COLUMN + '_right')).alias(FORECAST_COLUMN)
        ])
        df = df.drop([FORECAST_COLUMN + '_right', VALUE_COLUMN + '_right', 'DoubleCounting'])
        df = ppl.to_ppdf(df, meta)

        return df
