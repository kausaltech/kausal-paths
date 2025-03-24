from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, TypedDict

from django.utils.translation import gettext_lazy as _

import polars as pl

from common import polars as ppl
from nodes.actions import ActionNode
from nodes.calc import extend_last_historical_value_pl
from nodes.units import unit_registry
from params.param import StringParameter

from .constants import EMISSION_FACTOR_QUANTITY, EMISSION_QUANTITY, ENERGY_QUANTITY, FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from .exceptions import NodeError
from .simple import SimpleNode

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from .node import Node


class GenericNode(SimpleNode):
    explanation = _(
        """
        GenericNode: A highly configurable node that processes inputs through a sequence of operations.

        Operations are defined in the 'operations' parameter and executed in order.
        Each operation works on its corresponding basket of nodes.
        """
    )
    allowed_parameters = [
        *SimpleNode.allowed_parameters,
        StringParameter(local_id='operations', label='Comma-separated list of operations to execute in order')
    ]
    # Class-level default operations
    DEFAULT_OPERATIONS = 'multiply,add,other,apply_multiplier'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Instance-level operations (can be overridden by subclasses)
        self.default_operations = self.DEFAULT_OPERATIONS

        # Operation registry
        self.OPERATIONS: dict[str, Callable[..., tuple[Any, Any]]]  = {
            'multiply': self._operation_multiply,
            'add': self._operation_add,
            'other': self._operation_other,
            'apply_multiplier': self._operation_apply_multiplier,
        }

    def _get_input_baskets(self, nodes: list[Node]) -> dict[str, list[Node]]:
        """Return a dictionary of node 'baskets' categorized by type."""
        baskets: dict[str, list[Node]] = {
            'additive': [],
            'multiplicative': [],
            'other': []
        }
        tag_to_basket = {
            'additive': 'additive',
            'non_additive': 'multiplicative',
            'other_node': 'other',
            'rate': 'other',
            'base': 'other',
        }
        # Special tags that should be skipped completely
        skip_tags = {'ignore_content'}

        # Categorize nodes by tags
        for edge in self.edges:
            if edge.output_node != self or edge.input_node not in nodes:
                continue

            node = edge.input_node
            if any(tag in edge.tags or tag in node.tags for tag in skip_tags):
                continue

            assigned = False
            for tag, basket in tag_to_basket.items():
                if tag in edge.tags or tag in node.tags:
                    baskets[basket].append(node)
                    assigned = True
                    break

            if not assigned:
                if self.is_compatible_unit(self.unit, node.unit):
                    baskets['additive'].append(node)
                else:
                    baskets['multiplicative'].append(node)

        return baskets

    # Operation wrapper functions
    def _operation_multiply(self, df: ppl.PathsDataFrame | None, baskets: dict, **kwargs) -> tuple:
        """Multiply all nodes in the multiplicative basket."""
        nodes = baskets['multiplicative']
        if nodes and len(nodes) > 0:
            df = self.multiply_nodes_pl(
                df=df,
                nodes=nodes,
                metric=kwargs.get('metric'),
                unit=kwargs.get('unit'),
                start_from_year=kwargs.get('start_from_year')
            )
            baskets['multiplicative'] = []
        return df, baskets

    def _operation_add(self, df: ppl.PathsDataFrame | None, baskets: dict, **kwargs) -> tuple:
        """Add all nodes in the additive basket."""
        nodes = baskets['additive']
        if nodes and len(nodes) > 0:
            df = self.add_nodes_pl(
                df=df,
                nodes=nodes,
                metric=kwargs.get('metric'),
                keep_nodes=kwargs.get('keep_nodes', False),
                node_multipliers=kwargs.get('node_multipliers'),
                unit=kwargs.get('unit'),
                start_from_year=kwargs.get('start_from_year'),
                ignore_unit=True
            )
            baskets['additive'] = []
        return df, baskets

    def _operation_other(self, df: ppl.PathsDataFrame | None, baskets: dict, **kwargs) -> tuple:
        """Process other nodes - to be implemented by subclasses."""
        if type(self) is GenericNode and len(baskets['other']) > 0:
            raise NodeError(self, f"Generic node {self.id} cannot handle 'other' nodes.")
        return df, baskets

    def _operation_apply_multiplier(self, df: ppl.PathsDataFrame | None, baskets: dict, **kwargs) -> tuple:
        """Apply the node's multiplier parameter to the dataframe."""
        if df is None:
            raise NodeError(self, "Cannot apply multiplier because no PathsDataFrame is available.")
        mult = self.get_parameter_value('multiplier', required=False, units=True)
        if mult:
            df = df.multiply_quantity(VALUE_COLUMN, mult)
        return df, baskets

    def compute(self) -> ppl.PathsDataFrame:
        """Process inputs according to the operations sequence."""
        # Get operation sequence from parameter or class default
        operations_str = self.get_parameter_value_str('operations', required=False) or self.default_operations
        operations = [op.strip() for op in operations_str.split(',')]

        # Get input dataset and categorize nodes
        df = self.get_input_dataset_pl(tag='baseline', required=False)
        baskets = self._get_input_baskets(self.input_nodes)

        # Track original node counts for validation
        original_node_counts = {basket: len(nodes) for basket, nodes in baskets.items()}

        kwargs = {
            'metric': None,
            'unit': self.unit,
            'start_from_year': None,
            'keep_nodes': False,
            'node_multipliers': None
        }

        # Apply operations in sequence
        for op_name in operations:
            if op_name not in self.OPERATIONS:
                raise NodeError(self, f"Unknown operation: {op_name}")

            operation_func = self.OPERATIONS[op_name]
            df, baskets = operation_func(df, baskets, **kwargs)
        if not isinstance(df, ppl.PathsDataFrame):
            raise NodeError(self, "The output is not a PathsDataFrame.")

        # Validate that all nodes were used
        unused_nodes = [
            (node.id, basket)
            for basket, nodes in baskets.items()
            if nodes and original_node_counts.get(basket, 0) > 0
            for node in nodes]

        if unused_nodes:
            unused_str = ", ".join([f"{node_id} ({basket})" for node_id, basket in unused_nodes])
            raise NodeError(self, f"Unused input nodes found after all operations: {unused_str}")

        df = df.ensure_unit(VALUE_COLUMN, self.unit)

        return df


class LeverNode(GenericNode):
    explanation = _(
        """LeverNode replaces the upstream computation completely, if the lever is enabled."""
    )
    allowed_parameters = [
        *GenericNode.allowed_parameters,
        StringParameter(local_id='new_category'),
    ]

    def _operation_override_with_lever(self, df: ppl.PathsDataFrame | None, baskets: dict, **kwargs) -> tuple:
        """Override upstream computation with lever values if enabled."""
        if df is None:
            return None, baskets

        # Get the lever node from the "other" basket
        lever_nodes = baskets.get('other', [])
        baskets['other'] = []
        if len(lever_nodes) != 1:
            raise NodeError(self, "LeverNode requires exactly one 'other_node' tagged input as lever.")

        lever = lever_nodes[0]

        if not isinstance(lever, ActionNode):
            raise NodeError(self, f"Lever {lever} must be an action.")

        # If lever is not enabled, return original dataframe
        if not lever.is_enabled():
            return df, baskets

        # Get lever output and override values
        dfl = lever.get_output_pl(target_node=self)
        dfl = dfl.ensure_unit(VALUE_COLUMN, df.get_unit(VALUE_COLUMN))
        out = df.paths.join_over_index(dfl, how='left', index_from='left')
        out = out.with_columns(
            (pl.when(pl.col(FORECAST_COLUMN))
            .then(pl.col(VALUE_COLUMN + '_right'))
            .otherwise(pl.col(VALUE_COLUMN))).alias(VALUE_COLUMN)
        )
        out = out.drop(VALUE_COLUMN + '_right')

        if len(df) != len(out):
            s = f"({len(out)} rows) as the affected node {self.id} ({len(df)} rows)."
            raise NodeError(self, f"Lever {lever.id} must result in the same structure {s}")

        return out, baskets

    def _operation_fill_new_category(self, df: ppl.PathsDataFrame, baskets: dict, **kwargs) -> tuple:
        """Fill in a new category with complement values."""
        if df is None:
            return None, baskets

        category = self.get_parameter_value_str('new_category', required=True)
        dim, cat = category.split(':')

        df = df.ensure_unit(VALUE_COLUMN, 'dimensionless')

        df2 = df.paths.sum_over_dims(dim)
        df2 = df2.with_columns((pl.lit(1.0) - pl.col(VALUE_COLUMN)).alias(VALUE_COLUMN))
        df2 = df2.with_columns(pl.lit(cat).cast(pl.Categorical).alias(dim))
        df2 = df2.select(df.columns)

        df = df.paths.concat_vertical(df2)
        df = df.ensure_unit(VALUE_COLUMN, self.unit)

        if self.get_parameter_value('drop_nans', required=False):  # FIXME Not consistent with the parameter name!
            df = df.paths.to_wide()
            for col in df.metric_cols:
                df = df.filter(~pl.col(col).is_null())
            df = df.paths.to_narrow()

        return df, baskets

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register custom operations
        self.OPERATIONS['override_with_lever'] = self._operation_override_with_lever
        self.OPERATIONS['fill_new_category'] = self._operation_fill_new_category
        # Set default operations sequence
        self.default_operations = 'multiply,add,other,override_with_lever,fill_new_category,apply_multiplier'


class WeightedSumNode(GenericNode):
    explanation = _(
        """
        WeightedSumNode: Combines additive inputs using weights from a multidimensional weights DataFrame.
        """
    )
    # Class-level default overrides parent
    DEFAULT_OPERATIONS = 'multiply,add_with_weights,add,other,apply_multiplier'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register the custom operation
        self.OPERATIONS['add_with_weights'] = self._operation_add_with_weights

    def _process_single_weighted_node(self, node: Node, node_weights: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        """Process a single node with its weights and return the weighted output."""
        # Get node output
        node_output = node.get_output_pl(target_node=self)

        # Find the metric column that has non-null values for this node
        valid_metric = None
        metrics = node_weights.metric_cols.copy()
        for col in metrics:
            if col in node_weights.columns and node_weights[col].null_count() < len(node_weights):
                valid_metric = col
                metrics.remove(col)
                break

        node_weights = node_weights.drop(metrics)
        if not valid_metric:
            raise NodeError(self, f"No valid metric column found for weight dataset for node {node.id}")

        # Create a version with this metric renamed to VALUE_COLUMN
        if valid_metric != VALUE_COLUMN:
            node_weights = node_weights.rename({valid_metric: VALUE_COLUMN})

        # Multiply node output with weights
        return node_output.paths.multiply_with_dims(node_weights, how='inner')

    def _combine_weighted_outputs(self, weights_df: ppl.PathsDataFrame, additive_nodes: list[Node]) -> ppl.PathsDataFrame | None:
        """Process and combine all weighted node outputs."""
        # Create a lookup map for additive nodes
        node_map = {node.id: node for node in additive_nodes}
        result = None

        # Process each unique node in the weights dataframe
        for node_id in weights_df['node'].unique():
            if node_id not in node_map:
                self.logger.warning(f"Node {node_id} not found in additive nodes")
                continue

            # Get node and its weights
            node = node_map[node_id]
            node_weights = weights_df.filter(pl.col('node') == node_id).drop('node')

            # Process the node
            weighted_output = self._process_single_weighted_node(node, node_weights)

            # Add to result (first time initializes, subsequent adds)
            if result is None:
                result = weighted_output
            else:
                result = result.paths.add_with_dims(weighted_output, how='outer')

        return result

    def _operation_add_with_weights(self, df: ppl.PathsDataFrame, baskets: dict, **kwargs) -> tuple:
        """Combine additive node outputs weighted by values in a multidimensional weights DataFrame."""
        # Get weights dataframe from specifically tagged input
        weights_df = self.get_input_dataset_pl(tag='input_node_weights', required=False)
        if weights_df is None:
            return df, baskets

        # Focus specifically on nodes in the additive basket
        additive_nodes = baskets['additive']
        if not additive_nodes:
            raise NodeError(
                self,
                "If node contains weights, it must contain additive input nodes (typically with tag 'additive')."
            )

        # Process all weighted nodes
        result = self._combine_weighted_outputs(weights_df, additive_nodes)

        # FIXME Must get a result.
        # FIXME Also, must use df for non-weighted addition.
        if result is None:
            self.logger.warning(f"No matching nodes found in weights DataFrame for {self.id}")
            return df, baskets

        # Remove processed nodes from additive basket (to prevent double-counting)
        processed_nodes = [node for node in additive_nodes if node.id in weights_df['node'].unique()]
        for node in processed_nodes:
            baskets['additive'].remove(node)

        return result, baskets


class LogitNode(WeightedSumNode):
    explanation = _(
        """
        LogitNode gives a probability of event given a baseline and several determinants.

        The baseline is given as a dataset of observed values. The determinants are linearly
        related to the logit of the probability:
        ln(y / (1 - y)) = a + sum_i(b_i * X_i,)
        where y is the probability, a is baseline, X_i determinants and b_i coefficients.
        The node expects that a comes from dataset and sum_i(b_i * X_i,) is given by the input nodes
        when operated with the GenericNode compute(). The probability is calculated as
        ln(y / (1 - y)) = b <=> y = 1 / (1 + exp(-b)).
        """
    )

    def _operation_logit_transform(self, df: ppl.PathsDataFrame, baskets: dict, **kwargs) -> tuple:
        """Apply logit transform to combine observations with weighted sum."""
        if df is None:
            return None, baskets

        # Ensure our weighted sum is in dimensionless units
        df = df.ensure_unit(VALUE_COLUMN, 'dimensionless')

        # Get observations dataset
        df_obs = self.get_input_dataset_pl(tag='observations', required=False)
        if df_obs is None:
            raise NodeError(self, f"LogitNode {self.id} must have one dataset for baseline values.")

        # Validate and transform observations to logit space
        df_obs = df_obs.ensure_unit(VALUE_COLUMN, 'dimensionless')
        test = df_obs.with_columns(
            (pl.lit(0.0) < pl.col(VALUE_COLUMN)) & (pl.col(VALUE_COLUMN) < pl.lit(1.0))
        )['literal'].all()

        if not test:
            raise NodeError(self, f"All values in {self.id} must be between 0 and 1, exclusive.")

        df_obs = df_obs.with_columns(
            (pl.col(VALUE_COLUMN) / (pl.lit(1.0) - pl.col(VALUE_COLUMN))).log().alias(VALUE_COLUMN)
        )

        # Join observations with weighted sum
        df = df.paths.join_over_index(df_obs, how='outer', index_from='left')
        df = df.sum_cols([VALUE_COLUMN, VALUE_COLUMN + '_right'], VALUE_COLUMN).drop(VALUE_COLUMN + '_right')

        # Apply inverse logit function to get probabilities
        expr = pl.lit(1.0) / (pl.lit(1.0) + (pl.lit(-1.0) * pl.col(VALUE_COLUMN)).exp())
        df = df.with_columns(expr.alias(VALUE_COLUMN))
        df = df.ensure_unit(VALUE_COLUMN, self.unit)

        return df, baskets

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register the logit transform operation
        self.OPERATIONS['logit_transform'] = self._operation_logit_transform
        # Set default operations sequence
        self.default_operations = 'multiply,add_with_weights,add,logit_transform,other,apply_multiplier'


class SectorParseResult(TypedDict):
    pattern: str
    dimensions: dict[int, str]


class DimensionalSectorNode(GenericNode):
    default_unit = 'kt/a'
    quantity = EMISSION_QUANTITY
    allowed_parameters = [
        *GenericNode.allowed_parameters,
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

    def process_sector_data_pl(
        self,
        df: ppl.PathsDataFrame,
        columns: str | list[str]
    ) -> ppl.PathsDataFrame:
        """
        Process sector data from a polars DataFrame.

        Args:
            df: Input PathsDataFrame with sector data
            columns: Column names to include

        Returns:
            Processed PathsDataFrame with dimensional support

        """
        if isinstance(columns, str):
            columns = [columns]

        # Get sector pattern
        sector_name: str = self.get_parameter_value_str('sector')

        # Parse dimensions from sector pattern
        parsed_sectors = self.parse_dimension_names_from_sector_string(sector_name)
        full_pattern = parsed_sectors['pattern']
        dimension_map = parsed_sectors['dimensions']

        # Filter by sector pattern
        matching_sectors = df.filter(pl.col('sector').str.contains(full_pattern))
        meta = matching_sectors.get_meta()

        if len(matching_sectors) == 0:
            raise NodeError(self, f"Sector pattern '{full_pattern}' not found in input")

        # Handle dimensions if specified
        if dimension_map:
            # Create dimension columns
            for level_idx, dim_name in dimension_map.items():
                # Split sector and extract the level
                matching_sectors = matching_sectors.with_columns(
                    pl.col('sector').str.split('|').list.get(level_idx).alias(dim_name)
                )

                # Convert to proper dimension IDs if needed
                if dim_name in self.input_dimensions:
                    dim = self.input_dimensions[dim_name]
                    matching_sectors = matching_sectors.with_columns(
                        dim.series_to_ids_pl(matching_sectors[dim_name]).alias(dim_name)
                    )

            # Group by dimensions and year
            group_cols = [YEAR_COLUMN] + list(dimension_map.values())
            result = matching_sectors.group_by(group_cols).agg([
                pl.sum(col).alias(col) for col in columns
            ])

            # Add dimension columns to index
            result = ppl.to_ppdf(result, meta)
            for dim_name in dimension_map.values():
                result = result.add_to_index(dim_name)
        else:
            # No dimensions, just group by year
            result = matching_sectors.group_by(YEAR_COLUMN).agg([
                pl.sum(col).alias(col) for col in columns
            ])
            result = ppl.to_ppdf(result, meta)

        # Add forecast column if not present
        if FORECAST_COLUMN not in result.columns:
            result = result.with_columns(pl.lit(False).alias(FORECAST_COLUMN))  # noqa: FBT003

        return result

    def _operation_process_sector(self, df: ppl.PathsDataFrame | None, baskets: dict, **kwargs) -> tuple:
        """Process the sector data from HSY nodes."""
        if df is not None:
            raise NodeError(self, "process_sector must be the first operation, so df must be None.") # TODO Could be relaxed
        if len(baskets['other']) != 1:
            raise NodeError(self, "The node must have exactly one 'other_node' input.")

        # Get the output from HSY node
        n = baskets['other'][0]
        data_df = n.get_output_pl()

        # Process the sector data (default to emissions column)
        data_column = getattr(self, 'data_column', EMISSION_QUANTITY)
        result = self.process_sector_data_pl(data_df, columns=[data_column])

        result = result.rename({data_column: VALUE_COLUMN})
        result = extend_last_historical_value_pl(result, end_year=self.context.model_end_year)
        result = result.ensure_unit(VALUE_COLUMN, self.unit)
        baskets['other'] = []

        return result, baskets

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register the sector processing operation
        self.OPERATIONS['process_sector'] = self._operation_process_sector
        # Set default operations sequence
        self.default_operations = 'process_sector,multiply,add,apply_multiplier'


class DimensionalSectorEmissions(DimensionalSectorNode):
    default_unit = 'kt/a'
    quantity = EMISSION_QUANTITY

    def __init__(self, *args, **kwargs):
        # Set the data column before initializing
        self.data_column = EMISSION_QUANTITY
        super().__init__(*args, **kwargs)


class DimensionalSectorEnergy(DimensionalSectorNode):
    default_unit = 'GWh/a'
    quantity = ENERGY_QUANTITY

    def __init__(self, *args, **kwargs):
        # Set the data column before initializing
        self.data_column = ENERGY_QUANTITY
        super().__init__(*args, **kwargs)


class DimensionalSectorEmissionFactor(DimensionalSectorNode):
    default_unit = 'g/kWh'
    quantity = EMISSION_FACTOR_QUANTITY

    def _operation_process_emission_factor(self, df: ppl.PathsDataFrame | None, baskets: dict, **kwargs) -> tuple:
        """Calculate emission factors from energy and emission data."""
        if df is not None:
            raise NodeError(self, "process_sector must be the first operation, so df must be None.") # TODO Could be relaxed
        if len(baskets['other']) != 1:
            raise NodeError(self, "The node must have exactly one 'other_node' input.")

        # Get the output from HSY node
        n = baskets['other'][0]
        data_df = n.get_output_pl()

        # Process the sector data with both energy and emissions columns
        result = self.process_sector_data_pl(data_df, columns=[ENERGY_QUANTITY, EMISSION_QUANTITY])

        # Calculate emission factor: emissions / energy
        result = result.divide_cols([EMISSION_QUANTITY, ENERGY_QUANTITY], VALUE_COLUMN)
        result = result.drop([ENERGY_QUANTITY, EMISSION_QUANTITY])
        result = extend_last_historical_value_pl(result, end_year=self.context.model_end_year)
        result = result.ensure_unit(VALUE_COLUMN, self.unit)
        baskets['other'] = []

        return result, baskets

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override the operations to use the emission factor calculation
        self.OPERATIONS['process_sector'] = self._operation_process_emission_factor


class IterativeNode(GenericNode):
    explanation = _(
        """
        This is generic IterativeNode for calculating values year by year.
        It calculates one year at a time based on previous year's value and inputs and outputs
        starting from the first forecast year. In addition, it must have a feedback loop (otherwise it makes
        no sense to use this node class), which is given as a growth rate per year from the previous year's value.
        """)

    def _get_other_node(self, tag: str, baskets: dict) -> Node:
        """Get and validate a required node from 'other' basket."""
        other_nodes = baskets.get('other', [])

        node = self.get_input_node(tag=tag, required=True)

        if node not in other_nodes:
            raise NodeError(self, f"The node with tag '{tag}' must be in 'other' basket")

        baskets['other'].remove(node)

        return node

    def _operation_year_iteration(self, df: ppl.PathsDataFrame | None, baskets: dict, **kwargs) -> tuple:
        """
        Perform year-by-year iteration using previous values, growth rate and changes.

        This operation expects df to contain the summed changes from all additive inputs.
        """
        rate_node = self._get_other_node(tag='rate', baskets=baskets)
        base_node = self._get_other_node(tag='base', baskets=baskets)

        rate_df = rate_node.get_output_pl(target_node=self)
        rate_df = rate_df.ensure_unit(VALUE_COLUMN, '1/a')
        rate_df = rate_df.set_unit(VALUE_COLUMN, 'dimensionless', force=True)
        rate_df = rate_df.with_columns(pl.col(VALUE_COLUMN) + pl.lit(1.0).alias(VALUE_COLUMN))

        base_df = base_node.get_output_pl(target_node=self)

        if df is None:
            df_out = base_df.with_columns(pl.lit(0.0).alias('changes'))
        else:
            # The changes are in df (output from previous operations)
            # Rename it to "changes" for clarity
            df_out = base_df.paths.join_over_index(df, how='left', index_from='left')
            df_out = df_out.rename({VALUE_COLUMN + '_right': 'changes'})
            df_out = df_out.set_unit('changes', df_out.get_unit('changes') * unit_registry('a'), force=True)
            df_out = df_out.ensure_unit('changes', df_out.get_unit(VALUE_COLUMN))

        df_out = df_out.with_columns(pl.col('changes').fill_null(0.0))
        df_out = df_out.paths.join_over_index(rate_df, how='left', index_from='union')
        df_out = df_out.rename({VALUE_COLUMN + '_right': 'rate'})
        df_out = df_out.paths.to_wide()

        # Perform the year-by-year iteration
        historical_df = df_out.filter(~pl.col(FORECAST_COLUMN))
        if len(historical_df) == 0:
            raise NodeError(self, "IterativeNode must have historical values.")

        # Use explicit type conversion with assertions to help mypy
        last_historical_year: int = int(float(historical_df[YEAR_COLUMN].max()))
        end_year_value: int = int(self.get_end_year())

        # Use typed variables for calculations to avoid union type issues
        for forecast_year in range(last_historical_year + 1, end_year_value + 1):
            # Convert year to the same type as in df_out's YEAR_COLUMN for filtering
            year_filter = pl.col(YEAR_COLUMN) == forecast_year
            prev_year_filter = pl.col(YEAR_COLUMN) == (forecast_year - 1)

            # Extract values as floats to ensure compatibility
            prev_year_row = df_out.filter(prev_year_filter)
            current_year_row = df_out.filter(year_filter)

            if len(prev_year_row) == 0 or len(current_year_row) == 0:
                raise NodeError(self, "IterativeNode cannot have forecast years missing in between.")

            prev_value: float = float(prev_year_row[VALUE_COLUMN].item())
            change_value: float = float(current_year_row['changes'].item())
            rate_value: float = float(current_year_row['rate'].item())

            # Calculate new value with explicit float operations
            new_value: float = (prev_value + change_value) * rate_value

            # Update the dataframe
            df_out = df_out.with_columns(
                pl.when(year_filter)
                .then(pl.lit(new_value))
                .otherwise(pl.col(VALUE_COLUMN))
                .alias(VALUE_COLUMN)
            )

        # Clean up
        df_out = df_out.drop(['rate', 'changes'])

        return df_out, baskets


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register operations
        self.OPERATIONS['year_iteration'] = self._operation_year_iteration
        # Set default operations sequence to let standard add happen before year_iteration
        self.default_operations = 'multiply,add,year_iteration,other,apply_multiplier'
