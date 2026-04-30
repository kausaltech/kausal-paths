from __future__ import annotations

import functools
import re
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypedDict, overload

from django.utils.translation import gettext_lazy as _

import numpy as np
import polars as pl

from kausal_common.debugging.helpers import hide_from_traceback
from kausal_common.perf.perf_context import PerfKind, estimate_size_bytes

from common import polars as ppl
from common.polars import PathsDataFrame
from nodes.actions.action import ActionNode
from nodes.calc import extend_last_historical_value_pl
from nodes.node import NodeMetric
from nodes.units import Quantity, Unit, unit_registry
from params.param import BoolParameter, NumberParameter, StringParameter

from .calc import compute_scenario_impact
from .constants import (
    EMISSION_FACTOR_QUANTITY,
    EMISSION_QUANTITY,
    ENERGY_QUANTITY,
    FORECAST_COLUMN,
    IMPACT_COLUMN,
    IMPACT_GROUP,
    STACKABLE_QUANTITIES,
    VALUE_COLUMN,
    YEAR_COLUMN,
)
from .exceptions import NodeError
from .explanations import TAG_TO_BASKET
from .simple import SimpleNode

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from kausal_common.perf.perf_context import PerfAttrs, PerfSpanEntry

    from nodes.context import Context
    from nodes.node import Node
    from params import Parameter


# Operations return only the (possibly updated) dataframe; inputs are resolved by tag via get_input_nodes()
OperationReturn = PathsDataFrame | None


class GenericNode(SimpleNode):
    """
    GenericNode: A highly configurable node that processes inputs through a sequence of operations.

    Operations are defined in the 'operations' parameter and executed in order.
    Each operation works on its corresponding basket of nodes.
    """

    explanation = _('Multiply input nodes whose unit does not match the output. Add the rest.')

    allowed_parameters = [
        *SimpleNode.allowed_parameters,
        StringParameter(local_id='operations', label=_('Comma-separated list of operations to execute in order')),
        StringParameter(local_id='categories', label=_('Dimension and categories to select')),
        NumberParameter(local_id='selected_number', label=_('Number of the selected category')),
        BoolParameter(local_id='do_correction', label=_('Correct values with a correction factor?')),
        NumberParameter(local_id='no_correction_value', label=_('Value to use for no correction')),
    ]
    # Class-level default operations
    DEFAULT_OPERATIONS = 'get_single_dataset,multiply,add,other,apply_multiplier'  # FIXME

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Instance-level operations (can be overridden by subclasses)
        self.default_operations = self.DEFAULT_OPERATIONS

        # Operation registry (alphabetical). Each op takes (df) and returns updated df or None.
        self.OPERATIONS: dict[str, Callable[..., OperationReturn]] = {
            'add': self._operation_add,
            'add_datasets': self._operation_add_datasets,
            'add_from_incoming_dims': self._operation_add_from_incoming_dims,
            'add_to_existing_dims': self._operation_add_to_existing_dims,
            'apply_multiplier': self._operation_apply_multiplier,
            'concat_datasets': self._operation_concat_datasets,
            'do_correction': self._operation_do_correction,
            'get_single_dataset': self._operation_get_single_dataset,
            'goal_gap': self._operation_goal_gap,
            'multiply': self._operation_multiply,
            'other': self._operation_other,
            'select_variant': self._operation_select_variant,
            'skip_dim_test': self._operation_skip_dim_test,
            'split_by_existing_shares': self._operation_split_by_existing_shares,
            'split_evenly_to_cats': self._operation_split_evenly_to_cats,
            'use_as_totals': self._operation_use_as_totals,
            'use_as_shares': self._operation_use_as_shares,
        }

    # With ports, this operation is just part of _operation_add.
    def _operation_add_datasets(self, df: PathsDataFrame | None) -> OperationReturn:
        dfs = self.get_input_datasets_pl()
        if not dfs:
            return df
        out = df if df is not None else dfs.pop()
        out = out.paths._add_missing_years(out, self.context)
        for d in dfs:
            out = out.select(d.columns)
            di = d.paths._add_missing_years(d, self.context)
            out = out.paths.add_with_dims(di)
        assert isinstance(out, PathsDataFrame)
        return out

    def _operation_get_single_dataset(self, df: PathsDataFrame | None) -> OperationReturn:
        dfc = self.get_cleaned_dataset(required=False)
        if dfc is None:
            return df
        if df is None:
            return dfc
        return df.paths.add_with_dims(dfc)

    def _operation_goal_gap(self, df: PathsDataFrame | None) -> OperationReturn:
        """Compute gap = actual - goal from the single input node's output and goals."""
        if df is not None:
            raise NodeError(
                self,
                "goal_gap must be the first and only operation; it uses the input node's output, not an incoming df.",
            )
        if len(self.input_nodes) != 1:
            raise NodeError(
                self,
                'goal_gap requires exactly one input node; got %d.' % len(self.input_nodes),
            )
        source = self.input_nodes[0]
        df = source.get_output_pl(target_node=self)
        output_dims = set(self.output_dimensions.keys())
        sum_dims = set(df.dim_ids) - output_dims
        if sum_dims:
            df = df.paths.sum_over_dims(list(sum_dims))
        if not source.goals or not source.goals.root:
            self.logger.warning(f'No goals found for node {source.id}. Are you sure you want to use goal_gap?')
            return df.with_columns(pl.lit(0.0).alias(VALUE_COLUMN))
        goal_df = source.goals.root[0]._get_values_df()

        if set(goal_df.dim_ids) != output_dims:
            raise NodeError(
                self,
                'Goal dimensions must match input node output_dimensions: got %s, expected %s.'
                % (sorted(goal_df.dim_ids), sorted(output_dims)),
            )

        out = df.paths.subtract_with_dims(goal_df, how='inner')
        return out

    def _get_add_multiply_nodes(self) -> tuple[list[Node], list[Node]]:
        """
        Return (add_nodes, multiply_nodes) for add/multiply ops.

        Uses tag 'additive'/'non_additive' or unit compatibility. Excludes nodes in
        self._weighted_node_ids (set by add_with_weights so they are not added again).
        """
        skip_tags = {'ignore_content'}
        exclude_ids: set[str] = getattr(self, '_weighted_node_ids', set())
        add_nodes: list[Node] = []
        multiply_nodes: list[Node] = []
        for edge in self.edges:
            if edge.output_node != self:
                continue
            node = edge.input_node
            if any(tag in edge.tags or tag in node.tags for tag in skip_tags):
                continue
            edge_or_node_tags = set(edge.tags) | set(node.tags)
            if node.id in exclude_ids:
                continue
            if 'additive' in edge_or_node_tags:
                add_nodes.append(node)
                continue
            if 'non_additive' in edge_or_node_tags:
                multiply_nodes.append(node)
                continue
            # No add/multiply tag: check if node has another op-specific tag (then skip for add/multiply)
            if edge_or_node_tags.intersection(TAG_TO_BASKET.keys()):
                continue
            # Untagged: assign by unit compatibility
            out_df = node.get_output_pl(target_node=self)
            df_unit = out_df.get_unit(VALUE_COLUMN)
            if self.is_compatible_unit(self.unit, df_unit):
                add_nodes.append(node)
            else:
                multiply_nodes.append(node)
        return add_nodes, multiply_nodes

    def _operation_multiply(self, df: PathsDataFrame | None) -> OperationReturn:
        """Multiply all nodes tagged 'non_additive' or (untagged and unit-incompatible)."""
        _add, multiply_nodes = self._get_add_multiply_nodes()
        if multiply_nodes:
            df = self.multiply_nodes_pl(df=df, nodes=multiply_nodes)
        return df

    def _operation_add(self, df: PathsDataFrame | None) -> OperationReturn:
        """Add all nodes tagged 'additive' or (untagged and unit-compatible)."""
        add_nodes, _multiply = self._get_add_multiply_nodes()
        if add_nodes:
            df = self.add_nodes_pl(df=df, nodes=add_nodes, ignore_unit=True)
        return df

    def _operation_other(self, df: PathsDataFrame | None) -> OperationReturn:
        """Process other nodes - to be implemented by subclasses."""
        other_nodes = self.get_input_nodes(tag='other_node')
        if type(self) is GenericNode and other_nodes:
            raise NodeError(self, f"Generic node {self.id} cannot handle 'other' nodes.")
        return df

    def _operation_apply_multiplier(self, df: PathsDataFrame | None) -> OperationReturn:
        """Apply the node's multiplier parameter to the dataframe."""
        if df is None:
            raise NodeError(self, 'Cannot apply multiplier because no PathsDataFrame is available.')
        mult = self.get_parameter_value('multiplier', required=False, units=True)
        if mult is not None:
            df = df.multiply_quantity(VALUE_COLUMN, mult)
        return df

    def _operation_skip_dim_test(self, df: PathsDataFrame | None) -> OperationReturn:
        """Skip dimension test on loading; exactly one input must have tag 'skip_dim_test'."""
        if df is not None:
            raise NodeError(self, "'skip_dim_test' must be the first of the operations.")
        return self.get_input_node(tag='skip_dim_test', required=True).get_output_pl(target_node=self, skip_dim_test=True)

    def _operation_concat_datasets(self, df: PathsDataFrame | None) -> OperationReturn:
        dfs = self.get_input_datasets_pl()
        if not dfs:
            return df
        out = df if df is not None else dfs.pop()
        for d in dfs:
            out = out.select(d.columns)
            # Earlier datasets fill gaps only — drop rows whose index already exists in out # TODO Check logic!
            primary_keys = out.primary_keys
            d_meta = d.get_meta()
            dd = ppl.to_ppdf(d.join(out.select(primary_keys), on=primary_keys, how='anti'), meta=d_meta)
            if not dd.is_empty():
                out = out.paths.concat_vertical(dd)
        assert isinstance(out, PathsDataFrame)
        out = out.paths._add_missing_years(out, self.context)
        return out

    OperationType = Literal[
        'use_as_totals',
        'use_as_shares',
        'split_by_existing_shares',
        'split_evenly_to_cats',
        'add_to_existing_dims',
        'add_from_incoming_dims',
    ]

    def _preprocess_for_one(self, df: PathsDataFrame | None, operation: OperationType, stackable: bool = True) -> PathsDataFrame:
        if df is None:
            raise NodeError(self, 'Cannot operate because no PathsDataFrame is available.')
        if self.quantity not in STACKABLE_QUANTITIES and stackable:
            raise NodeError(self, f'Split operations are only allowed for stackable quantities, not {self.quantity}.')
        n = self.get_input_node(tag=operation, required=True)
        return n.get_output_pl(target_node=self, skip_dim_test=True)

    def _operation_split_dims(self, df: PathsDataFrame, operation: OperationType) -> OperationReturn:
        """
        Split operations with different strategies.

        Input nodes are resolved by tag (e.g. use_as_totals, add_to_existing_dims).
        """
        add_non_stackables = ['add_to_existing_dims', 'add_from_incoming_dims']
        if operation in add_non_stackables:
            return self._operation_add_non_stackable_dims(df, operation)
        stackable = True
        dfin = self._preprocess_for_one(df, operation, stackable=stackable)
        use_as = ['use_as_totals', 'use_as_shares']

        if operation in ['use_as_shares', 'add_from_incoming_dims']:
            splitter = dfin
            splittee = df
        else:
            splitter = df
            splittee = dfin

        assert isinstance(splitter, PathsDataFrame)
        assert isinstance(splittee, PathsDataFrame)

        newdims = [dim for dim in splittee.dim_ids if dim not in splitter.dim_ids]
        if newdims and operation not in use_as:
            raise NodeError(self, f'Splittee node cannot bring in new dimensions but has {newdims}.')

        dims = [dim for dim in splitter.dim_ids if dim not in splittee.dim_ids]
        if not dims and not newdims:
            raise NodeError(self, "No dimensions to split. Remove the split operation if you don't use it.")

        df_summed = splitter.paths.sum_over_dims(dims)

        if operation == 'split_evenly_to_cats':
            df_summed = df_summed.with_columns(pl.lit(1.0).alias(VALUE_COLUMN))

        df_ratio = splitter.paths.join_over_index(df_summed)
        df_ratio = df_ratio.divide_cols([VALUE_COLUMN, VALUE_COLUMN + '_right'], VALUE_COLUMN)
        df_ratio = df_ratio.with_columns(
            pl.when(pl.col(VALUE_COLUMN + '_right') == 0).then(pl.lit(0.0)).otherwise(pl.col(VALUE_COLUMN)).alias(VALUE_COLUMN)
        ).drop([VALUE_COLUMN + '_right'])

        df_scaled = splittee.paths.multiply_with_dims(df_ratio)

        if operation not in use_as:
            df_scaled = splittee.paths.add_with_dims(df_scaled)

        return df_scaled

    def _operation_add_non_stackable_dims(self, df: PathsDataFrame, operation: OperationType) -> OperationReturn:
        nodes = self.get_input_nodes(tag=operation)
        if not nodes:
            raise NodeError(self, f"At least one input node must have tag '{operation}'.")
        for node in nodes:
            dfin = node.get_output_pl(target_node=self, skip_dim_test=True)
            if operation == 'add_from_incoming_dims':
                splitter = dfin
                splittee = df
            else:
                splitter = df
                splittee = dfin

            assert isinstance(splitter, PathsDataFrame)
            assert isinstance(splittee, PathsDataFrame)

            newdims = [dim for dim in splittee.dim_ids if dim not in splitter.dim_ids]
            if newdims:
                raise NodeError(self, f'Splittee node cannot bring in new dimensions but has {newdims}.')

            dims = [dim for dim in splitter.dim_ids if dim not in splittee.dim_ids]
            if not dims and not newdims:
                raise NodeError(self, "No dimensions to split. Remove the split operation if you don't use it.")

            df_unity = splitter.with_columns(pl.lit(1.0).alias(VALUE_COLUMN))
            df_unity = df_unity.set_unit(VALUE_COLUMN, 'dimensionless', force=True)
            df_added = splittee.paths.multiply_with_dims(df_unity)
            df = df_added.paths.add_with_dims(splitter)

        return df

    # Splitting functions
    def _operation_use_as_totals(self, df: PathsDataFrame) -> OperationReturn:
        return self._operation_split_dims(df, 'use_as_totals')

    def _operation_use_as_shares(self, df: PathsDataFrame) -> OperationReturn:
        return self._operation_split_dims(df, 'use_as_shares')

    def _operation_split_by_existing_shares(self, df: PathsDataFrame) -> OperationReturn:
        return self._operation_split_dims(df, 'split_by_existing_shares')

    def _operation_split_evenly_to_cats(self, df: PathsDataFrame) -> OperationReturn:
        return self._operation_split_dims(df, 'split_evenly_to_cats')

    def _operation_add_to_existing_dims(self, df: PathsDataFrame) -> OperationReturn:
        return self._operation_split_dims(df, 'add_to_existing_dims')

    def _operation_add_from_incoming_dims(self, df: PathsDataFrame) -> OperationReturn:
        if self.quantity in STACKABLE_QUANTITIES:
            raise NodeError(self, f'Node cannot have stackable quantity but has {self.quantity}.')
        return self._operation_split_dims(df, 'add_from_incoming_dims')

    def drop_unnecessary_levels(self, df: PathsDataFrame, droplist: list[str]) -> PathsDataFrame:
        # Drop filter levels and empty dimension levels.
        drops = [d for d in droplist if d in df.columns]
        df = df.drop(drops)
        # Get all metric columns from the DataFrame's metadata
        metric_cols = list(df.get_meta().units.keys())

        # Only drop rows where all metric columns are null
        if metric_cols:
            null_condition = pl.lit(True)  # noqa: FBT003
            for col in metric_cols:
                null_condition = null_condition & pl.col(col).is_null()
            df = df.filter(~null_condition)

        null_cols = [col for col in df.columns if df[col].null_count() == len(df)]
        df = df.drop(null_cols)
        return df

    # -----------------------------------------------------------------------------------
    def add_missing_years(self, df: PathsDataFrame) -> PathsDataFrame:
        return df.paths._add_missing_years(df, self.context)

    def get_operation_span_attrs(
        self,
        op_name: str,
        df: PathsDataFrame | None,
        *,
        source: Literal['node', 'dataframe'],
    ) -> PerfAttrs:
        attrs: PerfAttrs = {
            'node.id': self.id,
            'node.class': type(self).__name__,
            'generic.op.name': op_name,
            'generic.op.source': source,
        }
        if df is not None:
            attrs['generic.op.input.rows'] = len(df)
            attrs['generic.op.input.columns'] = len(df.columns)
            attrs['generic.op.input.in_memory.bytes'] = estimate_size_bytes(df)
        return attrs

    def set_operation_result_attrs(self, event: PerfSpanEntry[Any] | None, df: PathsDataFrame | None) -> None:
        if event is None:
            return
        if df is None:
            event.set_attr('generic.op.output.none', value=True)
            return
        event.set_attr('generic.op.output.rows', len(df))
        event.set_attr('generic.op.output.columns', len(df.columns))
        event.set_attr('generic.op.output.in_memory.bytes', estimate_size_bytes(df))

    @overload
    def _measured_op(
        self,
        func: Callable[[PathsDataFrame, Context], OperationReturn],
        op_name: str,
        *,
        source: Literal['dataframe'],
    ) -> Callable[[PathsDataFrame, Context], OperationReturn]: ...

    @overload
    def _measured_op(
        self,
        func: Callable[[PathsDataFrame | None], OperationReturn],
        op_name: str,
        *,
        source: Literal['node'],
    ) -> Callable[[PathsDataFrame | None], OperationReturn]: ...

    def _measured_op(
        self,
        func: Callable[..., OperationReturn],
        op_name: str,
        *,
        source: Literal['node', 'dataframe'],
    ) -> Callable[..., OperationReturn]:
        @functools.wraps(func)
        def wrapped(df: PathsDataFrame | None, context: Context | None = None) -> OperationReturn:
            hide_from_traceback()
            attrs = self.get_operation_span_attrs(op_name, df, source=source)
            span_name = f'{self.id}: {source} op {op_name}'

            if source == 'dataframe':
                assert df is not None
                assert context is not None
            else:
                assert source == 'node'
                assert context is None
            with self.context.start_perf_span(
                span_name,
                kind=PerfKind.NODE,
                id=self.id,
                op=f'generic.{op_name}',
                attributes=attrs,
            ) as (_, event):
                if context is None:
                    result = func(df)
                else:
                    result = func(df, context)
                self.set_operation_result_attrs(event, result)
            return result

        return wrapped

    # -----------------------------------------------------------------------------------
    def _operation_select_variant(self, df: PathsDataFrame) -> OperationReturn:
        filt = self.get_parameter_value_str('categories', required=False)
        if filt is not None:
            # Validate format: "dimension:category[,category...]"
            if ':' not in filt:
                raise ValueError(f"Categories parameter must contain ':' to separate dimension and values, got {filt}")

            dim, cats = filt.split(':')
            dim = dim.strip()

            if dim not in df.columns:
                raise ValueError(f"Dimension '{dim}' not found in DataFrame columns: {list(df.columns)}")

            cat_list = [cat.strip() for cat in cats.split(',')]

            # Get unique values in the dimension
            unique_values = df[dim].unique().to_list()
            invalid_cats = [cat for cat in cat_list if cat not in unique_values]
            if invalid_cats:
                raise ValueError(
                    f"Categories {invalid_cats} not found in dimension '{dim}'. Valid categories are: {unique_values}"
                )

            do = self.get_typed_parameter_value('do_correction', bool, required=False)
            if do is not None:
                condition = pl.col(dim).is_in(cat_list)
                return df.filter(condition if do else ~condition)

            val = self.get_parameter_value('selected_number', required=True)
            if isinstance(val, (int, float)):
                idx = round(val)
                if idx < 0 or idx >= len(cat_list):
                    raise ValueError(f'Selected number {val} is out of range for categories {cat_list}')
                cat = cat_list[idx]
            else:
                cat = cat_list[0]  # Default to first category if no selection

            df = df.filter(pl.col(dim) == cat)

        return df

    # -----------------------------------------------------------------------------------
    def _operation_do_correction(self, df: PathsDataFrame | None) -> OperationReturn:
        if df is None:
            raise NodeError(self, "Dataframe missing for 'do correction'.")
        do_correction = self.get_parameter_value('do_correction', required=True)
        no_correction_value = self.get_typed_parameter_value('no_correction_value', float, required=False)
        if no_correction_value is None:
            no_correction_value = 1.0

        if not do_correction:
            df = df.with_columns(pl.col(VALUE_COLUMN) * pl.lit(0) + pl.lit(no_correction_value))

        return df

    # -----------------------------------------------------------------------------------

    def compute(self) -> PathsDataFrame:
        """Process inputs according to the operations sequence."""
        # Get operation sequence from parameter or class default
        operations_str = self.get_parameter_value_str('operations', required=False) or self.default_operations
        operations = [op.strip() for op in operations_str.split(',')]

        if 'goal_gap' in operations and operations != ['goal_gap']:
            raise NodeError(
                self,
                'goal_gap must be the first and only operation; got operations: %s.' % operations_str,
            )

        df = None
        for op_name in operations:
            if df is not None and df.paths.has_operation(op_name):
                op = self._measured_op(
                    df.paths.get_operation(op_name),
                    op_name,
                    source='dataframe',
                )
                df = op(df, self.context)
            else:
                if op_name not in self.OPERATIONS:
                    raise NodeError(self, f'Unknown operation: {op_name}')
                op = self._measured_op(
                    self.OPERATIONS[op_name],
                    op_name,
                    source='node',
                )
                df = op(df)
        if not isinstance(df, PathsDataFrame):
            raise NodeError(self, 'The output is not a PathsDataFrame.')

        if VALUE_COLUMN not in df.columns:
            raise NodeError(self, f'{VALUE_COLUMN} not found, only {df.metric_cols}.')
        for col in df.metric_cols:
            unit = next(metric.unit for metric in self.output_metrics.values() if metric.column_id == col)
            df = df.ensure_unit(col, unit)

        return df


class ScenarioImpactNode(GenericNode):
    """Node that outputs scenario impact of one input node (current vs reference scenario)."""

    explanation = _('Outputs Scenario, Reference, and Impact blocks for the single input node (tag: scenario_impact).')
    allowed_parameters = [
        *GenericNode.allowed_parameters,
        StringParameter(
            local_id='scenario_impact_reference',
            label=_('Reference scenario (default: baseline)'),
        ),
    ]
    DEFAULT_OPERATIONS = 'scenario_impact'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.OPERATIONS['scenario_impact'] = self._operation_scenario_impact
        self.default_operations = self.DEFAULT_OPERATIONS

    def _operation_scenario_impact(self, _df: PathsDataFrame | None) -> OperationReturn:
        """Output = scenario impact of one input node (current vs reference). Tag: scenario_impact."""
        if _df is not None:
            raise NodeError(self, 'scenario_impact must be the first operation, so df must be None.')
        target_node = self.get_input_node(tag='scenario_impact', required=True)
        reference_scenario_id = self.get_parameter_value_str('scenario_impact_reference', required=False) or 'baseline'
        df = compute_scenario_impact(target_node, reference_scenario_id)
        df = df.filter(pl.col(IMPACT_COLUMN).eq(IMPACT_GROUP)).drop(IMPACT_COLUMN)  # Impact column not needed here.
        return df


class ActionWithHistoryNode(GenericNode):
    """
    Calculate output when there is an input_node action that started already during historical years.

        IF(forecast, latest - latest_action, observed - action_implemented) + action_active,

    where latest = latest observed value, latest_action = latest implemented action value,
    action = values from the action node, _implemented = historical_actions scenario, _active = active scenario.
    If action unit is not compatible with the input dataframe, assume multiplication instead of addition.
    """

    allowed_parameters = [
        *GenericNode.allowed_parameters,
        StringParameter(
            local_id='historical_actions_scenario_id',
            label=_('Scenario id for implemented historical actions'),
            description=_(
                'Defaults to "historical_actions". Implemented column uses this scenario; '
                + 'user column uses the active scenario.'
            ),
            is_customizable=False,
        ),
    ]
    DEFAULT_OPERATIONS = 'get_single_dataset,action_with_history'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.OPERATIONS['action_with_history'] = self._operation_action_with_history
        self.default_operations = self.DEFAULT_OPERATIONS

    def _operation_action_with_history(self, df: PathsDataFrame | None) -> OperationReturn:
        if df is None:
            raise NodeError(
                self,
                'ActionWithHistoryNode needs a dataframe from a previous operation.',
            )
        if VALUE_COLUMN not in df.columns:
            raise NodeError(self, 'action_with_history requires %s on the incoming dataframe.' % VALUE_COLUMN)
        if len(df.metric_cols) != 1:
            raise NodeError(
                self,
                'action_with_history expects exactly one metric on the incoming dataframe; got %s.' % df.metric_cols,
            )

        action_n = self.get_input_node(tag='action_with_history', required=True)
        hist_scenario_id = self.get_parameter_value_str('historical_actions_scenario_id', required=False) or 'historical_actions'

        idf = action_n.get_output_pl_for_scenario(hist_scenario_id, target_node=self)
        udf = action_n.get_output_pl(target_node=self)
        forecast = False
        idf = idf.with_columns(pl.lit(forecast).alias(FORECAST_COLUMN))
        udf = udf.with_columns(pl.lit(forecast).alias(FORECAST_COLUMN))

        ctx = self.context
        u1 = df.get_unit(VALUE_COLUMN)
        u2 = idf.get_unit(VALUE_COLUMN)
        assert u1 is not None
        assert u2 is not None

        if u1.is_compatible_with(u2):
            partial = df.paths.subtract_with_dims(idf, how='left')
            baseline_hist = partial.paths._inventory_only(partial, ctx)
            if not len(baseline_hist):
                raise NodeError(self, 'No historical rows after inventory_only(observed - implemented action).')

            baseline = baseline_hist.paths._extend_values(baseline_hist, ctx)
            out = baseline.paths.add_with_dims(udf, how='left')
        else:
            partial = df.paths.divide_with_dims(idf, how='inner')
            baseline_hist = partial.paths._inventory_only(partial, ctx)
            if not len(baseline_hist):
                raise NodeError(self, 'No historical rows after inventory_only(observed * implemented action).')

            baseline = baseline_hist.paths._extend_values(baseline_hist, ctx)
            out = baseline.paths.multiply_with_dims(udf, how='inner')

        return out


class LeverNode(GenericNode):
    explanation = _("""LeverNode replaces the upstream computation completely, if the lever is enabled.""")
    allowed_parameters = [
        *GenericNode.allowed_parameters,
        StringParameter(local_id='new_category'),
    ]

    def _operation_override_with_lever(self, df: PathsDataFrame | None) -> OperationReturn:
        """Override upstream computation with lever values if enabled."""
        if df is None:
            return None

        lever = self.get_input_node(tag='other_node', required=True)
        if not isinstance(lever, ActionNode):
            raise NodeError(self, f'Lever {lever} must be an action.')

        if not lever.is_enabled():
            return df

        dfl = lever.get_output_pl(target_node=self)
        dfl = dfl.ensure_unit(VALUE_COLUMN, df.get_unit(VALUE_COLUMN))
        out = df.paths.join_over_index(dfl, how='left', index_from='left')
        out = out.with_columns(
            (pl.when(pl.col(FORECAST_COLUMN)).then(pl.col(VALUE_COLUMN + '_right')).otherwise(pl.col(VALUE_COLUMN))).alias(
                VALUE_COLUMN
            )
        )
        out = out.drop(VALUE_COLUMN + '_right')

        if len(df) != len(out):
            s = f'({len(out)} rows) as the affected node {self.id} ({len(df)} rows).'
            raise NodeError(self, f'Lever {lever.id} must result in the same structure {s}')

        return out

    def _operation_fill_new_category(self, df: PathsDataFrame) -> OperationReturn:
        """Fill in a new category with complement values."""

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

        return df

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

    def _process_single_weighted_node(self, node: Node, node_weights: PathsDataFrame) -> PathsDataFrame:
        """Process a single node with its weights and return the weighted output."""
        # Get node output
        node_output = node.get_output_pl(target_node=self)
        node_output_unit = node_output.get_unit(VALUE_COLUMN)

        # Prefer the metric whose unit makes the weighted result compatible with this node's output unit.
        valid_metric = None
        metrics = node_weights.metric_cols.copy()
        for col in metrics:
            if col not in node_weights.columns or node_weights[col].null_count() == len(node_weights):
                continue
            weighted_unit = node_output_unit * node_weights.get_unit(col)
            if self.is_compatible_unit(self.unit, weighted_unit):
                valid_metric = col
                break

        if valid_metric is None:
            for col in metrics:
                if col in node_weights.columns and node_weights[col].null_count() < len(node_weights):
                    valid_metric = col
                    break

        if not valid_metric:
            raise NodeError(self, f'No valid metric column found for weight dataset for node {node.id}')

        drop_metrics = [col for col in metrics if col != valid_metric]
        if drop_metrics:
            node_weights = node_weights.drop(drop_metrics)

        # Create a version with this metric renamed to VALUE_COLUMN
        if valid_metric != VALUE_COLUMN:
            node_weights = node_weights.rename({valid_metric: VALUE_COLUMN})

        # Multiply node output with weights
        return node_output.paths.multiply_with_dims(node_weights, how='inner')

    def _combine_weighted_outputs(self, weights_df: PathsDataFrame, additive_nodes: list[Node]) -> PathsDataFrame | None:
        """Process and combine all weighted node outputs."""
        # Create a lookup map for additive nodes
        node_map = {node.id: node for node in additive_nodes}
        result = None

        # Process each unique node in the weights dataframe
        for node_id in weights_df['node'].unique():
            if node_id not in node_map:
                self.logger.warning(f'Node {node_id} not found in additive nodes')
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

    def _operation_add_with_weights(self, df: PathsDataFrame) -> OperationReturn:
        """Combine additive node outputs weighted by values in a multidimensional weights DataFrame."""
        weights_df = self.get_input_dataset_pl(tag='input_node_weights', required=False)
        if weights_df is None:
            return df

        # Clear stale value from previous compute(); we set it again below for this run's add op
        self._weighted_node_ids = set()
        additive_nodes, _ = self._get_add_multiply_nodes()
        if not additive_nodes:
            raise NodeError(
                self,
                "If node contains weights, it must contain additive input nodes (typically with tag 'additive').",
            )

        # Mark these nodes as consumed so _operation_add does not add them again
        self._weighted_node_ids = set(weights_df['node'].unique())

        result = self._combine_weighted_outputs(weights_df, additive_nodes)

        if result is None:
            self.logger.warning(f'No matching nodes found in weights DataFrame for {self.id}')
            return df
        return result


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

    def _operation_logit_transform(self, df: PathsDataFrame) -> OperationReturn:
        """Apply logit transform to combine observations with weighted sum."""

        # Ensure our weighted sum is in dimensionless units
        df = df.ensure_unit(VALUE_COLUMN, 'dimensionless')

        # Get observations dataset
        df_obs = self.get_input_dataset_pl(tag='observations', required=False)
        if df_obs is None:
            raise NodeError(self, f'LogitNode {self.id} must have one dataset for baseline values.')

        # Validate and transform observations to logit space
        df_obs = df_obs.ensure_unit(VALUE_COLUMN, 'dimensionless')
        test = df_obs.with_columns((pl.lit(0.0) < pl.col(VALUE_COLUMN)) & (pl.col(VALUE_COLUMN) < pl.lit(1.0)))['literal'].all()

        if not test:
            raise NodeError(self, f'All values in {self.id} must be between 0 and 1, exclusive.')

        df_obs = df_obs.with_columns((pl.col(VALUE_COLUMN) / (pl.lit(1.0) - pl.col(VALUE_COLUMN))).log().alias(VALUE_COLUMN))

        # Join observations with weighted sum
        df = df.paths.join_over_index(df_obs, how='outer', index_from='left')
        df = df.sum_cols([VALUE_COLUMN, VALUE_COLUMN + '_right'], VALUE_COLUMN).drop(VALUE_COLUMN + '_right')

        # Apply inverse logit function to get probabilities
        expr = pl.lit(1.0) / (pl.lit(1.0) + (pl.lit(-1.0) * pl.col(VALUE_COLUMN)).exp())
        df = df.with_columns(expr.alias(VALUE_COLUMN))
        df = df.ensure_unit(VALUE_COLUMN, self.unit)

        return df

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
    explanation = _('Reads in a dataset and filters and interprets its content according to the <i>sector</i> parameter.')
    default_unit = 'kt/a'
    quantity = EMISSION_QUANTITY
    allowed_parameters = [
        *GenericNode.allowed_parameters,
        StringParameter(local_id='sector', label=_('Sector path in HSY emission database'), is_customizable=False),
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

    def process_sector_data_pl(self, df: PathsDataFrame, columns: str | list[str]) -> PathsDataFrame:
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
        matching_sectors = df.filter(pl.col('sector').cast(pl.Utf8).str.contains(full_pattern))
        meta = matching_sectors.get_meta()

        if len(matching_sectors) == 0:
            raise NodeError(self, f"Sector pattern '{full_pattern}' not found in input")

        # Handle dimensions if specified
        if dimension_map:
            # Create dimension columns
            for level_idx, dim_name in dimension_map.items():
                # Split sector and extract the level
                matching_sectors = matching_sectors.with_columns(
                    pl.col('sector').cast(pl.Utf8).str.split('|').list.get(level_idx).alias(dim_name)
                )

                # Convert to proper dimension IDs if needed
                if dim_name in self.input_dimensions:
                    dim = self.input_dimensions[dim_name]
                    matching_sectors = matching_sectors.with_columns(
                        dim.series_to_ids_pl(matching_sectors[dim_name]).alias(dim_name)
                    )

            # Group by dimensions and year
            group_cols = [YEAR_COLUMN] + list(dimension_map.values())
            result = matching_sectors.group_by(group_cols).agg([pl.sum(col).alias(col) for col in columns])

            # Add dimension columns to index
            result = ppl.to_ppdf(result, meta)
            for dim_name in dimension_map.values():
                result = result.add_to_index(dim_name)
        else:
            # No dimensions, just group by year
            result = matching_sectors.group_by(YEAR_COLUMN).agg([pl.sum(col).alias(col) for col in columns])
            result = ppl.to_ppdf(result, meta)

        # Add forecast column if not present
        if FORECAST_COLUMN not in result.columns:
            result = result.with_columns(pl.lit(False).alias(FORECAST_COLUMN))  # noqa: FBT003

        return result

    def _operation_process_sector(self, df: PathsDataFrame | None) -> OperationReturn:
        """Process the sector data from HSY nodes."""
        if df is not None:
            raise NodeError(self, 'process_sector must be the first operation, so df must be None.')
        n = self.get_input_node(tag='other_node', required=True)
        data_df = n.get_output_pl()

        data_column = getattr(self, 'data_column', EMISSION_QUANTITY)
        result = self.process_sector_data_pl(data_df, columns=[data_column])

        result = result.rename({data_column: VALUE_COLUMN})
        result = extend_last_historical_value_pl(result, end_year=self.context.model_end_year)
        result = result.ensure_unit(VALUE_COLUMN, self.unit)
        return result

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register the sector processing operation
        self.OPERATIONS['process_sector'] = self._operation_process_sector
        # Set default operations sequence
        self.default_operations = 'process_sector,multiply,add,apply_multiplier'


class DimensionalSectorEmissions(DimensionalSectorNode):
    explanation = _('Filters emissions according to the <i>sector</i> parameter.')
    default_unit = 'kt/a'
    quantity = EMISSION_QUANTITY

    def __init__(self, *args, **kwargs):
        # Set the data column before initializing
        self.data_column = EMISSION_QUANTITY
        super().__init__(*args, **kwargs)


class DimensionalSectorEnergy(DimensionalSectorNode):
    explanation = _('Filters energy use according to the <i>sector</i> parameter.')
    default_unit = 'GWh/a'
    quantity = ENERGY_QUANTITY

    def __init__(self, *args, **kwargs):
        # Set the data column before initializing
        self.data_column = ENERGY_QUANTITY
        super().__init__(*args, **kwargs)


class DimensionalSectorEmissionFactor(DimensionalSectorNode):
    explanation = _('Filters emissions and energy according to the <i>sector</i> parameter and calculates emission factor.')
    default_unit = 'g/kWh'
    quantity = EMISSION_FACTOR_QUANTITY

    def _operation_process_emission_factor(self, df: PathsDataFrame | None) -> OperationReturn:
        """Calculate emission factors from energy and emission data."""
        if df is not None:
            raise NodeError(self, 'process_sector must be the first operation, so df must be None.')
        n = self.get_input_node(tag='other_node', required=True)
        data_df = n.get_output_pl()

        result = self.process_sector_data_pl(data_df, columns=[ENERGY_QUANTITY, EMISSION_QUANTITY])
        result = result.divide_cols([EMISSION_QUANTITY, ENERGY_QUANTITY], VALUE_COLUMN)
        result = result.drop([ENERGY_QUANTITY, EMISSION_QUANTITY])
        result = extend_last_historical_value_pl(result, end_year=self.context.model_end_year)
        result = result.ensure_unit(VALUE_COLUMN, self.unit)
        return result

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
        """
    )

    def _operation_year_iteration(self, df: PathsDataFrame | None) -> OperationReturn:
        """Perform year-by-year iteration using previous values, growth rate and changes."""
        rate_node = self.get_input_node(tag='rate', required=True)
        base_node = self.get_input_node(tag='base', required=True)

        # Get rate dataframe and prepare it
        rate_df = rate_node.get_output_pl(target_node=self)
        rate_df = rate_df.ensure_unit(VALUE_COLUMN, '1/a')
        rate_df = rate_df.set_unit(VALUE_COLUMN, 'dimensionless', force=True)
        rate_df = rate_df.with_columns(pl.col(VALUE_COLUMN) + pl.lit(1.0).alias(VALUE_COLUMN))

        # Get base values
        base_df = base_node.get_output_pl(target_node=self)

        # Add changes column (from additive inputs or default to 0)
        if df is None:
            # No changes provided - create empty changes column
            df_out = base_df.with_columns(pl.lit(0.0).alias('changes'))
        else:
            # Join changes from provided dataframe
            df_out = base_df.paths.join_over_index(df, how='left', index_from='left')
            df_out = df_out.rename({VALUE_COLUMN + '_right': 'changes'})
            # Adjust units for changes
            df_out = df_out.set_unit('changes', df_out.get_unit('changes') * unit_registry.parse_units('a'), force=True)
            df_out = df_out.ensure_unit('changes', df_out.get_unit(VALUE_COLUMN))

        # Fill missing changes with zero and join rates
        df_out = df_out.with_columns(pl.col('changes').fill_null(0.0))
        df_out = df_out.paths.join_over_index(rate_df, how='left', index_from='union')
        df_out = df_out.rename({VALUE_COLUMN + '_right': 'rate'})
        df_out = df_out.with_columns(pl.col('rate').fill_null(1.0))  # Default rate to 1.0 (no change)

        # Find historical and forecast boundary
        historical_df = df_out.filter(~pl.col(FORECAST_COLUMN))
        if len(historical_df) == 0:
            raise NodeError(self, 'IterativeNode must have historical values.')

        # Get years needed for iteration
        last_historical_year = historical_df[YEAR_COLUMN].sort().last()
        assert isinstance(last_historical_year, (int, float))
        last_historical_year = float(last_historical_year)
        end_year = self.get_end_year()

        # Track processed years to build the final result
        processed_years = {}

        keep_cols = [YEAR_COLUMN, FORECAST_COLUMN, VALUE_COLUMN] + df_out.dim_ids
        # First add all historical data to our result (unchanged)
        processed_years[last_historical_year] = df_out.filter(pl.col(YEAR_COLUMN) <= last_historical_year).select(keep_cols)

        # Process each forecast year sequentially
        for forecast_year in range(int(last_historical_year) + 1, int(end_year) + 1):
            # Get previous year's data
            prev_year = forecast_year - 1
            prev_year_data = processed_years[prev_year].filter(pl.col(YEAR_COLUMN) == prev_year).drop(YEAR_COLUMN)

            # Get current year's changes and rates
            current_year_data = df_out.filter(pl.col(YEAR_COLUMN) == forecast_year)

            if len(current_year_data) == 0:
                raise NodeError(self, f'Year {forecast_year} is missing in the input data')

            # Create a new dataframe for the current year by joining with previous year
            # Use outer join to ensure we process all dimension combinations
            result_year = current_year_data.paths.join_over_index(prev_year_data)
            result_year = result_year.rename({VALUE_COLUMN + '_right': 'prev_value'})

            # Now calculate the new values with a vectorized operation on all dimensions
            result_year = result_year.with_columns([
                # New value = (prev_value + changes) * rate
                ((pl.col('prev_value').fill_null(0.0) + pl.col('changes').fill_null(0.0)) * pl.col('rate').fill_null(1.0)).alias(
                    VALUE_COLUMN
                )
            ])

            # Add the needed columns to our processed years
            processed_years[forecast_year] = result_year.select(keep_cols)

        # Combine all years into final result
        result_frames = list(processed_years.values())
        final_result = result_frames[0]

        for dfy in result_frames[1:]:
            final_result = final_result.paths.concat_vertical(dfy)

        final_result = final_result.ensure_unit(VALUE_COLUMN, self.unit)

        return final_result

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register operations
        self.OPERATIONS['year_iteration'] = self._operation_year_iteration
        # Set default operations sequence to let standard add happen before year_iteration
        self.default_operations = 'multiply,add,year_iteration,other,apply_multiplier'


class CoalesceNode(GenericNode):
    explanation = _("""Coalesces the empty values with the values from the node with the tag 'coalesce'.""")
    DEFAULT_OPERATIONS = 'multiply,coalesce,add'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register threshold operation
        self.OPERATIONS['coalesce'] = self._operation_coalesce

    def _operation_coalesce(self, df: PathsDataFrame | None) -> OperationReturn:
        """Coalesce requires exactly one input with tag primary or secondary. Prefer using coalesce in formula."""
        if df is None:
            raise NodeError(self, 'Cannot apply coalesce because no PathsDataFrame is available.')

        primary = self.get_input_nodes(tag='primary')
        secondary = self.get_input_nodes(tag='secondary')
        if len(primary) == 1 and len(secondary) == 0:
            node = primary[0]
            df_co = node.get_output_pl(target_node=self)
            df = df_co.paths.coalesce_df(df, how='outer', debug=self.debug, id=self.id)
        elif len(primary) == 0 and len(secondary) == 1:
            node = secondary[0]
            df_co = node.get_output_pl(target_node=self)
            df = df.paths.coalesce_df(df_co, how='outer', debug=self.debug, id=self.id)
        else:
            raise NodeError(
                self,
                ("Coalesce requires exactly one input with tag 'primary' or 'secondary'; got %d primary, %d secondary.")
                % (len(primary), len(secondary)),
            )
        return df


class CohortNode(GenericNode):
    explanation = _(
        """
    Cohort node take in initial age structure (inventory) and follows the cohort in time as it ages.

    Harvest describes how much is removed from the cohort.
    """
    )

    output_metrics = {
        'hectares': NodeMetric('t_ha', 'area'),
        'total_volume': NodeMetric('m3_solid', 'volume'),
        'harvest_volume': NodeMetric('m3_solid/a', 'volume'),
        'natural_mortality': NodeMetric('m3_solid/a', 'volume'),
    }
    # allowed_parameters = [
    #     *GenericNode.allowed_parameters,
    #     NumberParameter('harvest_probabilities', 'Age of forest with most of the tree felling'),
    #     NumberParameter('harvest_fraction', 'Fraction of trees cut at the harvest age, rest is protected')
    # ]

    def get_transition_matrix(self, n_ages, harvest_probabilities):
        """Create a transition matrix with age-specific harvest probabilities."""
        # This method is fine as is - no changes needed
        # Ensure we have the right number of probabilities
        assert len(harvest_probabilities) == n_ages, 'Must provide harvest probability for each age group'

        # Create transition matrix
        transition_matrix = np.zeros((n_ages, n_ages))

        # For each current age
        for current_age in range(n_ages):
            harvest_prob = harvest_probabilities[current_age]

            # Sanity check on probability
            assert 0 <= harvest_prob <= 1, (
                f'Harvest probability must be between 0 and 1, got {harvest_prob} for age {current_age}'
            )

            # Probability of being harvested (moves to age 0)
            if harvest_prob > 0:
                transition_matrix[0, current_age] = harvest_prob

            # Probability of aging normally
            if current_age < n_ages - 1:
                # Move to next age with probability (1-harvest_prob)
                transition_matrix[current_age + 1, current_age] = 1.0 - harvest_prob
            else:
                # Last age group stays in place if not harvested
                transition_matrix[current_age, current_age] = 1.0 - harvest_prob

        return transition_matrix

    def simulate_age_dynamics(self, initial_ages, years, harvest_probabilities, growth_curve, mortality_rate_fn):
        """
        Core simulation function handling pure age dynamics and returning a DataFrame.

        Parameters
        ----------
        initial_ages
            numpy array of hectares by age (0 to max_age)
        years
            sequence of years (e.g. range(2020, 2051))
        harvest_probabilities
            probabilities of harvesting for each age
        growth_curve
            function that returns volume per hectare given age
        mortality_rate_fn
            function that returns mortality rate given age

        Returns
        -------
        Polars DataFrame with simulation results (without dimension data)

        """
        # Initialize results structure
        n_years = len(years)
        n_ages = len(initial_ages)

        # Output arrays for calculations
        hectares = np.zeros((n_years, n_ages))
        hectares[0, :] = initial_ages

        harvest_volume = np.zeros((n_years, n_ages))
        natural_mortality = np.zeros((n_years, n_ages))

        # Pre-calculate volume by age (internal only)
        volume_by_age = np.array([growth_curve(age) for age in range(n_ages)])

        # For each year in simulation (starting from second year)
        for year_idx in range(1, n_years):
            # Create transition matrix for this year
            transition_matrix = self.get_transition_matrix(n_ages, harvest_probabilities)

            # Calculate pre-transition state
            pre_hectares = hectares[year_idx - 1].copy()

            # Track harvest volume by age class
            for age in range(n_ages):
                # Calculate harvest volume based on probability for each age class
                prob = harvest_probabilities[age]
                if prob > 0:
                    vol_per_ha = volume_by_age[age]
                    hectares_harvested = pre_hectares[age] * prob
                    harvest_volume[year_idx, age] += hectares_harvested * vol_per_ha

            # Apply transition - this handles both aging and harvesting
            new_hectares = transition_matrix @ pre_hectares

            # Calculate natural mortality by age class
            mortality_hectares = np.zeros(n_ages)
            for age in range(n_ages):
                mort_rate = mortality_rate_fn(age)
                mortality_hectares[age] = new_hectares[age] * mort_rate
                new_hectares[age] -= mortality_hectares[age]

                # Record mortality volume by age
                natural_mortality[year_idx, age] = mortality_hectares[age] * volume_by_age[age]

            # Mortality creates new stands
            total_mortality_hectares = np.sum(mortality_hectares)
            if total_mortality_hectares > 0:
                new_hectares[0] += total_mortality_hectares

            # Store the new state
            hectares[year_idx] = new_hectares

        # Convert results directly to DataFrame
        results_data = []
        for year_idx, year in enumerate(years):
            for age in range(n_ages):
                # Calculate volume per ha and total volume
                volume_per_ha = volume_by_age[age]
                total_volume = hectares[year_idx, age] * volume_per_ha

                row = {
                    YEAR_COLUMN: year,
                    'annual_age': age,
                    'hectares': hectares[year_idx, age],
                    'total_volume': total_volume,
                    'harvest_volume': harvest_volume[year_idx, age],
                    'natural_mortality': natural_mortality[year_idx, age],
                }

                results_data.append(row)

        # Create and return DataFrame without dimension data
        return pl.DataFrame(results_data)

    def simulate_cohort(self, initial_year: PathsDataFrame, years: range, max_age: int = 161):
        """
        Simulate forest development for all dimension combinations.

        Parameters
        ----------
        initial_year
            DataFrame with initial state
        years
            Range of years to simulate
        max_age
            Maximum age to model (default: 161)

        Returns
        -------
        DataFrame with combined results

        """
        if len(initial_year.dim_ids) < 2:
            raise NodeError(self, 'CohortNode must receive at least one dimension in addition to Age.')
        # Process each dimension combination separately
        all_results = []

        # Get all unique dimension combinations
        dim_combinations = initial_year.select(initial_year.dim_ids).drop('annual_age').unique()

        for combo in dim_combinations.iter_rows(named=True):
            # Extract data for this combination of dimensions
            combo_data = initial_year.filter(pl.all_horizontal(pl.col(dim) == combo[dim] for dim in dim_combinations.columns))

            # Create array of hectares by age and harvest probabilities
            initial_ages = np.zeros(max_age)
            harvest_probabilities = np.zeros(max_age)

            for row in combo_data.iter_rows(named=True):
                age = row['annual_age']
                if age < max_age:
                    initial_ages[age] += row['hectares']
                    harvest_probabilities[age] = row['harvest_probability']

            # Get parameters for this dimension combination
            params = self.create_default_params(combo, max_age)

            # Extract parameters
            growth_curve = params['growth_curve']
            mortality_rate_fn = params['mortality_rates']

            # Run core simulation without dimension data
            sim_results = self.simulate_age_dynamics(initial_ages, years, harvest_probabilities, growth_curve, mortality_rate_fn)

            # Add dimension data to results
            for dim in dim_combinations.columns:
                sim_results = sim_results.with_columns(pl.lit(combo[dim]).alias(dim))

            # Add to results
            all_results.append(sim_results)

        # Combine all results
        out = pl.concat(all_results)

        # Apply metadata
        meta = initial_year.get_meta()
        meta.primary_keys = [col for col in out.columns if col in initial_year.primary_keys]
        meta.units['hectares'] = meta.units['hectares']
        meta.units['natural_mortality'] = Unit('m3_solid/a')
        meta.units['total_volume'] = Unit('m3_solid')
        meta.units['harvest_volume'] = Unit('m3_solid/a')

        return ppl.to_ppdf(out, meta=meta)

    def create_default_params(self, combo: dict[str, str], max_age: int) -> dict[str, Any]:
        """
        Create default parameters for a given dimension combination.

        Parameters
        ----------
        combo
            Dictionary containing dimension values (e.g. {'region': 'uusimaa', 'species': 'pine'})
        max_age
            Maximum age to model

        Returns
        -------
        Dictionary of parameters for simulation

        """
        # Site factors for growth - use region if available, otherwise default
        site_factors = {'herb-rich': 1.3, 'fresh': 1.0, 'sub-dry': 0.7, 'dry': 0.5}

        # Species factors for growth - use species if available, otherwise default
        species_factors = {'pine': 1.0, 'spruce': 1.2, 'birch': 0.8, 'other': 0.7}

        # Get region and species from combo if available, otherwise use defaults
        region = combo.get('region', 'unknown')
        species = combo.get('species', 'pine')

        # Get growth factors
        site_factor = site_factors.get(region, 0.8)
        sp_factor = species_factors.get(species, 1.0)

        # Create growth curve function
        def growth_curve(age: int) -> float:
            return site_factor * 7.0 * age * sp_factor  # 560 m3_solid / ha at 100 a

        # Create mortality rate function - could be customized based on other dimensions
        def mortality_rates(age: int) -> float:
            return 0.005 + 0.0005 * age / 10

        # Build parameter dictionary - include all combo items for reference
        params = {
            **combo,  # Include all dimension values
            'max_age': max_age,
            'growth_curve': growth_curve,
            'mortality_rates': mortality_rates,
        }

        return params

    def expand_to_annual_ages(
        self,
        forest_df: PathsDataFrame,
        age_groups: list[tuple[int, int]],
    ) -> PathsDataFrame:
        """Convert aggregated age group data to annual age classes."""
        annual_data = []
        meta = forest_df.get_meta()

        # Process each record in the dataframe
        for record in forest_df.iter_rows(named=True):
            age_group = record['age']
            hectares = record[VALUE_COLUMN]
            age_start = age_end = None
            for start, end in age_groups:
                if age_group == str(start):
                    age_start = start
                    age_end = end
                    break

            # For other age groups, distribute across years in the group
            assert age_start is not None
            assert age_end is not None
            years_in_group = age_end - age_start + 1

            # Distribute hectares evenly across years
            hectares_per_year = hectares / years_in_group

            for age in range(age_start, age_end + 1):
                annual_data.append({  # noqa: PERF401
                    **{k: v for k, v in record.items() if k not in {'age', VALUE_COLUMN, VALUE_COLUMN + '_right'}},
                    'annual_age': age,
                    'hectares': hectares_per_year,  # Initial_year
                    'harvest_probability': record[VALUE_COLUMN + '_right'],  # Harvest_probability
                })

        new_keys = {VALUE_COLUMN: 'hectares', VALUE_COLUMN + '_right': 'harvest_probability'}

        for old_key, new_key in new_keys.items():
            if old_key in meta.units:
                meta.units[new_key] = meta.units.pop(old_key)

        out = pl.DataFrame(annual_data)
        meta.primary_keys = [col for col in out.columns if col in meta.primary_keys + ['annual_age']]
        out = ppl.to_ppdf(out, meta=meta)

        return out

    def aggregate_to_age_groups(self, annual_results: PathsDataFrame, age_groups: list[tuple[int, int]]) -> PathsDataFrame:
        """Aggregate annual age results back to original age groups."""

        def find_age_group(age: int) -> str:  # , age_groups: list[tuple[int, int]]) -> str:
            for start, end in age_groups:
                if start <= age <= end:
                    return f'{start}'

            # If no range matches, return the start of the last group
            return f'{age_groups[-1][0]}'

        # Add age group ID
        df = annual_results.with_columns([pl.col('annual_age').map_elements(find_age_group, return_dtype=pl.Utf8).alias('age')])
        df = df.add_to_index('age')
        df = df.with_columns((pl.col(YEAR_COLUMN) > pl.col(YEAR_COLUMN).min()).alias(FORECAST_COLUMN))
        df = df.paths.sum_over_dims('annual_age')

        return df

    def compute(self) -> PathsDataFrame:
        # Define age groups
        age_groups = [(0, 0), (1, 20), (21, 40), (41, 60), (61, 80), (81, 100), (101, 120), (121, 140), (141, 160)]

        # Define simulation years
        years = range(2022, 2050)
        max_age = 161  # 0 to 160 years

        # Get input data
        node = self.get_input_node(tag='inventory')
        df = node.get_output_pl(target_node=self)

        # Get harvest probabilities and join with inventory data
        harvest = self.get_input_node(tag='harvest').get_output_pl(target_node=self)
        harvest = harvest.ensure_unit(VALUE_COLUMN, '1/a')
        df = df.paths.join_over_index(harvest).get_last_historical_values()

        # Convert to annual ages
        annual_forest = self.expand_to_annual_ages(df, age_groups)

        # Run the simulation with dimension-agnostic approach
        results = self.simulate_cohort(annual_forest, years, max_age)

        # Aggregate back to age groups
        return self.aggregate_to_age_groups(results, age_groups)


# This node class is copied from DatasetReduceAction
# TODO Both classes should be streamlined and chopped into smaller functions that both use.
class DatasetReduceNode(GenericNode):
    explanation = _("""
    Receive goal input from a dataset or node and cause a linear effect.

    The output will be a time series with the difference to the
    last historical value of the input node.

    The goal input can also be relative (for e.g. percentage
    reductions), in which case the input will be treated as
    a multiplier.
    """)

    allowed_parameters: ClassVar[list[Parameter[Any]]] = [
        BoolParameter(local_id='relative_goal'),
    ]

    def compute(self) -> PathsDataFrame:  # noqa: PLR0915
        n = self.get_input_node(tag='historical', required=False)
        if n is None:
            df = self.get_input_dataset_pl(tag='historical')
            if FORECAST_COLUMN not in df.columns:
                df = df.with_columns(pl.lit(value=False).alias(FORECAST_COLUMN))
            assert len(df.metric_cols) == 1
            df = df.rename({df.metric_cols[0]: VALUE_COLUMN})
        else:
            df = n.get_output_pl(target_node=self)
            df = df.filter(~pl.col(FORECAST_COLUMN))  # FIXME FOR DIFF

        max_hist_year = df[YEAR_COLUMN].max()
        df = df.filter(pl.col(YEAR_COLUMN) == max_hist_year)

        goal_input_df = self.get_input_dataset_pl(tag='goal', required=False)
        if goal_input_df is None:
            goal_input_node = self.get_input_node(tag='goal', required=True)
            goal_input_df = goal_input_node.get_output_pl(target_node=self)
        assert goal_input_df is not None
        gdf = goal_input_df

        gdf = gdf.paths.cast_index_to_str()
        df = df.paths.cast_index_to_str()

        if not set(gdf.dim_ids).issubset(set(self.input_dimensions.keys())):
            raise NodeError(self, 'Dimension mismatch to input nodes')

        # Filter historical data with only the categories that are
        # specified in the goal dataset.

        exprs = [pl.col(dim_id).is_in(gdf[dim_id].unique()) for dim_id in gdf.dim_ids]
        if exprs:
            df = df.filter(pl.all_horizontal(exprs))

        end_year = self.get_end_year()
        assert len(gdf.metric_cols) == 1
        gdf = (
            gdf.rename({gdf.metric_cols[0]: VALUE_COLUMN}).with_columns(pl.lit(True).alias(FORECAST_COLUMN))  # noqa: FBT003
        )

        is_mult = self.get_parameter_value('relative_goal', required=False)
        if is_mult:
            # If the goal series is relative (i.e. a multiplier), transform
            # it into absolute values by multiplying with the last historical values.
            gdf = gdf.rename({VALUE_COLUMN: 'Multiplier'})
            hdf = df.drop(YEAR_COLUMN)
            metric_cols = [m.column_id for m in self.output_metrics.values()]
            hdf = hdf.rename({m: 'Historical%s' % m for m in metric_cols})
            gdf = gdf.paths.join_over_index(hdf, how='outer', index_from='union')
            gdf = gdf.filter(~pl.all_horizontal([pl.col('Historical%s' % col).is_null() for col in metric_cols]))
            for m in self.output_metrics.values():
                col = m.column_id
                gdf = gdf.multiply_cols(['Multiplier', 'Historical%s' % col], col, out_unit=m.unit)
                gdf = gdf.with_columns(pl.col(col).fill_nan(None))
            gdf = gdf.select_metrics(metric_cols)

        df = df.paths.to_wide()
        gdf = gdf.paths.to_wide()

        meta = df.get_meta()
        gdf = gdf.filter(pl.col(YEAR_COLUMN) > max_hist_year)
        df = ppl.to_ppdf(pl.concat([df, gdf], how='diagonal'), meta=meta)
        df = df.paths.make_forecast_rows(end_year=self.get_end_year())
        df = df.with_columns([pl.col(m).interpolate() for m in df.metric_cols])

        # Change the time series to be a difference to the last historical
        # year.
        exprs = [pl.col(m) - pl.first(m) for m in df.metric_cols]
        df = df.select([YEAR_COLUMN, FORECAST_COLUMN, *exprs])
        df = df.filter(pl.col(FORECAST_COLUMN))
        df = df.filter(pl.col(YEAR_COLUMN).lt(end_year + 1))
        df = df.paths.to_narrow()
        for m in self.output_metrics.values():  # TODO Not sure that multimetric functionalities are needed.
            if m.column_id not in df.metric_cols:
                raise NodeError(self, "Metric column '%s' not found in output" % m.column_id)
            df = df.ensure_unit(m.column_id, m.unit)
        return df


class GenerationCapacityNode(GenericNode):
    output_metrics = {
        'default': NodeMetric('MWh/a', 'energy', 'default', 'Energy generated with old and new capacity', VALUE_COLUMN),
        'emissions_avoided': NodeMetric(
            'kt_co2e/a', 'emissions', 'emissions_avoided', 'Emissions avoided by replacing other sources', 'emissions_avoided'
        ),
        'upstream_emissions': NodeMetric(
            'kt_co2e/a', 'emissions', 'upstream_emissions', 'Production emissions caused upstream', 'upstream_emissions'
        ),
    }
    allowed_parameters = [
        *GenericNode.allowed_parameters,
        NumberParameter(local_id='lifetime', label=_('Lifetime of the installation in full years')),
        NumberParameter(local_id='efficiency', label=_('Intrinsic production efficiency')),
        NumberParameter(local_id='performance_ratio', label=_('Performance without losses')),
        NumberParameter(local_id='ef_upstream_production', label=_('Scope 3 emissions from upstream of installation')),
    ]
    DEFAULT_OPERATIONS = 'add,generation_capacity'

    def _operation_generation_capacity(self, df: PathsDataFrame | None) -> OperationReturn:
        if df is None:
            raise NodeError(self, 'Node must receive new installations as input node(s).')

        stock = df.paths._cumulative(df, self.context)
        _lifetime = self.get_parameter_value_int('lifetime')  # FIXME Add retirement

        up = self.get_input_dataset_pl('ef_upstream_production', required=True)
        df = df.paths.multiply_with_dims(up).rename({VALUE_COLUMN: 'upstream_emissions'})

        efficiency = self.get_parameter_value('efficiency', required=True, units=True)
        assert isinstance(efficiency, Quantity)
        performance = self.get_parameter_value_float('performance_ratio', units=False)

        stock = (  # Calculate energy generation
            stock.multiply_quantity(VALUE_COLUMN, efficiency).with_columns(pl.col(VALUE_COLUMN) * pl.lit(performance))
        )

        ef = self.get_input_dataset_pl(tag='ef_displacement')

        stock = (  # FIXME Emissions avoided do double count with downstream emissions
            stock.paths
            .join_over_index(ef)  # TODO Should this be inner join?
            .multiply_cols([VALUE_COLUMN, VALUE_COLUMN + '_right'], 'emissions_avoided')
            .drop(VALUE_COLUMN + '_right')
            .with_columns(pl.col('emissions_avoided').fill_null(0.0))
        )
        df = df.paths.join_over_index(stock)

        return df

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.OPERATIONS['generation_capacity'] = self._operation_generation_capacity


class ChpNode(GenericNode):
    allowed_parameters = [
        *GenericNode.allowed_parameters,
        StringParameter(local_id='method', label=_('Emission splitting method')),
        NumberParameter(local_id='electricity_fraction', label=_('Fraction of electricity in the output energy')),
        NumberParameter(local_id='t_supply', label=_('Temperature (in K) of district heating supply slow')),
        NumberParameter(local_id='t_return', label=_('Temperature (in K) of district heating return flow')),
        NumberParameter(local_id='electricity_reference_efficiency', label=_('Efficiency of producing electricity separately')),
        NumberParameter(local_id='heat_reference_efficiency', label=_('Efficiency of producing heat separately')),
    ]
    DEFAULT_OPERATIONS = 'add,chp_ef_split'

    def _operation_chp_ef_split(self, df: PathsDataFrame | None) -> OperationReturn:
        if df is None:
            raise NodeError(self, 'Node must receive average CHP fuel emission factors.')

        if 'energy_carrier' in df.dim_ids:
            raise NodeError(self, 'Input emission factors contain dimension energy_carrier but it must be averaged over it.')

        methods = {
            'energy_content': self._energy_method,
            'work_potential': self._exergetic_method,
            'bisko': self._bisko_method,
            'efficiency': self._efficiency_method,
        }
        method = self.get_parameter_value_str('method', required=True)
        metfun = methods.get(method)
        if metfun is None:
            raise NodeError(self, f'Parameter method got value {method} but must be one of: {methods.keys()}.')

        df = metfun(df)  # Add z factors

        f_el = self.get_parameter_value_float('electricity_fraction', required=True)
        df = df.with_columns([pl.lit(f_el).alias('f_el'), pl.lit(1.0 - f_el).alias('f_heat')])
        df = df.with_columns([
            (pl.col('z_el') * pl.col('f_el') / (pl.col('z_el') * pl.col('f_el') + pl.col('z_heat') * pl.col('f_heat'))).alias(
                'a_el'
            )
        ])
        df = df.with_columns([(pl.lit(1.0) - pl.col('a_el')).alias('a_heat')])

        drops = ['f_el', 'f_heat', 'z_el', 'z_heat', 'a_el', 'a_heat']
        df_el = (
            df
            .with_columns([pl.col(VALUE_COLUMN) * pl.col('a_el'), pl.lit('electricity').alias('energy_carrier')])
            .drop(drops)
            .add_to_index('energy_carrier')
        )
        df_heat = (
            df
            .with_columns([pl.col(VALUE_COLUMN) * pl.col('a_heat'), pl.lit('district_heating').alias('energy_carrier')])
            .drop(drops)
            .add_to_index('energy_carrier')
        )
        df = df_el.paths.concat_vertical(df_heat)

        return df

    def _energy_method(self, df: PathsDataFrame) -> PathsDataFrame:
        df = df.with_columns([
            pl.lit(1.0).alias('z_el'),
            pl.lit(1.0).alias('z_heat'),
        ])
        return df

    def _exergetic_method(self, df: PathsDataFrame) -> PathsDataFrame:
        t_supply = self.get_parameter_value_float('t_supply', required=True)
        t_return = self.get_parameter_value_float('t_return', required=True)
        z_heat = 1 - t_return / t_supply
        df = df.with_columns([
            pl.lit(1.0).alias('z_el'),
            pl.lit(z_heat).alias('z_heat'),
        ])
        return df

    def _bisko_method(self, df: PathsDataFrame) -> PathsDataFrame:
        t_supply = self.get_parameter_value_float('t_supply', required=True)
        t_return = 283
        z_heat = 1 - t_return / t_supply
        df = df.with_columns([
            pl.lit(1.0).alias('z_el'),
            pl.lit(z_heat).alias('z_heat'),
        ])
        return df

    def _efficiency_method(self, df: PathsDataFrame) -> PathsDataFrame:
        n_el = self.get_parameter_value_float('electricity_reference_efficiency', required=True)
        n_heat = self.get_parameter_value_float('heat_reference_efficiency', required=True)
        df = df.with_columns([
            pl.lit(1.0 / n_el).alias('z_el'),
            pl.lit(1.0 / n_heat).alias('z_heat'),
        ])
        return df

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.OPERATIONS['chp_ef_split'] = self._operation_chp_ef_split


class ConstantNode(GenericNode):
    allowed_parameters = [
        *GenericNode.allowed_parameters,
        NumberParameter(local_id='constant', label=_('Constant value')),
        BoolParameter(local_id='condition', label=_('Boolean parameter to convert to float')),
    ]
    DEFAULT_OPERATIONS = 'constant,add'

    def _operation_constant(self, df: PathsDataFrame | None) -> OperationReturn:
        constant = self.get_parameter_value('constant', required=False, units=True)
        if constant is None:
            const_bool = self.get_typed_parameter_value('condition', bool, required=True)
            const_float = 1.0 if const_bool else 0.0
            constant = const_float * Quantity(1.0, 'dimensionless')
        if df is not None:
            raise NodeError(self, "Operation 'constant' must be the first of the operations.")
        start_year = self.context.instance.reference_year
        end_year = self.context.instance.model_end_year
        last_historical_year = self.context.instance.maximum_historical_year
        if last_historical_year is None or last_historical_year < start_year:
            last_historical_year = start_year
        years = range(start_year, end_year + 1)
        df = PathsDataFrame({YEAR_COLUMN: years})
        df._units = {}
        df._primary_keys = [YEAR_COLUMN]
        df = df.with_columns([
            pl.lit(constant.m).alias(VALUE_COLUMN),
            (pl.col(YEAR_COLUMN) > pl.lit(last_historical_year)).alias(FORECAST_COLUMN),
        ]).set_unit(VALUE_COLUMN, str(constant.units))

        return df

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.OPERATIONS['constant'] = self._operation_constant


class DatasetPlusOneNode(GenericNode):
    """
    Goal-setting node that includes reference_year+1 data alongside the regular dataset.

    When the global parameter 'measure_data_baseline_year_only' is True, the normal
    GenericNode pipeline would otherwise discard all years except those needed for
    inventory. This subclass inserts a 'baseline_plus_one' operation right after
    'get_single_dataset' to additionally keep reference_year+1, which some downstream
    action nodes require for interpolation.

    Operation order: get_single_dataset → baseline_plus_one → multiply → add → other → apply_multiplier
    """

    explanation = _(
        'GenericNode for goal-setting: keeps reference_year+1 rows in addition to the '
        + 'standard baseline filtering when measure_data_baseline_year_only is enabled.'
    )
    DEFAULT_OPERATIONS = 'get_single_dataset,baseline_plus_one,multiply,add,other,apply_multiplier'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.OPERATIONS['baseline_plus_one'] = self._operation_baseline_plus_one

    def _operation_baseline_plus_one(self, df: PathsDataFrame | None) -> OperationReturn:
        """
        Filter data based on the following rules.

        When measure_data_baseline_year_only is True, filter data to:
          - reference_year
          - reference_year + 1  (needed by downstream action interpolation)
          - years beyond maximum_historical_year
          - any row already marked as forecast
        """
        if df is None:
            return None
        if not self.get_global_parameter_value('measure_data_baseline_year_only', required=False):
            return df

        ref_year = self.context.instance.reference_year
        max_hist_year = self.context.instance.maximum_historical_year

        filt = (pl.col(YEAR_COLUMN) == ref_year) | (pl.col(YEAR_COLUMN) > max_hist_year)
        if isinstance(ref_year, int):
            filt = filt | (pl.col(YEAR_COLUMN) == ref_year + 1)
        if FORECAST_COLUMN in df.columns:
            filt = filt | pl.col(FORECAST_COLUMN)

        return df.filter(filt)


class ObservableNode(GenericNode):
    """
    GenericNode that blends modelled values with user observations.

    The dataset must be loaded as an ``ObservationDataset`` (tag
    ``observation_dataset`` in the YAML) so that it carries ``observed`` and
    ``placeholder`` boolean columns after loading.

    Operation ``apply_observations`` is inserted right after
    ``get_single_dataset``.  It reads the global parameter
    ``use_observations`` (bool) and the context reference year, then:

    * **Always** (all scenarios): overrides the reference-year value with the
      observation/placeholder value if one is available.  This anchors the
      model to real-world data at the start of the forecast.
    * **When** ``use_observations = True`` (progress-tracking scenario): uses
      *all* available historical observations, extended to cover the full model
      time range (equivalent to the old ``observed_only_extend_all`` formula).
    * **Otherwise** (default scenario): uses the modelled output for all years
      except reference year.

    The uuid dimension (if present) is dropped before returning so downstream
    nodes only see the semantic category dimensions.
    """

    explanation = _('GenericNode that blends modelled values with user observations from the database.')
    DEFAULT_OPERATIONS = 'get_single_dataset,apply_observations,multiply,add,other,apply_multiplier'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.OPERATIONS['apply_observations'] = self._operation_apply_observations

    def _get_add_multiply_nodes(self) -> tuple[list[Node], list[Node]]:
        """Exclude `modeled`-tagged inputs from add/multiply: they are consumed by apply_observations."""
        add_nodes, multiply_nodes = super()._get_add_multiply_nodes()
        modeled_ids = {edge.input_node.id for edge in self.edges if edge.output_node == self and 'modeled' in edge.tags}
        add_nodes = [n for n in add_nodes if n.id not in modeled_ids]
        multiply_nodes = [n for n in multiply_nodes if n.id not in modeled_ids]
        return add_nodes, multiply_nodes

    # ------------------------------------------------------------------
    # Dataset loading (sparse — do NOT extend to all years)
    # ------------------------------------------------------------------

    def _operation_get_single_dataset(self, df: PathsDataFrame | None) -> OperationReturn:
        """
        Load observation dataset as-is, without extending to all model years.

        The parent class ``get_cleaned_dataset`` would call ``_extend_values``
        which forward-fills ``observed`` / ``placeholder`` flags to model_end_year,
        making every future year look "observed".  We intentionally skip that step
        so ``apply_observations`` receives a sparse DataFrame that only covers the
        years actually present in the DVC data and user DB entries.
        """
        raw_df = self.get_input_dataset_pl(required=False)
        if raw_df is None:
            return df
        if len(raw_df.metric_cols) == 1:
            raw_df = raw_df.rename({raw_df.metric_cols[0]: VALUE_COLUMN})
        raw_df = raw_df.paths._drop_unnecessary_levels(raw_df, self.context)
        if df is None:
            return raw_df
        return df.paths.add_with_dims(raw_df)

    def _select_and_extend_observations(
        self,
        df: PathsDataFrame,
        *,
        use_obs: bool,
        ref_year: int,
    ) -> PathsDataFrame:
        """
        Pick the best available source per category and extend to all model years.

        - If *use_obs* is True:  user obs > placeholder > DVC default, all years.
        - If *use_obs* is False: only the reference-year row (obs/placeholder).

        Returns an empty DataFrame (len 0) if no suitable observations are available.
        """
        has_obs = pl.col('observed')
        has_any = pl.col('observed') | pl.col('placeholder')

        if use_obs:
            dim_ids = df.dim_ids
            if dim_ids:
                df = df.with_columns([
                    pl.col('observed').any().over(dim_ids).alias('_has_obs'),
                    has_any.any().over(dim_ids).alias('_has_any'),
                ])
            else:
                df = df.with_columns([
                    pl.col('observed').any().alias('_has_obs'),
                    has_any.any().alias('_has_any'),
                ])
            df = df.filter(
                pl.when(pl.col('_has_obs')).then(has_obs).when(pl.col('_has_any')).then(has_any).otherwise(pl.lit(True))  # noqa: FBT003
            ).drop(['_has_obs', '_has_any'])
        else:
            # Default scenario: only keep ref_year row if there is an obs/placeholder
            ref_with_data = df.filter((pl.col(YEAR_COLUMN) == ref_year) & has_any)
            if len(ref_with_data) == 0:
                # Signal "no observation at reference year"
                return df.filter(pl.lit(False))  # noqa: FBT003
            df = ref_with_data

        drop_cols = [c for c in ['observed', 'placeholder'] if c in df.columns]
        if drop_cols:
            df = df.drop(drop_cols)
        df = df.with_columns(pl.lit(False).alias(FORECAST_COLUMN))  # noqa: FBT003

        # Extend to cover the full model time range
        from nodes.calc import extend_last_forecast_value_pl, extend_to_history_pl

        end_year = self.context.instance.model_end_year
        df = extend_last_forecast_value_pl(df, end_year)
        start_year = self.context.instance.minimum_historical_year
        if start_year is not None:
            df = extend_to_history_pl(df, start_year)
        return df

    # ------------------------------------------------------------------
    # Operation
    # ------------------------------------------------------------------

    def _operation_apply_observations(self, df: PathsDataFrame | None) -> OperationReturn:
        """Blend observation data (from ObservationDataset) with modelled input."""
        if df is None:
            return None
        if 'observed' not in df.columns or 'placeholder' not in df.columns:
            # Dataset did not provide observation flags - no-op
            return df

        # Drop uuid dimension: it was only for DB lookup, not semantic
        df = df.drop('uuid', strict=False)

        use_obs: bool = bool(self.get_global_parameter_value('use_observations', required=False) or False)
        ref_year: int = self.context.instance.reference_year

        modeled_node = self.get_input_node(tag='modeled', required=False)
        modeled_df: PathsDataFrame | None = modeled_node.get_output_pl(target_node=self) if modeled_node is not None else None

        if use_obs:
            # progress_tracking: extend observations to all years
            result = self._select_and_extend_observations(df, use_obs=True, ref_year=ref_year)
            if len(result) == 0 and modeled_df is not None:
                return modeled_df
            return result

        # default scenario -------------------------------------------------------
        # Start with modelled output (all years), then override reference year.
        obs_at_ref = self._select_and_extend_observations(df, use_obs=False, ref_year=ref_year)

        if len(obs_at_ref) == 0:
            # No reference-year observation available: return modelled unchanged
            if modeled_df is not None:
                return modeled_df
            # No modelled input either: just return the dataset without flags
            return df.drop([c for c in ['observed', 'placeholder'] if c in df.columns])

        # We have a ref_year observation. Overlay it onto the modelled output.
        if modeled_df is None:
            # No modelled input: use extended obs directly
            return obs_at_ref

        # Join obs ref-year value onto modelled_df at reference year
        # obs_at_ref: all years (extended from ref_year observation)
        # We only want to inject the ref_year value into modelled_df
        ref_obs_row = obs_at_ref.filter(pl.col(YEAR_COLUMN) == ref_year)

        # Identify the shared dimension keys (Year + common dim_ids)
        shared_dims = [d for d in ref_obs_row.dim_ids if d in modeled_df.dim_ids]
        join_keys = [YEAR_COLUMN] + shared_dims

        ref_obs_join = ppl.to_ppdf(
            ref_obs_row.select(join_keys + [pl.col(VALUE_COLUMN).alias('_obs_val')]),
            meta=ppl.DataFrameMeta(primary_keys=join_keys, units={'_obs_val': ref_obs_row.get_unit(VALUE_COLUMN)}),
        )

        merged = ppl.to_ppdf(
            modeled_df.join(ref_obs_join, on=join_keys, how='left'),
            meta=modeled_df.get_meta(),
        )
        result = ppl.to_ppdf(
            merged.with_columns(
                pl
                .when(pl.col(YEAR_COLUMN) == ref_year)
                .then(pl.coalesce(['_obs_val', VALUE_COLUMN]))
                .otherwise(pl.col(VALUE_COLUMN))
                .alias(VALUE_COLUMN)
            ).drop('_obs_val'),
            meta=modeled_df.get_meta(),
        )
        return result
