"""
Actions acting on other nodes' outputs ("hooks").

An action can act on any node: the node computes as it always does, and the
action's output is added to that result before it reaches the node's
consumers. In graph terms a hooked output behaves like an extra additive node
spliced into the target's outgoing edges; at runtime it is applied inside
`Node._get_output_pl()` instead of existing as a node.

A hook only ever adds. An action whose effect is naturally relative reads the
target's un-hooked value (its *base*, see `Node.get_base_output_pl()`) and
turns it into an amount itself, e.g. a `FormulaAction` with
`final_energy_use * (factor - 1)`. Every hook on a node sees the same base:
effects add up and each action's effect is exactly its own output.

Contributions only enter years after the instance's last historical year, so
a hook can never move a historical balance. The historical part of an action's
output is kept for counterfactual (historical-impact) views, which are not
built yet.

See docs/architecture/action-hooks.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from nodes.constants import FORECAST_COLUMN, YEAR_COLUMN
from nodes.exceptions import NodeError

if TYPE_CHECKING:
    from common import polars as ppl
    from nodes.actions.action import ActionNode
    from nodes.defs.node_defs import ActionHookDef
    from nodes.edges import Edge
    from nodes.node import Node, NodeMetric


@dataclass(eq=False)
class ActionHook:
    """Runtime form of `ActionHookDef`: `action` acts on `target_metric` of `target`."""

    action: ActionNode
    target: Node
    edge: Edge
    """Carries the dimension operations; never one of either node's `edges`."""
    target_metric: NodeMetric
    action_metric: NodeMetric
    definition: ActionHookDef
    reads_base: bool = False
    """Whether the action's computation reads the target's un-hooked output."""

    def hash_part(self) -> bytes:
        return self.definition.model_dump_json(exclude_none=True).encode('utf-8')


def last_historical_year(node: Node, df: ppl.PathsDataFrame) -> int | None:
    """Return the instance's last historical year, or else the last non-forecast year of `df`."""
    year = node.context.instance.maximum_historical_year
    if year is not None:
        return year
    if FORECAST_COLUMN not in df.columns:
        return None
    return df.filter(~pl.col(FORECAST_COLUMN))[YEAR_COLUMN].max()  # type: ignore[return-value]


def hook_contribution(hook: ActionHook, after_year: int | None) -> ppl.PathsDataFrame:
    """Shape the action's output as a contribution: one metric in the target's unit and dimensions, future years only."""
    action, target = hook.action, hook.target
    df = action.get_output_pl()
    col = hook.action_metric.column_id
    if col not in df.metric_cols:
        raise NodeError(action, 'Output has no metric %s to act on %s' % (col, target.id))
    target_col = hook.target_metric.column_id
    df = df.select_metrics(col, rename=target_col if col != target_col else None)
    df = action._apply_edge_transforms(df, hook.edge)
    if set(df.dim_ids) != set(target.output_dimensions):
        raise NodeError(
            action,
            'Dimensions (%s) do not match the output of %s it acts on (%s)'
            % (', '.join(df.dim_ids), target.id, ', '.join(target.output_dimensions)),
        )
    target_unit = hook.target_metric.unit
    if not df.get_unit(target_col).is_compatible_with(target_unit):
        raise NodeError(action, 'Unit %s cannot act on %s in %s' % (df.get_unit(target_col), target.id, target_unit))
    df = df.ensure_unit(target_col, target_unit)
    if after_year is None:
        return df
    return df.filter(pl.col(YEAR_COLUMN) > after_year)


def apply_hooks(node: Node, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
    """Add the contributions of the (enabled) actions acting on `node` to its un-hooked output `df`."""
    after_year = last_historical_year(node, df)
    for hook in node.hooks:
        if not hook.action.is_enabled():
            continue
        contribution = hook_contribution(hook, after_year)
        if not len(contribution):
            continue
        col = hook.target_metric.column_id
        right = '%s_right' % col
        joined = df.paths.join_over_index(contribution, how='outer', index_from='left')
        df = joined.with_columns(
            pl.when(pl.col(right).is_null()).then(pl.col(col)).otherwise(pl.col(col).fill_null(0.0) + pl.col(right)).alias(col),
        ).drop(right)
        if FORECAST_COLUMN in df.columns:
            df = df.with_columns(pl.col(FORECAST_COLUMN).fill_null(value=True))
    return df
