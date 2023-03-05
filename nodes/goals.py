from __future__ import annotations

import typing

from pydantic import BaseModel
import polars as pl
from common import polars as ppl
from nodes.constants import YEAR_COLUMN
from nodes.exceptions import NodeError

if typing.TYPE_CHECKING:
    from .node import Node


class GoalValue(BaseModel):
    year: int
    value: float


class NodeGoals(BaseModel):
    values: list[GoalValue]
    normalized_by: str | None = None

    def get_values(self, node: Node):
        context = node.context
        if self.normalized_by:
            if self.normalized_by not in context.normalizations:
                raise NodeError(node, "Goal normalization '%s' not found" % self.normalized_by)

            goal_norm = context.normalizations[self.normalized_by]
        else:
            goal_norm = None

        if context.active_normalization == goal_norm:
            return self.values

        m = node.get_default_output_metric()
        zdf = pl.DataFrame({YEAR_COLUMN: [x.year for x in self.values], m.column_id: [x.value for x in self.values]})
        if goal_norm:
            for q in goal_norm.quantities:
                if q.id == m.quantity:
                    break
            else:
                raise Exception()
            unit = q.unit
        else:
            unit = m.unit

        df = ppl.to_ppdf(zdf, meta=ppl.DataFrameMeta(primary_keys=[YEAR_COLUMN], units={m.column_id: unit}))
        if goal_norm:
            df = goal_norm.denormalize_output(m, df)

        if context.active_normalization:
            _, df = context.active_normalization.normalize_output(m, df)

        return [GoalValue(year=row[YEAR_COLUMN], value=row[m.column_id]) for row in df.to_dicts()]
