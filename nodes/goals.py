from __future__ import annotations

import typing

from pydantic import BaseModel, Field, ValidationError, root_validator
import polars as pl
from common import polars as ppl
from nodes.constants import YEAR_COLUMN
from nodes.exceptions import NodeError

if typing.TYPE_CHECKING:
    from .node import Node


class GoalValue(BaseModel):
    year: int
    value: float


class NodeGoalsDimension(BaseModel):
    group: str | None
    category: str | None


class NodeGoalsEntry(BaseModel):
    values: list[GoalValue]
    normalized_by: str | None = None
    dimensions: dict[str, NodeGoalsDimension] = Field(default_factory=dict)

    def dim_to_path(self) -> str:
        entries = []
        for dim_id, path in self.dimensions.items():
            entries.append('%s:%s' % (dim_id, path.group or path.category))
        return '/'.join(entries)

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


class NodeGoals(BaseModel):
    __root__: typing.List[NodeGoalsEntry]

    @root_validator
    @classmethod
    def validate_unique(cls, data: dict):
        entries: list[NodeGoalsEntry] = data['__root__']
        paths = set()
        for entry in entries:
            path = entry.dim_to_path()
            if path in paths:
                raise ValueError('Duplicate dimensions in goals')
            paths.add(path)
        return data

    def get_dimensionless(self) -> NodeGoalsEntry | None:
        vals = list(filter(lambda x: not x.dimensions, self.__root__))
        if not vals:
            return None
        return vals[0]

    def get_exact_match(self, dimension_id: str, group_id: str | None = None, category_id: str | None = None) -> NodeGoalsEntry | None:
        for e in self.__root__:
            dim = e.dimensions.get(dimension_id)
            if not dim:
                continue
            if group_id:
                if dim.group == group_id:
                    break
            elif category_id:
                if dim.category == category_id:
                    break
        else:
            return None
        return e
