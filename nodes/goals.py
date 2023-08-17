import typing
from dataclasses import dataclass

from pydantic import BaseModel, RootModel, Field, PrivateAttr, model_validator
import polars as pl
from common import polars as ppl
from common.i18n import I18nBaseModel, I18nString, I18nStringInstance, TranslatedString
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.exceptions import NodeError

if typing.TYPE_CHECKING:
    from .node import Node


class GoalValue(BaseModel):
    year: int
    value: float
    is_interpolated: bool = False


@dataclass
class GoalActualValue:
    year: int
    goal: float | None
    actual: float | None
    is_forecast: bool
    is_interpolated: bool


class NodeGoalsDimension(BaseModel):
    groups: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)


class NodeGoalsEntry(I18nBaseModel):
    values: list[GoalValue]
    label: I18nStringInstance | None = None
    normalized_by: str | None = None
    dimensions: dict[str, NodeGoalsDimension] = Field(default_factory=dict)
    linear_interpolation: bool = False
    is_main_goal: bool = False
    default: bool = False

    _node: 'Node' = PrivateAttr()

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self._node = None  # type:ignore

    def set_node(self, node: 'Node'):
        self._node = node

    def get_node(self) -> 'Node':
        return self._node

    def dim_to_path(self) -> str:
        entries = []
        for dim_id, path in self.dimensions.items():
            grp = '+'.join(path.groups)
            cat = '+'.join(path.categories)
            entries.append('%s:%s' % (dim_id, grp or cat))
        return '/'.join(entries)

    def get_dimension_categories(self) -> dict[str, list[str]]:
        out = {}
        for dim_id, gdim in self.dimensions.items():
            dim = self._node.context.dimensions[dim_id]
            if gdim.categories:
                cats = [*gdim.categories]
            else:
                cats = []
                assert len(gdim.groups)
                for grp in gdim.groups:
                    cats += [cat.id for cat in dim.get_cats_for_group(grp)]
            out[dim_id] = cats
        return out

    def filter_df(self, df: ppl.PathsDataFrame):
        goal_dims = self.get_dimension_categories()
        for dim_id in df.dim_ids:
            goal_cats = goal_dims.get(dim_id)
            if goal_cats is not None:
                df = df.filter(pl.col(dim_id).is_in(goal_cats))
        return df

    def get_id(self) -> str:
        id_parts: list[str] = [self._node.id]
        if self.dimensions:
            id_parts.append(self.dim_to_path())
        return '/'.join(id_parts)

    def get_normalization_info(self):
        context = self._node.context
        if self.normalized_by:
            if self.normalized_by not in context.normalizations:
                raise NodeError(self._node, "Goal normalization '%s' not found" % self.normalized_by)

            goal_norm = context.normalizations[self.normalized_by]
        else:
            goal_norm = None

        m = self._node.get_default_output_metric()
        if goal_norm:
            for q in goal_norm.quantities:
                if q.id == m.quantity:
                    break
            else:
                raise Exception()
            unit = q.unit
        else:
            unit = m.unit
        return goal_norm, unit


    def _get_values_df(self):
        context = self._node.context
        goal_norm, unit = self.get_normalization_info()
        m = self._node.get_default_output_metric()
        zdf = pl.DataFrame({YEAR_COLUMN: [x.year for x in self.values], m.column_id: [x.value for x in self.values]})
        df = ppl.to_ppdf(zdf, meta=ppl.DataFrameMeta(primary_keys=[YEAR_COLUMN], units={m.column_id: unit}))
        if goal_norm:
            df = goal_norm.denormalize_output(m, df)

        df = df.with_columns([pl.lit(False).alias('IsInterpolated')])

        if self.linear_interpolation and len(df) > 1:
            years = range(df[YEAR_COLUMN].min(), df[YEAR_COLUMN].max() + 1)  # type: ignore
            ydf = ppl.to_ppdf(
                pl.DataFrame(years, schema=[YEAR_COLUMN], orient='row'),
                meta=ppl.DataFrameMeta(primary_keys=[YEAR_COLUMN], units={})
            )
            df = df.paths.join_over_index(ydf, how='outer', index_from='left')
            df = df.with_columns([
                pl.col(VALUE_COLUMN).interpolate(),
                pl.col('IsInterpolated').fill_null(True)
            ])

        if context.active_normalization:
            _, df = context.active_normalization.normalize_output(m, df)
        return df

    def get_values(self):
        df = self._get_values_df()
        m = self._node.get_default_output_metric()
        return [
            GoalValue(year=row[YEAR_COLUMN], value=row[m.column_id], is_interpolated=row['IsInterpolated'])
            for row in df.to_dicts()
        ]

    def get_actual(self) -> list[GoalActualValue]:
        node = self._node
        df = node.get_output_pl()
        context = node.context
        for dim_id, path in self.dimensions.items():
            dim = context.dimensions[dim_id]
            if path.groups:
                f_expr = dim.ids_to_groups(pl.col(dim_id)).is_in(path.groups)
            else:
                f_expr = pl.col(dim_id).is_in(path.categories)
            df = df.filter(f_expr)
        df = df.paths.sum_over_dims()
        m = node.get_default_output_metric()
        if context.active_normalization:
            _, df = context.active_normalization.normalize_output(m, df)
        gdf = self._get_values_df()
        gdf = gdf.rename({m.column_id: 'Goal'})
        df = df.paths.join_over_index(gdf).sort(YEAR_COLUMN)
        out = [
            GoalActualValue(
                year=row[YEAR_COLUMN], actual=row[m.column_id], is_forecast=row[FORECAST_COLUMN],
                is_interpolated=row['IsInterpolated'], goal=row['Goal'],
            ) for row in df.to_dicts()
        ]
        return out


class NodeGoals(RootModel):
    root: typing.List[NodeGoalsEntry]
    _node: 'Node' = PrivateAttr()

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self._node = None  # type:ignore

    def set_node(self, node: 'Node'):
        self._node = node
        for ge in self.root:
            ge.set_node(node)


    @model_validator(mode='after')
    def validate_unique(self):
        entries: list[NodeGoalsEntry] = self.root
        paths = set()
        for entry in entries:
            path = entry.dim_to_path()
            if path in paths:
                raise ValueError('Duplicate dimensions in goals')
            paths.add(path)
        return self

    def get_dimensionless(self) -> NodeGoalsEntry | None:
        vals = list(filter(lambda x: not x.dimensions, self.root))
        if not vals:
            return None
        return vals[0]

    def get_exact_match(self, dimension_id: str, groups: list[str] = [], categories: list[str] = []) -> NodeGoalsEntry | None:
        for e in self.root:
            dim = e.dimensions.get(dimension_id)
            if not dim:
                continue
            if groups:
                if set(dim.groups) == set(groups):
                    break
            elif categories:
                if set(dim.categories) == set(categories):
                    break
        else:
            return None
        return e
