from __future__ import annotations

import typing
from dataclasses import dataclass

import polars as pl

from .constants import KNOWN_QUANTITIES, VALUE_COLUMN, YEAR_COLUMN

if typing.TYPE_CHECKING:
    from common.polars import PathsDataFrame

    from .context import Context
    from .node import Node, NodeMetric
    from .units import Unit


@dataclass
class NormalizationQuantity:
    id: str
    unit: Unit


@dataclass
class Normalization:
    normalizer_node: Node
    quantities: list[NormalizationQuantity]
    default: bool = False

    @classmethod
    def from_config(cls, context: Context, config: dict[str, typing.Any]) -> typing.Self:
        node_id = config['normalizer_node']
        node = context.nodes[node_id]
        quantities = config['quantities']
        qs = []
        for q in quantities:
            q_id = q['id']
            assert q_id in KNOWN_QUANTITIES
            unit = context.unit_registry.parse_units(q['unit'])
            qs.append(NormalizationQuantity(id=q_id, unit=unit))
        return cls(normalizer_node=node, quantities=qs, default=config.get('default', False))

    def denormalize_output(self, to_metric: NodeMetric, df: PathsDataFrame) -> PathsDataFrame:
        for q in self.quantities:
            if to_metric.quantity == q.id:
                break
        else:
            raise Exception("Unable to denormalize")
        assert YEAR_COLUMN in df.primary_keys
        assert df.get_unit(to_metric.column_id) == q.unit

        ndf = self.normalizer_node.get_output_pl()
        ndf = (
            ndf.filter(pl.col(YEAR_COLUMN).is_in(df[YEAR_COLUMN]))
            .select([YEAR_COLUMN, pl.col(VALUE_COLUMN).alias('_N')])
        )
        df = df.paths.join_over_index(ndf, how='left')
        df = (
            df.multiply_cols([to_metric.column_id, '_N'], to_metric.column_id, to_metric.unit)
            .drop('_N')
        )
        return df

    def get_normalized_unit(self, metric: NodeMetric) -> Unit | None:
        for q in self.quantities:
            if metric.quantity == q.id:
                break
        else:
            return None

        normalized_unit: Unit = typing.cast('Unit', metric.unit / self.normalizer_node.unit)
        if not normalized_unit.is_compatible_with(q.unit):
            return None
        return normalized_unit

    def normalize_output(self, metric: NodeMetric, df: PathsDataFrame) -> tuple[Node | None, PathsDataFrame]:
        def nop() -> tuple[None, PathsDataFrame]:
            return (None, df)

        for q in self.quantities:
            if metric.quantity == q.id:
                break
        else:
            return nop()

        normalized_unit = self.get_normalized_unit(metric)
        if normalized_unit is None:
            self.normalizer_node.warning("Metric unit %s is incompatible with normalization" % metric.unit)
            return nop()
        if YEAR_COLUMN not in df.primary_keys:
            self.normalizer_node.warning("Year column not in dataframe")
            return nop()

        ndf = self.normalizer_node.get_output_pl()
        ndf = (
            ndf.filter(pl.col(YEAR_COLUMN).is_in(df[YEAR_COLUMN]))
            .select([YEAR_COLUMN, pl.col(VALUE_COLUMN).alias('_N')])
        )
        df = df.paths.join_over_index(ndf, how='left')
        df = (
            df.divide_cols([metric.column_id, '_N'], metric.column_id)
            .drop('_N')
            .ensure_unit(metric.column_id, q.unit)
        )

        return (self.normalizer_node, df)
