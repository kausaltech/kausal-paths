from __future__ import annotations

import typing
from dataclasses import dataclass

import polars as pl

from common.polars import PathsDataFrame

from .constants import KNOWN_QUANTITIES, VALUE_COLUMN, YEAR_COLUMN

if typing.TYPE_CHECKING:
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

    @classmethod
    def from_config(cls, context: Context, config: dict[str, typing.Any]):
        node_id = config['normalizer_node']
        node = context.nodes[node_id]
        quantities = config['quantities']
        qs = []
        for q in quantities:
            q_id = q['id']
            assert q_id in KNOWN_QUANTITIES
            unit = context.unit_registry.parse_units(q['unit'])
            qs.append(NormalizationQuantity(id=q_id, unit=unit))
        return cls(normalizer_node=node, quantities=qs)

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

    def normalize_output(self, metric: NodeMetric, df: PathsDataFrame) -> typing.Tuple[Node | None, PathsDataFrame]:
        def nop():
            return (None, df)

        for q in self.quantities:
            if metric.quantity == q.id:
                break
        else:
            return nop()

        normalized_unit: Unit = metric.unit / self.normalizer_node.unit
        if not normalized_unit.is_compatible_with(q.unit):
            metric.node.warning("Metric unit %s is incompatible with normalization" % normalized_unit)
            return nop()
        if YEAR_COLUMN not in df.primary_keys:
            metric.node.warning("Year column not in dataframe")
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
