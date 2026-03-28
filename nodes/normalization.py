"""
Normalization of metrics (e.g. per-capita).

Normalization is applied in:
- nodes/metric_gen.py:381  - main chart / dimensional metric for a node
- nodes/schema.py:762       - GraphQL impact metric resolution
- nodes/metric.py:91, 106   - metric value resolution
- nodes/actions/action.py:234 - action effect computation
- pages/blocks.py:408       - block/impact display
- nodes/goals.py:149, 174   - goal values
- load_nodes.py:510         - CLI/script path
"""

from __future__ import annotations

import typing

from pydantic import BaseModel, field_validator

import polars as pl

from .constants import KNOWN_QUANTITIES, VALUE_COLUMN, YEAR_COLUMN
from .units import Unit

if typing.TYPE_CHECKING:
    from common.polars import PathsDataFrame

    from .context import Context
    from .node import Node, NodeMetric


class NormalizationQuantitySpec(BaseModel):
    """A quantity that can be normalized by a specific normalizer node."""

    id: str
    """The quantity identifier affected by this normalization."""

    unit: Unit
    """The unit expected after normalization has been applied."""

    @field_validator('id')
    @classmethod
    def validate_quantity_id(cls, value: str) -> str:
        """Ensure the normalization only references known quantity identifiers."""
        if value not in KNOWN_QUANTITIES:
            raise ValueError(f'Unknown quantity: {value}')
        return value


class NormalizationSpec(BaseModel):
    """Serialized normalization configuration stored in instance specs."""

    normalizer_node_id: str
    """The node id whose output is used as the normalization divisor."""

    quantities: list[NormalizationQuantitySpec]
    """Quantities that this normalization can be applied to."""

    default: bool = False
    """Whether this normalization should be activated by default for the instance."""


class Normalization:
    """Runtime normalization bound to a live normalizer node in a context."""

    spec: NormalizationSpec
    """The underlying serialized normalization definition."""

    normalizer_node: Node
    """The live node that provides normalization values."""

    def __init__(self, spec: NormalizationSpec, context: Context):
        self.spec = spec
        self.normalizer_node = context.nodes[spec.normalizer_node_id]

    def denormalize_output(self, to_metric: NodeMetric, df: PathsDataFrame) -> PathsDataFrame:
        """Multiply a normalized metric back to the original unit using the normalizer output."""
        for quantity in self.spec.quantities:
            if to_metric.quantity == quantity.id:
                break
        else:
            raise Exception('Unable to denormalize')
        assert YEAR_COLUMN in df.primary_keys
        assert df.get_unit(to_metric.column_id) == quantity.unit

        ndf = self.normalizer_node.get_output_pl()
        ndf = ndf.filter(pl.col(YEAR_COLUMN).is_in(df[YEAR_COLUMN])).select([YEAR_COLUMN, pl.col(VALUE_COLUMN).alias('_N')])
        df = df.paths.join_over_index(ndf, how='left')
        df = df.multiply_cols([to_metric.column_id, '_N'], to_metric.column_id, to_metric.unit).drop('_N')
        return df

    def get_normalized_unit(self, metric: NodeMetric) -> Unit | None:
        """Return the normalized unit for a metric, or `None` if this normalization does not apply."""
        for quantity in self.spec.quantities:
            if metric.quantity == quantity.id:
                break
        else:
            return None

        normalized_unit: Unit = typing.cast('Unit', metric.unit / self.normalizer_node.unit)
        if not normalized_unit.is_compatible_with(quantity.unit):
            return None
        return normalized_unit

    def normalize_output(self, metric: NodeMetric, df: PathsDataFrame) -> tuple[Node | None, PathsDataFrame]:
        """Divide a metric dataframe by the normalizer output when units are compatible."""

        def nop() -> tuple[None, PathsDataFrame]:
            return (None, df)

        for quantity in self.spec.quantities:
            if metric.quantity == quantity.id:
                break
        else:
            return nop()

        normalized_unit = self.get_normalized_unit(metric)
        if normalized_unit is None:
            self.normalizer_node.warning('Metric unit %s is incompatible with normalization' % metric.unit)
            return nop()
        if YEAR_COLUMN not in df.primary_keys:
            self.normalizer_node.warning('Year column not in dataframe')
            return nop()

        ndf = self.normalizer_node.get_output_pl()
        # Inner join keeps only years where the normalizer has data; divide_with_dims is the standard approach
        df = df.paths.divide_with_dims(ndf, how='inner')
        df = df.ensure_unit(metric.column_id, quantity.unit)
        return (self.normalizer_node, df)
