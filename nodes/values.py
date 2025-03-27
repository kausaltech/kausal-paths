from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from django.utils.translation import gettext_lazy as _

import polars as pl

from nodes.calc import extend_last_historical_value_pl
from nodes.units import Unit
from params.param import NumberParameter

from .actions.action import ActionNode
from .constants import VALUE_COLUMN
from .exceptions import NodeError
from .simple import AdditiveNode, MultiplicativeNode, SimpleNode

if TYPE_CHECKING:
    from common import polars as ppl

    from .node import Node


class UtilityNode(AdditiveNode): # FIXME Use generic.WeightedSumNode instead
    """
    Utility nodes take in outcome nodes and value weight parameters.

    They produce a dataframe showing the value-weighted sums of outcomes.
    Cost-effectiveness can be implemented if a decision criterion is known, e.g. 50 €/t.
    Then, emissions and cost are put to an equal scale by multiplying
    cost by 0.02 1/EUR and emissions by 1 1/t, resulting in a scale
    where cost-effective scenarios show value < 0 when cumulated over the time span.
    Therefore, in this case, the threshold parameter should get value 0.
    """

    allowed_parameters = AdditiveNode.allowed_parameters + [
        NumberParameter(
            local_id='impact_threshold',
            is_customizable=True,
        ),
        NumberParameter(
            local_id='emissions_weight',
            is_customizable=True,
        ),
        NumberParameter(
            local_id='economic_weight',
            is_customizable=True,
        ),
        NumberParameter(
            local_id='prosperity_weight',
            is_customizable=True,
        ),
        NumberParameter(
            local_id='purity_weight',
            is_customizable=True,
        ),
        NumberParameter(
            local_id='health_weight',
            is_customizable=True,
        ),
        NumberParameter(
            local_id='equity_weight',
            is_customizable=True,
        ),
        NumberParameter(
            local_id='biodiversity_weight',
            is_customizable=True,
        ),
        NumberParameter(
            local_id='legality_weight',
            is_customizable=True,
        )
    ]

    def weighted_impact(self, weight, quantity=None, tag=None):
        w = self.get_parameter_value(weight, required=False, units=True)
        if w is None:
            return None
        nodes = self.get_input_nodes(quantity=quantity, tag=tag)
        if not len(nodes):
            raise NodeError(self, "Tag %s not found in the inputs nodes" % tag)

        df = self.add_nodes_pl(None, nodes, unit=nodes[0].unit)
        df: ppl.PathsDataFrame = df.multiply_quantity(VALUE_COLUMN, w)
        if not self.is_compatible_unit(df.get_unit(VALUE_COLUMN), self.unit):
            raise NodeError(self, "Node(s) %s and their weight result in an incompatible unit %s, should be %s." % (
                nodes, df.get_unit(VALUE_COLUMN), self.unit))
        return df

    def sum_dfs(self, dfs: list[ppl.PathsDataFrame | None]) -> ppl.PathsDataFrame:
        def join_and_sum(df1: ppl.PathsDataFrame, df2: ppl.PathsDataFrame | None) -> ppl.PathsDataFrame:
            if df2 is None:
                return df1
            df = df1.paths.join_over_index(df2, how='outer', index_from='union')
            df = df.sum_cols([VALUE_COLUMN, VALUE_COLUMN + '_right'], VALUE_COLUMN)
            df = df.drop(VALUE_COLUMN + '_right')
            return df

        dfs = [x for x in dfs if x is not None]
        df = functools.reduce(lambda df1, df2: join_and_sum(df1, df2), [df for df in dfs])  # noqa: C416
        return df

    def compute(self) -> ppl.PathsDataFrame:
        nodes: list[Node] = []

        for node in self.input_nodes:
            if not isinstance(node, ActionNode):
                nodes += [node]
        assert len(nodes) > 0

        dfs = [
            self.weighted_impact('emissions_weight', tag='emissions'),
            self.weighted_impact('economic_weight', tag='economic'),
            self.weighted_impact('prosperity_weight', tag='prosperity'),
            self.weighted_impact('health_weight', tag='health'),
            self.weighted_impact('equity_weight', tag='equity'),
            self.weighted_impact('purity_weight', tag='purity'),
            self.weighted_impact('biodiversity_weight', tag='biodiversity'),
            self.weighted_impact('legality_weight', tag='legality')
        ]
        df = self.sum_dfs(dfs)
        df = df.ensure_unit(VALUE_COLUMN, self.unit)
        th = self.get_parameter_value('impact_threshold', required=True, units=True)
        th = th.to(df.get_unit(VALUE_COLUMN))
        df = df.with_columns([(
            pl.col(VALUE_COLUMN) - pl.lit(th.m)).alias(VALUE_COLUMN)
        ])

        df = self.maybe_drop_nulls(df)
        return df


class AssociationNode(SimpleNode):  # FIXME Use ignore_content operation instead
    explanation = _("""
    Association nodes connect to their upstream nodes in a loose way:
    Their values follow the relative changes of the input nodes but
    their quantities and units are not dependent on those of the input nodes.
    The node MUST have exactly one dataset, which is the prior estimate.
    Fractions 1..3 can be used to tell how much the input node should adjust
    the output node. The default relation is "increase", if "decrease" is used,
    that must be explicitly said in the tags.
    """)

    allowed_parameters = MultiplicativeNode.allowed_parameters + [
        NumberParameter(local_id='fraction1', label='Fraction for node 1', is_customizable=False),
        NumberParameter(local_id='fraction2', label='Fraction for node 2', is_customizable=False),
        NumberParameter(local_id='fraction3', label='Fraction for node 3', is_customizable=False)
    ]

    def compute(self) -> ppl.PathsDataFrame:
        df = self.get_input_dataset_pl()
        df = extend_last_historical_value_pl(df, end_year=self.get_end_year())
        for edge in self.edges:
            if edge.output_node is self:
                node = edge.input_node
                m = VALUE_COLUMN # = node.get_default_output_metric().column_id

                fraction = 0.1  # Default fraction of the output node's output
                for fr in ['fraction1', 'fraction2', 'fraction3']:
                    if fr in edge.tags:
                        fraction = self.get_parameter_value(fr, units=False, required=True)

                default = self.context.get_scenario('default')
                with default.override():
                    mean_in = node.get_output_pl(target_node=self)[m].mean()
                mean_out = df[VALUE_COLUMN].mean()
                assert isinstance(mean_in, float)
                assert isinstance(mean_out, float)

                if abs(mean_in) > 0.01:  # Relative adjustment makes no sense when too close to zero.
                    multiplier = fraction * mean_out / mean_in
                else:
                    multiplier = fraction
                if 'decrease' in edge.tags:
                    multiplier *= -1

                dfn = node.get_output_pl(target_node=self)
                df = df.paths.join_over_index(dfn, how='outer', index_from='union')
                df = df.with_columns(pl.col(m + '_right').fill_null(0))
                df = df.with_columns(
                    pl.col(m) + pl.lit(multiplier) * pl.col(m + '_right').alias(m)
                    ).drop(m + '_right')

        df = self.maybe_drop_nulls(df)
        return df


class LogicalNode(SimpleNode): # TODO Make a generic node operation instead
    # TODO Create a sticky_switch parameter: if the value changes, the change will be permanent.
    explanation = _(
        """
        This is a LogicalNode.

        It will take in logical inputs (with values 1 (True)
        or False (0)). Then it will operate Boolean AND or OR operators
        depending on the tags used. The 'and' tag is critical; otherwise 'or' is assumed.
        AND operations are performed first, then the OR operations. If you want more complex
        structures, use several subsequent nodes.
        """
    )
    allowed_parameters = [
        *SimpleNode.allowed_parameters,
    ]
    quantity = 'fraction'
    unit = Unit('dimensionless')

    def _validate_input(self, df: ppl.PathsDataFrame) -> None:
        if not df[VALUE_COLUMN].is_in([0, 1]).all():
            raise ValueError("Input values must be either 0 or 1")

    def _process_nodes(self, df: ppl.PathsDataFrame | None, nodes: list[Node],
                       boolean_and: bool) -> ppl.PathsDataFrame | None:
        if nodes:
            for n in nodes:
                df2 = n.get_output_pl(target_node=self)
                self._validate_input(df2)
                if df is None:
                    df = df2
                elif boolean_and:
                    df = df.paths.join_over_index(df2, how='inner', index_from='union')
                    assert df is not None
                    df = df.with_columns([
                        pl.col(VALUE_COLUMN) * pl.col(VALUE_COLUMN + '_right')
                    ]).drop(VALUE_COLUMN + '_right')
                else:
                    df = df.paths.join_over_index(df2, how='outer', index_from='union')
                    assert df is not None
                    df = df.with_columns(
                        pl.max_horizontal(
                            pl.col([VALUE_COLUMN, VALUE_COLUMN + '_right']).fill_null(0.0)
                    )).drop(VALUE_COLUMN + '_right')

        return df

    def compute(self) -> ppl.PathsDataFrame:
        if not self.input_nodes:
            raise ValueError("LogicalNode requires at least one input node")
        and_nodes: list[Node] = []
        or_nodes: list[Node] = []
        df: ppl.PathsDataFrame | None = None
        for n in self.input_nodes:
            tags: list[str] | None = None
            for edge in n.edges:
                if edge.output_node == self:
                    tags = edge.tags
                    break
            if 'and' in (tags or []):
                and_nodes.append(n)
            else:
                or_nodes.append(n)

        df = self._process_nodes(df, nodes=and_nodes, boolean_and=True)
        df = self._process_nodes(df, nodes=or_nodes, boolean_and=False)
        assert df is not None

        return df


class ThresholdNode(AdditiveNode): # TODO Create a generic node operation instead.
    explanation = _(
        """
        ThresholdNode computes the preliminary result like a regular AdditiveNode.
        Then it gives True (1) if the result if the preliminary result is grater
        than or equal to the threshold, otherwise False (0).
        """
    )
    allowed_parameters = [
        *AdditiveNode.allowed_parameters,
        NumberParameter(
            local_id='threshold',
            label='Gives 1 (True) if the preliminary output is >= threshold',
            is_customizable=True
        ),
    ]
    quantity = 'fraction'
    unit = Unit('dimensionless')

    def compute(self) -> ppl.PathsDataFrame:
        df = AdditiveNode.compute(self)
        threshold = self.get_parameter_value('threshold', units=True, required=True)
        u = str(threshold.units)
        df = df.ensure_unit(VALUE_COLUMN, u)
        df = df.with_columns([
            pl.when(pl.col(VALUE_COLUMN) >= pl.lit(threshold.m))
            .then(1.0)
            .otherwise(0.0).alias(VALUE_COLUMN)
        ])
        df = df.clear_unit(VALUE_COLUMN).set_unit(VALUE_COLUMN, 'dimensionless')

        return df
