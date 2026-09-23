from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import polars as pl

from kausal_common.i18n.pydantic import gettext_lazy as _

from nodes.actions.action import ActionNode
from nodes.constants import VALUE_COLUMN
from nodes.formula import FormulaNode

if TYPE_CHECKING:
    from common.polars import PathsDataFrame
    from nodes.formula import EvalVars
    from nodes.node import Node


class FormulaAction(ActionNode, FormulaNode):
    """
    An action whose effect is a formula.

    In the formula, the identifier of a node the action acts on (see
    `nodes.hooks`) stands for that node's own value, before any action acts on
    it. That is how an effect stated relative to the node becomes the amount a
    hook adds, explicitly::

        end_energy_emission_factors * (factor - 1)
    """

    explanation = _(
        'This action computes its effect with a formula. In the formula, the identifier of a node the action '
        "acts on stands for that node's own value, before any action acts on it."
    )
    no_effect_value = 0.0
    allowed_parameters = [*ActionNode.allowed_parameters, *FormulaNode.allowed_parameters]

    def _formula_names(self) -> set[str]:
        formula = self.get_parameter_value_str('formula', required=False)
        if not formula:
            return set()
        tree = ast.parse(formula, '<string>', mode='eval')
        return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    def reads_hook_base(self, target: Node) -> bool:
        return target.id in self._formula_names()

    def _collect_eval_vars(self) -> EvalVars:
        varss = super()._collect_eval_vars()
        names = self._formula_names()
        for hook in self.hook_targets:
            target = hook.target
            if target.id not in names or target.id in varss.nodes:
                continue
            base = target.get_base_output_pl()
            varss.datasets[target.id] = base.select_metrics(hook.target_metric.column_id, rename=VALUE_COLUMN)
        return varss

    def compute_effect(self) -> PathsDataFrame:
        df = FormulaNode.compute(self)
        if not self.is_enabled():
            df = df.with_columns([pl.lit(self.no_effect_value).alias(col) for col in df.metric_cols])
        return df
