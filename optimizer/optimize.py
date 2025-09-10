from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod

# from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from io import StringIO
from typing import TYPE_CHECKING, Callable

import django

# from django.conf import settings
from pydantic import BaseModel

import networkx as nx
import numpy as np
import polars as pl
from rich import print
from rich.syntax import Syntax
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

# from ruamel.yaml.scalarfloat import ScalarFloat
from scipy import optimize

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'paths.settings')

# Configure Django
django.setup()

from common import polars as ppl  # noqa: E402, I001
from nodes.actions.shift import ShiftAction, ShiftParameterValue  # noqa: E402
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN  # noqa: E402
from nodes.instance import Instance  # noqa: E402, TC001
from nodes.simple import MixNode  # noqa: E402
from notebooks.notebook_support import get_context, get_nodes  # noqa: E402
from nodes.actions import ActionNode  # noqa: E402

if TYPE_CHECKING:
    from nodes.actions.shift import ShiftAmount
    from nodes.context import Context
    from nodes.node import Node
    from params import Parameter


yaml = YAML()


@dataclass
class OptimizeParameterEntry:
    id: str
    x0: float
    bounds: tuple[float, float]
    xstep: float
    set_value: Callable
    finalize: Callable | None


class OptimizeParameter(ABC):
    """Base class for parameter optimization."""

    def __init__(self, action: ActionNode):
        self.action = action
        self.entries: list[OptimizeParameterEntry] = []

    @abstractmethod
    def restore(self):
        """Restore parameters to original values."""
        pass

    def reset(self):
        self.entries = []

    @abstractmethod
    def configure_for_optimization(self, start_year: int, end_year: int):
        """Configure optimization parameters for the given year range."""
        pass

    @abstractmethod
    def to_yaml_dict(self):
        """Convert optimized values to YAML-serializable format."""
        pass

    @abstractmethod
    def save(self, param_cfg: dict):
        """Save optimized values to parameter config."""
        pass


class ShiftOptimizeParameter(OptimizeParameter):
    """Original implementation for ShiftAction parameters."""

    def __init__(self, action: ActionNode, param: Parameter):
        super().__init__(action)
        self.param = param
        self.original_value = param.value
        self.value = param.value
        if isinstance(self.value, BaseModel):
            param.value = self.value = self.value.model_copy(deep=True)

    def restore(self):
        self.param.value = self.original_value

    def configure_for_optimization(self, start_year: int, end_year: int):
        """Configure for shift optimization."""
        self.configure_for_shift(start_year, end_year)

    def set_source_value(self, start: ShiftAmount | None, end: ShiftAmount, new_val: float):
        if start is not None:
            start.source_amount = new_val
        end.source_amount = new_val

    def set_dest_value(self, start: ShiftAmount | None, end: ShiftAmount, idx: int, new_val: float):
        if start is not None:
            start.dest_amounts[idx] = new_val
        end.dest_amounts[idx] = new_val

    def set_last_dest_value(self, start: ShiftAmount | None, end: ShiftAmount, idx: int):
        others = sum([x for i, x in enumerate(end.dest_amounts) if i != idx])
        self.set_dest_value(start, end, idx, 100 - others)

    def configure_for_shift(self, start_year: int, end_year: int):
        value = self.value
        assert isinstance(value, ShiftParameterValue)

        self.reset()

        # remove all the values after our start year
        for eidx, entry in enumerate(value.root):
            entry.amounts = list(sorted([a for a in entry.amounts if a.year <= start_year], key=lambda x: x.year))
            start = entry.amounts[-1]
            if start.year != start_year:
                start = entry.amounts[-1].model_copy()
                start.year = start_year
                entry.amounts.append(start)
            end = start.model_copy(update=dict(year=end_year, deep=True))
            entry.amounts.append(end)

            x0 = start.source_amount
            if start.source_amount < 0:
                bounds = (-100, 0)
            else:
                bounds = (0, 100)

            id_prefix = '%d+%d-%d' % (start.year, end_year, eidx)

            # First the source values
            e = OptimizeParameterEntry(
                id="%s-source" % (id_prefix),
                x0=x0,
                bounds=bounds,
                xstep=0.001,
                set_value=partial(self.set_source_value, start, end),
                finalize=None,
            )
            self.entries.append(e)

            # If only one dest, no reason to optimize it
            if len(start.dest_amounts) == 1:
                start.dest_amounts = [100]
                end.dest_amounts = [100]
                continue

            dests = start.dest_amounts[:-1]
            x0 = 100.0 / len(start.dest_amounts)
            for idx, amount in enumerate(dests):
                # Set the finalizer for the penultimate dest param
                if idx == len(dests) - 1:
                    finalize = partial(self.set_last_dest_value, start, end, idx + 1)
                else:
                    finalize = None
                e = OptimizeParameterEntry(
                    id="%s-dest-%d" % (id_prefix, idx),
                    x0=x0,
                    bounds=(0, 100),
                    xstep=0.01,
                    set_value=partial(self.set_dest_value, start, end, idx),
                    finalize=finalize,
                )
                self.entries.append(e)

    def to_yaml_dict(self):
        out: list[dict] = self.value.model_dump(exclude_none=True, exclude_unset=True)

        def format_num(val: float, prec: int) -> int | float:
            val = round(float(val), prec)
            n = float(int(val))
            if math.isclose(val, n):
                return int(n)
            return val

        for entry in out:
            amounts = entry["amounts"]
            for idx, amt in enumerate(list(amounts)):
                src = round(amt['source_amount'], 3)
                amt['source_amount'] = format_num(src, prec=3)
                dests = []
                for x in amt['dest_amounts']:
                    if isinstance(x, int):
                        dests.append(x)
                        continue
                    if not isinstance(x, float):
                        print(out)
                        raise Exception()
                    dests.append(format_num(x, prec=1))
                amt['dest_amounts'] = dests
                m = CommentedMap(amt)
                m.fa.set_flow_style()
                amounts[idx] = m
        return out

    def to_yaml_string(self):
        string_stream = StringIO()
        yaml.dump(self.to_yaml_dict(), string_stream)
        return string_stream.getvalue()

    def save(self, param_cfg: dict):
        for pc in param_cfg:
            if pc["id"] == "shift":
                break
        else:
            raise Exception("Action %s does not have 'shift' param" % self.action.id)
        pc["value"] = self.to_yaml_dict()


class DynamicNumberParametersOptimizer(OptimizeParameter):
    """Automatically discovers and optimizes NumberParameters in an action."""

    def __init__(self, action: ActionNode, global_config: dict = None):
        super().__init__(action)
        self.global_config = global_config or {}
        self.original_values = {}
        self.parameters = {}

        # Get parameters the SAME way ShiftOptimizeParameter does
        for param_name in action.parameters.keys():
            if param_name == 'enabled':
                continue

            # Use get_parameter() method like ShiftOptimizeParameter
            try:
                param = action.get_parameter(param_name)  # This gets the LIVE parameter
                if hasattr(param, 'value') and isinstance(param.value, (int, float)):
                    self.parameters[param_name] = param
                    self.original_values[param_name] = param.value
                    print(f"Found parameter {param_name} via get_parameter(): {param.value}")
                    print(f"  Parameter object ID: {id(param)}")
            except:
                # Fallback to the old method
                param = action.parameters[param_name]
                if hasattr(param, 'value') and isinstance(param.value, (int, float)):
                    self.parameters[param_name] = param
                    self.original_values[param_name] = param.value
                    print(f"Found parameter {param_name} via parameters dict: {param.value}")
                    print(f"  Parameter object ID: {id(param)}")

    def restore(self):
        """Restore all parameters to original values."""
        for param_name, original_value in self.original_values.items():
            self.parameters[param_name].value = original_value

    def configure_for_optimization(self, start_year: int, end_year: int):
        """Configure optimization entries for all discovered NumberParameters."""
        self.reset()

        for param_name, param in self.parameters.items():
            current_value = float(param.value)

            # Use parameter's own bounds if available
            min_val = getattr(param, 'min_value', None)
            max_val = getattr(param, 'max_value', None)

            if min_val is not None and max_val is not None:
                bounds = (float(min_val), float(max_val))
            # Fallback to global config or sensible defaults
            elif current_value > 0:
                bounds = self.global_config.get('default_bounds', (0.1, current_value * 5))
            else:
                bounds = self.global_config.get('default_bounds', (-abs(current_value) * 5, abs(current_value) * 5))

            # Calculate step size as a fraction of the range
            step_range = bounds[1] - bounds[0]
            default_xstep = max(step_range / 20, 0.02)

            entry = OptimizeParameterEntry(
                id=f"{self.action.id}-{param_name}",
                x0=current_value,
                bounds=bounds,
                xstep=self.global_config.get('default_xstep', default_xstep),
                set_value=partial(self.set_parameter_value, param_name),
                finalize=None,
            )
            self.entries.append(entry)

    def set_parameter_value(self, param_name: str, new_val: float):
        """Set a parameter to a new value."""
        old_val = self.parameters[param_name].value
        self.parameters[param_name].value = float(new_val)
        print(f"SET {param_name}: {old_val} -> {new_val} (id: {id(self.parameters[param_name])})")

        # Verify the change took effect
        actual_val = self.parameters[param_name].value
        if abs(actual_val - new_val) > 1e-10:
            print(f"WARNING: Parameter {param_name} didn't change! Expected {new_val}, got {actual_val}")

    def to_yaml_dict(self):
        """Convert optimized parameter values to YAML format."""
        return {
            param_name: float(param.value)
            for param_name, param in self.parameters.items()
        }

    def save(self, param_cfg: dict):
        """Save optimized values to parameter config."""
        # Assuming param_cfg is a list of parameter dictionaries
        params_by_id = {p["id"]: p for p in param_cfg if isinstance(p, dict) and "id" in p}

        for param_name, param in self.parameters.items():
            if param_name in params_by_id:
                params_by_id[param_name]["value"] = float(param.value)


class OptimizeParameterSet:
    def __init__(self):
        self.params: list[OptimizeParameter] = []
        self.frozen = False

    def add(self, optp: OptimizeParameter):
        assert not self.frozen
        self.params.append(optp)

    def restore(self):
        for param in self.params:
            param.restore()

    def freeze(self):
        self.frozen = True

    @property
    def x0(self) -> tuple[float, ...]:
        return tuple(e.x0 for param in self.params for e in param.entries)

    @property
    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        lower = tuple(e.bounds[0] for param in self.params for e in param.entries)
        upper = tuple(e.bounds[1] for param in self.params for e in param.entries)
        return (lower, upper)

    @property
    def value_setters(self) -> tuple[Callable, ...]:
        return tuple(e.set_value for param in self.params for e in param.entries)

    @property
    def xstep(self) -> tuple[float, ...]:
        return tuple(e.xstep for param in self.params for e in param.entries)

    @property
    def finalizers(self) -> tuple[Callable, ...]:
        return tuple(e.finalize for param in self.params for e in param.entries if e.finalize is not None)

    def set_values(self, vals: np.ndarray):
        for val, set_value in zip(vals, self.value_setters, strict=False):
            set_value(float(val))
        for finalize in self.finalizers:
            finalize()

    def print(self):
        from rich.console import Console
        from rich.table import Table

        table = Table()
        for col in ("Action", "Param", "x0", "bounds", "step"):
            table.add_column(col)
        for param in self.params:
            for e in param.entries:
                table.add_row(param.action.id, e.id, str(e.x0), str(e.bounds), str(e.xstep))
        console = Console()
        console.print(table)

    def save_to_yaml(self, instance: Instance):
        assert instance.yaml_file_path
        cfg = yaml.load(open(instance.yaml_file_path, "r", encoding="utf8"))
        main = cfg
        if "instance" in cfg:
            main = cfg["instance"]
        acts = main["actions"]
        acts_by_id = {act["id"]: act for act in acts}

        for param in self.params:
            act_cfg = acts_by_id[param.action.id]
            param_cfg = act_cfg["params"]
            print(param_cfg)
            param.save(param_cfg)
            print(param_cfg)

        with open(instance.yaml_file_path, "w", encoding="utf8") as f:  # noqa: PTH123
            yaml.dump(cfg, f)

    def print_params(self):
        from rich.console import Console

        out = []
        for param in self.params:
            if isinstance(param, ShiftOptimizeParameter):
                d = param.to_yaml_dict()
                out.append(dict(
                    id=param.action.id,
                    params=[dict(id='shift', value=d)]
                ))
            elif isinstance(param, DynamicNumberParametersOptimizer):
                d = param.to_yaml_dict()
                param_list = [dict(id=k, value=v) for k, v in d.items()]
                out.append(dict(
                    id=param.action.id,
                    params=param_list
                ))

        string_stream = StringIO()
        yaml.dump(out, string_stream)
        s = string_stream.getvalue()
        console = Console()
        syntax = Syntax(s, 'yaml')
        console.print(syntax)


class Optimizer:
    def __init__(
            self, context: Context, outcome_node: Node, goal_df: ppl.PathsDataFrame | None,
            action_nodes: list[ActionNode], optimization_config: dict = None
        ):
        self.context = context
        self.outcome_node = outcome_node

        outcome_df: ppl.PathsDataFrame = outcome_node.compute()
        if goal_df is None:
            goal_df = outcome_df.with_columns(pl.col(VALUE_COLUMN) * 0.0).drop(FORECAST_COLUMN)
        outcome_cols = outcome_df.paths.to_wide(only_category_names=True).metric_cols
        goal_df = goal_df.paths.to_wide(only_category_names=True)
        goal_df = goal_df.select_metrics(outcome_cols)
        self.goal_df = goal_df
        self.action_nodes = action_nodes
        self.outcome_cols = outcome_cols
        self.optimization_config = optimization_config or {}

    def compute_and_compare(
        self, x: np.ndarray, year: int, goal: np.ndarray, params: OptimizeParameterSet  # pyright: ignore[reportMissingTypeArgument]
    ):
        # print('before setting')
        # params.print_params()
        # print(self.outcome_node.hasher)
        params.set_values(x)
        # print('after settinig')
        self.outcome_node.hasher.mark_modified()
        for action in self.action_nodes:
            action.hasher.mark_modified()
        # print(self.outcome_node.hasher)

        import gc
        gc.collect()

        print('petrol: ', id(params.params[0].parameters['petrol']))
        print('petrol in action', id(self.action_nodes[0].parameters['petrol']))
        # params.print_params()

        df = (
            self.outcome_node.compute().paths.to_wide(only_category_names=True).lazy()
            .drop(FORECAST_COLUMN).filter(pl.col(YEAR_COLUMN) == year)
            .drop(YEAR_COLUMN)
            .collect()
        )
        print('hydrogen:', self.action_nodes[0].get_parameter_value_float('hydrogen'))
        # print(df)
        # print(self.goal_df)
        assert df.columns == self.outcome_cols
        outcome = df.to_numpy()[0]
        print(outcome)
        print(goal)
        diff = outcome - goal
        # print(diff)
        # If one share goes below zero, we punish for 10x more severely
        mults = [10 if x < 0 else 1 for x in outcome]
        return np.abs(diff) * mults

    def run_for_years(self, params: OptimizeParameterSet, start_year: int, target_year: int):
        df = self.goal_df.filter(pl.col(YEAR_COLUMN) == target_year).drop(YEAR_COLUMN)
        goal = df.to_numpy()[0]

        for opt in params.params:
            opt.configure_for_optimization(start_year, target_year)

        params.print()
        print('cache skip status: ', self.context.skip_cache)

        self.context.model_end_year = target_year
        with self.context.run():
            res = optimize.least_squares(
                self.compute_and_compare,
                params.x0,
                bounds=params.bounds,
                diff_step=params.xstep,
                max_nfev=500,
                method="trf",
                kwargs=dict(
                    goal=goal,
                    year=target_year,
                    params=params,
                ),
            )
            print(res)
            params.set_values(res.x)

    def run(self) -> OptimizeParameterSet:
        ctx = self.context
        params = OptimizeParameterSet()
        path_nodes = set()

        for act in self.action_nodes:
            all_paths = list(nx.all_simple_paths(
                ctx.node_graph, source=act.id, target=self.outcome_node.id
            ))
            assert all_paths
            for path in all_paths:
                path_nodes.update(path)

            # Determine which type of optimizer to create
            if isinstance(act, ShiftAction):
                param = act.get_parameter("shift")
                opt = ShiftOptimizeParameter(act, param)
            else:
                # Use dynamic NumberParameters optimizer
                global_config = self.optimization_config.get('global', {})
                opt = DynamicNumberParametersOptimizer(act, global_config)
                # Skip if no optimizable parameters found
                if not opt.parameters:
                    print(f"Warning: No NumberParameters found in action {act.id}")
                    continue

            params.add(opt)

            # for act in self.action_nodes:
            #     if isinstance(act, ShiftAction):
            #         param = act.get_parameter("shift")  # Gets the LIVE parameter
            #         opt = ShiftOptimizeParameter(act, param)
            #     else:
            #         # Let the optimizer discover parameters using get_parameter()
            #         global_config = self.optimization_config.get('global', {})
            #         opt = DynamicNumberParametersOptimizer(act, global_config)
            #         if not opt.parameters:
            #             print(f"Warning: No NumberParameters found in action {act.id}")
            #             continue
                
            #     params.add(opt)


        params.freeze()

        ctx.skip_cache = True
        for node_id in list(path_nodes):
            ctx.nodes[node_id].disable_cache = True

        if isinstance(self.outcome_node, MixNode):
            self.outcome_node.skip_normalize = True

        hist_df = self.outcome_node.get_output_pl().filter(~pl.col(FORECAST_COLUMN))
        last_hist_year: int = list(hist_df[YEAR_COLUMN])[-1]
        years = sorted(self.goal_df[YEAR_COLUMN])
        years = [2030, 2040, 2050]
        # Start with the first forecast year
        range_start_year = last_hist_year + 1
        # previous_end_year = ctx.model_end_year
        try:
            for goal_year in years:
                print('%d -> %d' % (range_start_year, goal_year))
                self.run_for_years(params, range_start_year, goal_year)
                params.print_params()
                range_start_year = goal_year + 1
        except:
            raise
        else:
            out_df = self.outcome_node.compute()
            unit = self.outcome_node.get_default_output_metric().unit
            df = (
                out_df
                .paths.to_wide(only_category_names=True)
                .filter(pl.col(YEAR_COLUMN).is_in(self.goal_df[YEAR_COLUMN]))
                .drop(FORECAST_COLUMN)
                .with_columns(pl.lit('After').alias('Optimize'))
            )
            gdf = self.goal_df.with_columns(pl.lit('Before').alias('Optimize'))
            df = ppl.to_ppdf(pl.concat([gdf, df]), meta=df.get_meta())
            df = df.with_columns(
                pl.sum_horizontal(df.metric_cols).alias("Sum")
            ).set_unit('Sum', unit).select_metrics([*self.outcome_cols, 'Sum'])
            print(df)
        finally:
            params.restore()
        return params


def main():
    # # For ShiftAction (existing functionality)
    # optimizer = Optimizer(
    #     context=context,
    #     outcome_node=outcome_node,
    #     goal_df=goal_df,
    #     action_nodes=[shift_action1, shift_action2]
    # )

    # For NumberParameters with global config
    optimization_config = {
        'global': {
            'default_xstep': 0.001,
            # default_bounds only used if parameter has no min_value/max_value
            'default_bounds': (0.1, 10.0)
        }
    }

    context = get_context('potsdam-gpc')
    nodes = get_nodes('potsdam-gpc')
    node = nodes['emission_difference']
    action = nodes['multiplier']
    assert isinstance(action, ActionNode)

    optimizer = Optimizer(
        context=context,
        outcome_node=node,
        goal_df=None,
        action_nodes=[action],
        optimization_config=optimization_config
    )

    # # Mixed case (both ShiftActions and NumberParameter actions)
    # optimizer = Optimizer(
    #     context=context,
    #     outcome_node=outcome_node,
    #     goal_df=goal_df,
    #     action_nodes=[shift_action, number_param_action],
    #     optimization_config=optimization_config
    # )

    # Run optimization
    optimized_params = optimizer.run()
    # optimized_params.print_params()

if __name__ == '__main__':
    main()
