from __future__ import annotations
from contextlib import contextmanager
from io import StringIO
import math
from typing import TYPE_CHECKING
import polars as pl

from rich import print
from rich.syntax import Syntax
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.scalarfloat import ScalarFloat

from dataclasses import dataclass
import numpy as np
from functools import cached_property, partial, wraps
from typing import Callable, Tuple
import networkx as nx
from pydantic import BaseModel
from scipy import optimize

from nodes.constants import FORECAST_COLUMN, YEAR_COLUMN
from nodes.instance import Instance
from nodes.simple import MixNode
from nodes.actions.shift import ShiftAction, ShiftParameterValue
from common import polars as ppl

if TYPE_CHECKING:
    from nodes import Node
    from nodes.context import Context
    from nodes.actions.shift import ShiftAmount
    from params import Parameter
    from nodes.actions import ActionNode


yaml = YAML()


@dataclass
class OptimizeParameterEntry:
    id: str
    x0: float
    bounds: Tuple[float, float]
    xstep: float
    set_value: Callable
    finalize: Callable | None


class OptimizeParameter:
    entries: list[OptimizeParameterEntry]

    def __init__(self, action: ActionNode, param: Parameter):
        self.action = action
        self.param = param
        self.original_value = param.value
        self.value = param.value
        if isinstance(self.value, BaseModel):
            param.value = self.value = self.value.model_copy(deep=True)
        self.reset()

    def restore(self):
        self.param.value = self.original_value

    def reset(self):
        self.entries = []

    def set_source_value(self, start: ShiftAmount | None, end: ShiftAmount, new_val: float):
        #new_val = round(float(new_val), 3)
        if start is not None:
            start.source_amount = new_val
            # print('%s: set source %d from %f to %f' % (self.action.id, id(start), start.source_amount, new_val))
        end.source_amount = new_val

    def set_dest_value(self, start: ShiftAmount | None, end: ShiftAmount, idx: int, new_val: float):
        #new_val = round(float(new_val), 1)
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
                #x0 = -0.1
            else:
                bounds = (0, 100)
                #x0 = 0.1

            id_prefix = '%d+%d-%d' % (start.year, end.year, eidx)

            # First the source values
            e = OptimizeParameterEntry(
                id="%s-source" % (id_prefix),
                # x0=start.source_amount,
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
        out: list[dict] = self.value.model_dump(exclude_none=True, exclude_unset=True)  # pyright: ignore

        def format_num(val: float, prec: int):
            val = round(float(val), prec)
            n = float(int(val))
            if math.isclose(val, n):
                return int(n)
            return val

        for entry in out:
            amounts = entry["amounts"]  # pyright: ignore
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
        from io import StringIO

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
    def x0(self) -> Tuple[float, ...]:
        return tuple(e.x0 for param in self.params for e in param.entries)

    @property
    def bounds(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        lower = tuple(e.bounds[0] for param in self.params for e in param.entries)
        upper = tuple(e.bounds[1] for param in self.params for e in param.entries)
        return (lower, upper)

    @property
    def value_setters(self) -> Tuple[Callable, ...]:
        return tuple(e.set_value for param in self.params for e in param.entries)

    @property
    def xstep(self) -> Tuple[float, ...]:
        return tuple(e.xstep for param in self.params for e in param.entries)

    @property
    def finalizers(self) -> Tuple[Callable, ...]:
        return tuple(e.finalize for param in self.params for e in param.entries if e.finalize is not None)

    def set_values(self, vals: np.ndarray):
        for val, set_value in zip(vals, self.value_setters):  # pyright: ignore
            set_value(float(val))
        for finalize in self.finalizers:  # pyright: ignore
            finalize()

    def print(self):
        from rich.table import Table
        from rich.console import Console

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

        with open(instance.yaml_file_path, "w", encoding="utf8") as f:
            yaml.dump(cfg, f)

    def print_params(self):
        from rich.console import Console

        out = []
        for param in self.params:
            d = param.to_yaml_dict()
            out.append(dict(
                id=param.action.id,
                params=[dict(id='shift', value=d)]
            ))

        string_stream = StringIO()
        yaml.dump(out, string_stream)
        s = string_stream.getvalue()
        console = Console()
        syntax = Syntax(s, 'yaml')
        console.print(syntax)


class Optimizer:
    def __init__(
            self, context: Context, outcome_node: Node, goal_df: ppl.PathsDataFrame,
            action_nodes: list[ActionNode]
        ):
        self.context = context
        self.outcome_node = outcome_node

        outcome_df = outcome_node.compute()
        outcome_cols = outcome_df.paths.to_wide(only_category_names=True).metric_cols
        goal_df = goal_df.paths.to_wide(only_category_names=True)
        goal_df = goal_df.select_metrics(outcome_cols)
        self.goal_df = goal_df
        self.action_nodes = action_nodes
        self.outcome_cols = outcome_cols

    def compute_and_compare(
        self, x: np.ndarray, year: int, goal: np.ndarray, params: OptimizeParameterSet
    ):
        params.set_values(x)

        df = (
            self.outcome_node.compute().paths.to_wide(only_category_names=True).lazy()
            .drop(FORECAST_COLUMN).filter(pl.col(YEAR_COLUMN) == year)
            .drop(YEAR_COLUMN)
            .collect()
        )
        assert df.columns == self.outcome_cols
        outcome = df.to_numpy()[0]
        diff = outcome - goal
        # If one share goes below zero, we punish for 10x more severely
        mults = [10 if x < 0 else 1 for x in outcome]
        return np.abs(diff) * mults

    def run_for_years(self, params: OptimizeParameterSet, start_year: int, target_year: int):
        df = self.goal_df.filter(pl.col(YEAR_COLUMN) == target_year).drop(YEAR_COLUMN)
        goal = df.to_numpy()[0]

        for opt in params.params:
            opt.configure_for_shift(start_year, target_year)

        params.print()

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
            # df = outcome_node.compute().paths.to_wide(only_category_names=True)\
            #    .drop(FORECAST_COLUMN).filter(pl.col(YEAR_COLUMN) == target_year)\
            #    .drop(YEAR_COLUMN)
            # print(df)
            # print(x0)
            # print([round(x, 2) for x in res.x])
            params.set_values(res.x)

    def run(self) -> OptimizeParameterSet:
        ctx = self.context
        params = OptimizeParameterSet()
        path_nodes = set()
        for act in self.action_nodes:
            all_paths = list(nx.all_simple_paths(
                ctx.node_graph, source=act.id, target=self.outcome_node.id
            ))
            assert len(all_paths)
            for path in all_paths:
                path_nodes.update(path)

            assert isinstance(act, ShiftAction)
            param = act.get_parameter("shift")
            opt = OptimizeParameter(act, param)
            params.add(opt)

        params.freeze()

        for node_id in list(path_nodes):
            ctx.nodes[node_id].disable_cache = True

        if isinstance(self.outcome_node, MixNode):
            self.outcome_node.skip_normalize = True

        hist_df = self.outcome_node.get_output_pl().filter(~pl.col(FORECAST_COLUMN))
        last_hist_year: int = list(hist_df[YEAR_COLUMN])[-1]
        years = sorted(self.goal_df[YEAR_COLUMN])
        # Start with the first forecast year
        range_start_year = last_hist_year + 1
        previous_end_year = ctx.model_end_year
        try:
            for goal_year in years:
                print('%d -> %d' % (range_start_year, goal_year))
                # with cProfile.Profile() as pr:
                self.run_for_years(params, range_start_year, goal_year)
                params.print_params()
                #    pr.dump_stats('/tmp/opt.profile')
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
