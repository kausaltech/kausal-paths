#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

from contextlib import ExitStack

from kausal_common.development.django import init_django

from kausal_common.logging.init import is_pretty_terminal

from common.utils import install_node_error_handler

init_django()

import argparse
import cProfile
import math
import os
import random
import time
from math import log10
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import polars as pl
import sentry_sdk
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.table import Table
from sentry_sdk import start_span, start_transaction

from kausal_common.debugging.perf import PerfCounter

from nodes.actions.action import ActionNode
from nodes.constants import IMPACT_COLUMN, IMPACT_GROUP, YEAR_COLUMN
from nodes.excel_results import InstanceResultExcel
from nodes.instance_loader import InstanceLoader

if TYPE_CHECKING:
    from nodes.models import InstanceConfig
    from nodes.units import Quantity
    from nodes.visualizations import VisualizationEntryType


load_dotenv()

console = Console()

parser = argparse.ArgumentParser(description='Execute the computational graph')
parser.add_argument('-i', '--instance', type=str, help='instance identifier')
parser.add_argument('-c', '--config', type=str, help='config yaml file')
parser.add_argument('--baseline', action='store_true', help='generate baseline scenario values')
parser.add_argument('--scenario', type=str, help='select scenario')
parser.add_argument('--param', action='append', type=str, help='set a parameter')
parser.add_argument('--list-params', action='store_true', help='list parameters')
parser.add_argument('--debug-nodes', type=str, nargs='+', help='enable debug messages for nodes')
parser.add_argument('--check', action='store_true', help='perform sanity checking')
parser.add_argument('--skip-cache', action='store_true', help='skip caching altogether')
parser.add_argument('-n', '--node', type=str, nargs='+', help='compute node')
parser.add_argument('--filter', type=str, nargs='+', help='filter node output')
parser.add_argument('--normalize', type=str, metavar='NODE', help='normalize by other node')
parser.add_argument('--pull-datasets', action='store_true', help='refresh all datasets')
parser.add_argument('--print-graph', action='store_true', help='print the graph')
parser.add_argument('--update-instance', action='store_true', help='update an existing InstanceConfig instance')
parser.add_argument('--update-nodes', action='store_true', help='update existing NodeConfig instances')
parser.add_argument('--overwrite', action='store_true', help='Overwrite contents in the database')
parser.add_argument('--skip-descriptions', action='store_true', help='skip description updates in the database')
parser.add_argument('--delete-stale-nodes', action='store_true', help='delete NodeConfig instances that no longer exist')
parser.add_argument('--validate', action='store_true', help='print validation results and exit')
parser.add_argument(
    '--validation-level',
    choices=['error', 'warning', 'info'],
    default='warning',
    help='minimum severity to show in validation output',
)
parser.add_argument('--validation-valid-also', action='store_true', help='include validations that pass without errors')
parser.add_argument(
    '--print-impact-overviews', action='store_true', help='calculate and print impact overviews (previously action efficiencies)'
)
parser.add_argument(
    '--print-visualizations-meta',
    action='store_true',
    help='print node visualization config (structure, referenced nodes, dimensions)',
)
parser.add_argument(
    '--print-visualizations-data', action='store_true', help='compute and print the actual data for each node visualization'
)
parser.add_argument('--show-perf', action='store_true', help='show performance info')
parser.add_argument('--profile', action='store_true', help='profile computation performance')
parser.add_argument('--disable-ext-cache', action='store_true', help='disable external cache')
parser.add_argument('--cache-benchmark', action='store_true', help='Perform cache benchmarks')
parser.add_argument('--generate-result-excel', type=str, metavar='FILENAME', help='Create an Excel file from model outputs')
parser.add_argument('-q', '--quiet', action='store_true', help='Sshhhh!')
parser.add_argument(
    '--format',
    choices=['long', 'wide'],
    default='long',
    help="""
                   Excel output format (default: long). Originally, long was used for climate city contracts,
                   wide for GPC datasets.
                   """,
)

# parser.add_argument('--sync', action='store_true', help='sync db to node contents')
args = parser.parse_args()

if (args.instance and args.config) or (not args.instance and not args.config):
    print('Specify either "--instance" or "--config"')
    exit(1)

if args.disable_ext_cache:
    os.environ['REDIS_URL'] = ''

_instance_obj: InstanceConfig | None = None


@overload
def get_ic(instance_id: str, /, *, required: Literal[True] = True) -> InstanceConfig: ...


@overload
def get_ic(instance_id: str, /, *, required: bool = False) -> InstanceConfig | None: ...


def get_ic(instance_id: str, /, *, required: bool = False) -> InstanceConfig | None:
    from nodes.models import InstanceConfig

    global _instance_obj  # noqa: PLW0603

    if _instance_obj is not None:
        return _instance_obj
    ic = InstanceConfig.objects.filter(identifier=instance_id).first()
    if required and ic is None:
        raise Exception("InstanceConfig with identifier '%s' not found" % args.instance)
    _instance_obj = ic
    return ic


install_node_error_handler()

stack = ExitStack()
root_span = stack.enter_context(start_transaction(name='load-nodes', op='function'))

if args.instance:
    with start_span(name='django-init', op='init') as span:
        span.set_data('instance_id', args.instance)
        instance_obj = get_ic(args.instance, required=True)
        instance = instance_obj.get_instance()
        context = instance.context
else:
    with start_span(name='yaml-init', op='init') as span:
        span.set_data('config_id', args.config)
        yaml_path = Path(args.config).resolve()
        loader = InstanceLoader.from_yaml(yaml_path)
        context = loader.context
        instance = loader.instance


def print_db_datasets():
    from nodes.datasets import DBDataset, DVCDataset

    if TYPE_CHECKING:
        from nodes.node import Node

    if not is_pretty_terminal():
        return

    db_datasets: dict[str, tuple[DBDataset, list[Node]]] = {}
    dvc_datasets: dict[str, tuple[DVCDataset, list[Node]]] = {}
    for node in context.nodes.values():
        for ds in node.input_dataset_instances:
            if isinstance(ds, DBDataset):
                db_datasets.setdefault(ds.id, (ds, []))[1].append(node)
            elif isinstance(ds, DVCDataset):
                dvc_datasets.setdefault(ds.id, (ds, []))[1].append(node)
    print('Datasets in use:')
    if context.dataset_repo is not None:
        commit = context.dataset_repo.target_commit_id
        print(f'Commit: {commit}')
    table = Table()
    table.add_column('Source')
    table.add_column('Dataset ID')
    table.add_column('Modified at')
    table.add_column('Used by nodes')
    node_colors = ['white', 'bright_white']
    node_color = node_colors[0]
    for ds_id, (ds, nodes) in db_datasets.items():
        ds_obj = ds.db_dataset_obj
        table.add_row(
            '[green]DB[/green]',
            '[%s]%s[/%s]' % (node_color, ds_id, node_color),
            ds_obj.last_modified_at.strftime('%Y-%m-%d %H:%M:%S') if ds_obj else '',
            ', '.join('[%s]%s[/%s]' % (node_color, node.id, node_color) for node in nodes),
        )
        node_color = node_colors[(node_colors.index(node_color) + 1) % len(node_colors)]
    for ds_id, (_, nodes) in dvc_datasets.items():
        dvc_ds = context.load_dvc_dataset(ds_id)
        table.add_row(
            '[blue]DVC[/blue]',
            '[%s]%s[/%s]' % (node_color, ds_id, node_color),
            dvc_ds.modified_at.strftime('%Y-%m-%d %H:%M:%S') if dvc_ds.modified_at else '',
            ', '.join('[%s]%s[/%s]' % (node_color, node.id, node_color) for node in nodes),
        )
        node_color = node_colors[(node_colors.index(node_color) + 1) % len(node_colors)]
    console.print(table)


if not args.quiet:
    print_db_datasets()

if args.pull_datasets:
    with start_span(name='pull-datasets', op='init') as span:
        context.pull_datasets()

if args.validate:
    nes = context.node_explanation_system
    if nes is None:
        print('No validation system available.')
        exit(1)
    messages = nes.show_messages(level=args.validation_level, valid_also=args.validation_valid_also)
    if not messages:
        print(f"No validation messages at level '{args.validation_level}' or worse.")
    else:
        table = Table(title='Validation results')
        table.add_column('Node')
        table.add_column('Level')
        table.add_column('Message')
        for node_id, node_messages in messages.items():
            for msg in node_messages:
                table.add_row(node_id, msg.level, msg.message)
        console.print(table)
    exit(1 if nes.has_errors() else 0)

if args.check:
    context.check_mode = True

if args.show_perf:
    context.perf_context.enabled = True


def cache_benchmark():
    from kausal_common.debugging.perf import PerfCounter

    from common.cache import CacheKind

    pc = PerfCounter('benchmark init')
    nodes = list(context.nodes.values())
    test_dfs = []
    rand_state = random.getstate()
    random.seed(0)
    pc.display('computing node output')
    for n in random.sample(nodes, k=10):
        df = n.get_output_pl()
        test_dfs.append(df)
        pc.display('%s done' % n.id)
    pc.display('node output computation done')

    cache = context.cache
    cache.set_lru_size(256 * 1024 * 1024)
    cache.clear()
    pc.finish()

    allowed_kinds = set(cache.allowed_kinds)

    for kind in (CacheKind.RUN, CacheKind.LOCAL, CacheKind.EXT):
        nr_rounds = 10000
        pc = PerfCounter('%s Cache' % kind.name)
        cache.set_allowed_cache_kinds({kind})
        with context.run():
            pc.display('Measuring cache misses for %d keys' % nr_rounds)
            for i in range(nr_rounds):
                key = 'key-%d' % i
                res = cache.get(key)
                assert not res.is_hit
            pc.display('Miss benchmark done', show_time_to_last=True)

            pc.display('Setting %d keys' % nr_rounds)
            for i in range(nr_rounds):
                key = 'key-%d' % i
                cache.set(key, test_dfs[i % len(test_dfs)], expiry=30)
            pc.display('Set done', show_time_to_last=True)
            assert cache.run is not None
            cache.run.flush()
            pc.display('Flush done', show_time_to_last=True)

            pc.display('Reading %d keys' % nr_rounds)
            for i in range(nr_rounds):
                key = 'key-%d' % i
                res = cache.get(key, expiry=30)
                assert res.is_hit
            pc.display('Read done', show_time_to_last=True)
        cache.clear()
        pc.display('Run finished', show_time_to_last=True)

    random.setstate(rand_state)
    cache.set_allowed_cache_kinds(allowed_kinds)


if args.cache_benchmark:
    cache_benchmark()
    exit()

profile: cProfile.Profile | None
if args.profile:
    profile = cProfile.Profile()
else:
    profile = None


if args.print_graph:
    context.print_graph(include_datasets=True)

if args.print_visualizations_meta:
    from nodes.visualizations import VisualizationGroup, VisualizationNodeOutput

    def _add_entries(entries_list: list, prefix: str = '', *, table: Table) -> None:  # type: ignore[type-arg]
        for entry in entries_list:
            if isinstance(entry, VisualizationNodeOutput):
                dims = (
                    ', '.join(
                        f'{d.id}{"[" + ",".join(d.categories) + "]" if d.categories else ""}{"(flat)" if d.flatten else ""}'
                        for d in entry.dimensions
                    )
                    or '(none)'
                )
                scenarios = ', '.join(entry.scenarios) if entry.scenarios else '(all)'
                details = f'node_id={entry.node_id}  desired={entry.desired_outcome.value}\ndims: {dims}\nscenarios: {scenarios}'
                if entry.output_metric_id:
                    details += f'\nmetric: {entry.output_metric_id}'
                table.add_row(prefix + entry.id, 'node', details)
            elif isinstance(entry, VisualizationGroup):
                label_str = str(entry.label) if entry.label else '(no label)'
                table.add_row(prefix + entry.id, 'group', f'label={label_str}')
                _add_entries(entry.children, prefix=prefix + '  ', table=table)

    total = 0
    for node in context.nodes.values():
        if node.visualizations is None:
            continue
        entries = node.visualizations.root
        if not entries:
            continue
        total += 1
        table = Table(title=f'Node: [bold]{node.id}[/bold]', show_header=True)
        table.add_column('ID')
        table.add_column('Kind')
        table.add_column('Details')
        _add_entries(entries, table=table)
        console.print(table)

    if total == 0:
        print('No nodes have visualizations configured.')

if args.print_visualizations_data:
    from nodes.constants import YEAR_COLUMN
    from nodes.visualizations import VisualizationGroup, VisualizationNodeOutput

    min_year: int = context.instance.minimum_historical_year
    max_year: int | None = context.instance.maximum_historical_year

    def _print_node_viz_data(root_node_id: str, entry: VisualizationNodeOutput, depth: int = 0) -> None:  # noqa: PLR0912, C901
        indent = '  ' * depth
        label_part = f'  [italic]{entry.label}[/italic]' if entry.label else ''
        rule_text = (
            f'{indent}[cyan]{entry.id}[/cyan]{label_part}'
            f'  root=[bold]{root_node_id}[/bold]  data from=[bold]{entry.node_id}[/bold]'
        )
        console.rule(rule_text)
        ref_node = context.nodes[entry.node_id]

        if entry.scenarios:
            scenarios = [context.scenarios[sid] for sid in entry.scenarios]
        else:
            scenarios = [context.get_default_scenario()]

        # Collect one filtered series per scenario.
        scenario_cols: dict[str, dict[int, float | None]] = {}
        years_union: list[int] = []
        unit_str = ''
        for scenario in scenarios:
            with scenario.override(set_active=True):
                df = entry.get_output(ref_node)
            metric_cols = df.metric_cols
            if not metric_cols:
                continue
            val_col = metric_cols[0]
            if not unit_str:
                unit_str = str(df.get_unit(val_col))
            year_filter = pl.col(YEAR_COLUMN) >= min_year
            if max_year is not None:
                year_filter = year_filter & (pl.col(YEAR_COLUMN) <= max_year)
            # Sum over any remaining dimensions so we always get a scalar per year.
            remaining_dims = df.dim_ids
            for dim_id in remaining_dims:
                df = df.paths.sum_over_dims(dim_id)
            df = df.filter(year_filter).sort(YEAR_COLUMN)
            year_vals: dict[int, float | None] = {int(row[0]): row[1] for row in df.select([YEAR_COLUMN, val_col]).iter_rows()}
            scenario_cols[scenario.id] = year_vals
            for y in year_vals:
                if y not in years_union:
                    years_union.append(y)

        if not scenario_cols:
            print('[red]  No data[/red]')
            return

        years_union.sort()
        table = Table(title=f'unit: {unit_str}', show_lines=False)
        table.add_column('Year', style='dim')
        for scenario in scenarios:
            if scenario.id in scenario_cols:
                table.add_column(scenario.id)
        for year in years_union:
            row_vals: list[str] = [str(year)]
            for scenario in scenarios:
                if scenario.id not in scenario_cols:
                    continue
                v = scenario_cols[scenario.id].get(year)
                row_vals.append(f'{v:.4g}' if v is not None else '—')
            table.add_row(*row_vals)
        console.print(table)

    def _print_viz_entry_data(root_node_id: str, entry: VisualizationEntryType, depth: int = 0) -> None:
        if isinstance(entry, VisualizationNodeOutput):
            _print_node_viz_data(root_node_id, entry, depth=depth)
        elif isinstance(entry, VisualizationGroup):
            label_str = str(entry.label) if entry.label else '(no label)'
            console.rule(f'[yellow]group {entry.id}[/yellow]  {label_str}')
            for child in entry.children:
                _print_viz_entry_data(root_node_id, child, depth=depth + 1)

    total = 0
    with context.run():
        for node in context.nodes.values():
            if node.visualizations is None:
                continue
            entries = node.visualizations.root
            if not entries:
                continue
            total += 1
            console.rule(f'[bold yellow]Node: {node.id}[/bold yellow]', style='yellow')
            for entry in entries:
                _print_viz_entry_data(node.id, entry)

    if total == 0:
        print('No nodes have visualizations configured.')

if args.skip_cache:
    context.skip_cache = True

if args.scenario:
    context.activate_scenario(context.get_scenario(args.scenario))

if args.list_params:
    context.print_all_parameters()

# with root_span.start_child(name='load-dvc-datasets', op='function'):
#    context.load_all_dvc_datasets()

for node_id in args.debug_nodes or []:
    node = context.get_node(node_id)
    node.debug = True

if args.baseline:
    pc = PerfCounter('Baseline')
    if profile is not None:
        profile.enable()
    pc.display('generating baseline values')
    with context.run():
        context.generate_baseline_values()
    pc.display('done')
    if profile is not None:
        profile.disable()
        profile.dump_stats('baseline_profile.out')


def update_instance():
    from django.db import transaction

    from nodes.models import InstanceConfig

    ic = get_ic(instance.id, required=False)
    if ic is None:
        print('Creating instance %s' % instance.id)
        instance_obj = InstanceConfig.create_for_instance(instance)
        # TODO: Create InstanceHostname somewhere?
    else:
        instance_obj = ic

    with transaction.atomic(), instance_obj.enter_instance_context():
        if args.update_instance:
            instance_obj.update_from_instance(instance, overwrite=True)
            instance_obj.save()
        instance_obj.sync_nodes(
            update_existing=args.update_nodes,
            delete_stale=args.delete_stale_nodes,
            overwrite=args.overwrite,
            skip_descriptions=args.skip_descriptions,
        )
        instance_obj.sync_dimensions(update_existing=True, delete_stale=args.delete_stale_nodes)
        instance_obj.refresh_from_db()
        instance_obj.create_default_content()
    return instance_obj


if args.check:
    context.check_mode = True
    old_cache_prefix = context.cache.prefix
    context.cache.prefix = old_cache_prefix + '-' + str(time.time())
    for node_id, node in context.nodes.items():
        print('Checking %s' % node_id)
        try:
            node.check()
        except Exception as e:
            print(e)

    context.cache.prefix = old_cache_prefix


if args.update_instance or args.update_nodes:
    try:
        instance_obj = update_instance()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise

for param_arg in args.param or []:
    param_id, val = param_arg.split('=')
    context.set_parameter_value(param_id, val)

if args.normalize:
    norm = context.normalizations[args.normalize]
    context.active_normalization = norm

all_filters = []
for line in args.filter or []:
    all_filters += line.split(',')


def generate_result_excel():
    excel_path = Path(args.generate_result_excel)
    existing_wb: Path | None
    if excel_path.exists():
        existing_wb = excel_path
        context.log.info("Excel workbook '%s' exists; opening it as the base" % excel_path)
    else:
        context.log.info("Excel workbook '%s' does not exist; creating new" % excel_path)
        existing_wb = None

    ic = get_ic(instance.id)
    with context.run():
        if not instance.result_excels:
            if ic is None:
                raise Exception("Instance '%s' not found" % instance.id)
            out = InstanceResultExcel.create_for_instance(
                ic=instance_obj, existing_wb=existing_wb, context=context, format=args.format
            )
        else:
            excel_res = instance.result_excels[0]
            excel_res.format = args.format  # Add this line
            out = excel_res.create_result_excel(instance, existing_wb=existing_wb)
    with excel_path.open('wb') as f:
        f.write(out.getvalue())


if args.generate_result_excel:
    with start_span(name='generate-result-excel', op='function'):
        generate_result_excel()

for node_id in args.node or []:
    with start_span(name='print-node-output: %s' % node_id, op='function'):
        with start_span(name='get-node', op='function'):
            node = context.get_node(node_id)
        with start_span(name='run-node', op='function'), context.run(), start_span(name='print-output', op='function'):
            node.print_output(filters=all_filters or None)
            # node.plot_output(filters=all_filters or None)

    if isinstance(node, ActionNode):
        output_nodes = node.output_nodes
        for n in output_nodes:
            print('Impact of %s on OUTPUT node %s' % (node, n))
            node.print_impact(n)

        for n in context.get_outcome_nodes():
            print('Impact of action %s on OUTCOME node %s' % (node, n))
            node.print_impact(n)

        """
        for n in context.nodes.values():
            if n.output_nodes:
                # Not a leaf node
                continue
            if n not in downstream_nodes:
                print("%s has no impact on %s" % (node, n))
                continue
        """


def round_quantity(e: Quantity):
    if abs(e.m) > 10000:
        e = round(e, 1)  # type: ignore
    else:
        if math.isclose(e.m, 0):
            digits = 2
        else:
            digits = int(-log10(abs(e.m))) + 4
        e = round(e, ndigits=digits)  # type: ignore
    return e


if args.print_impact_overviews:

    def print_impact_overviews():  # FIXME Adjust to be functional sith different types of impact overviews
        pc = PerfCounter('Impact overviews')
        for aep in context.impact_overviews:
            cn = aep.cost_node
            if cn:
                cost_id = cn.id
            else:
                cost_id = 'None'
            title = '%s / %s' % (cost_id, aep.effect_node.id)
            pc.display('%s starting' % title)
            table = Table(title=title)
            table.add_column('Action')
            table.add_column('Cumulative efficiency')
            if args.node:
                actions = [context.get_action(node_id) for node_id in args.node]
            else:
                actions = None
            rows = []
            for out in aep.calculate_iter(context, actions=actions):
                action = out.action
                pc.display('%s computed' % action.id)
                # e = out.cumulative_efficiency
                # if e:
                #     e = round_quantity(e)

                rows.append((action.id, None))

            rows = sorted(rows, key=lambda x: x[1].m if x[1] is not None else 1e100)  # pyright: ignore[reportUnknownLambdaType]
            for row in rows:
                table.add_row(row[0], str(row[1]))
            console.print(table)

    def print_impacts():
        pc = PerfCounter('Action impacts')
        for outcome_node in context.get_outcome_nodes():
            title = outcome_node.id
            pc.display('%s starting' % title)
            table = Table(title=title)
            years = []
            table.add_column('Action')
            table.add_column('Impact %s' % context.target_year)
            years.append(context.target_year)
            if context.model_end_year != context.target_year:
                table.add_column('Impact %s' % context.model_end_year)
                years.append(context.model_end_year)
            m = outcome_node.get_default_output_metric()
            rows = []
            for action in context.get_actions():
                df = action.compute_impact(outcome_node)
                if context.active_normalization:
                    _, df = context.active_normalization.normalize_output(outcome_node.get_default_output_metric(), df)
                dims = df.dim_ids
                dims.remove(IMPACT_COLUMN)
                if dims:
                    df = df.paths.sum_over_dims(dims)
                df = df.filter(pl.col(IMPACT_COLUMN).eq(IMPACT_GROUP))
                vals = []
                for year in years:
                    val = df.filter(pl.col(YEAR_COLUMN).eq(year))[m.column_id][0]
                    e = val * df.get_unit(m.column_id)
                    e = round_quantity(e)
                    vals.append(e)
                rows.append([action.id, *vals])

            console = Console()
            rows = sorted(rows, key=lambda x: x[1])  # pyright: ignore[reportUnknownLambdaType]
            for row in rows:
                table.add_row(row[0], *[str(x) for x in row[1:]])
            console.print(table)

    if profile is not None:
        profile.enable()

    with context.run():
        if context.impact_overviews:
            print_impact_overviews()
        else:
            print_impacts()
    if profile is not None:
        profile.disable()
        profile.dump_stats('impact_overview_profile.out')

if False:
    loader.context.dataset_repo.pull_datasets()
    loader.context.print_all_parameters()
    loader.context.generate_baseline_values()
    # for sector in page.get_sectors():
    #    print(sector)

stack.close()
