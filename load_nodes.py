#!/usr/bin/env python3
# ruff: noqa: E402

import os

import django

from kausal_common.telemetry import init_django_telemetry, init_telemetry


def init_django():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "paths.settings")
    init_django_telemetry()
    django.setup()


# Some imports already need Django to be initialized
init_django()

import argparse
import cProfile
from math import log10
import math
import os
from pathlib import Path
import time

from opentelemetry import trace
from dotenv import load_dotenv
from kausal_common.logging.init import init_logging
from nodes.actions.action import ActionNode
from nodes.constants import IMPACT_COLUMN, IMPACT_GROUP, YEAR_COLUMN
from nodes.instance import InstanceLoader
from kausal_common.debugging.perf import PerfCounter
from rich import print
import rich.traceback
from rich.table import Table
from rich.console import Console
import polars as pl

from nodes.units import Quantity

load_dotenv()

console = Console()

if True:
    from kausal_common.logging.warnings import register_warning_handler
    rich.traceback.install(max_frames=10)
    register_warning_handler()

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
parser.add_argument('--delete-stale-nodes', action='store_true', help='delete NodeConfig instances that no longer exist')
parser.add_argument('--print-action-efficiencies', action='store_true', help='calculate and print action efficiencies')
parser.add_argument('--show-perf', action='store_true', help='show performance info')
parser.add_argument('--profile', action='store_true', help='profile computation performance')
parser.add_argument('--disable-ext-cache', action='store_true', help='disable external cache')
parser.add_argument('--cache-benchmark', action='store_true', help='Perform cache benchmarks')
parser.add_argument('--generate-result-excel', type=str, metavar='FILENAME', help='Create an Excel file from model outputs')

# parser.add_argument('--sync', action='store_true', help='sync db to node contents')
args = parser.parse_args()

if (args.instance and args.config) or (not args.instance and not args.config):
    print('Specify either "--instance" or "--config"')
    exit(1)

if args.disable_ext_cache:
    os.environ['REDIS_URL'] = ''

if args.instance:
    tracer = trace.get_tracer('load-nodes')

    with tracer.start_as_current_span('django-init', attributes={'instance_id': args.instance}) as span:
        from nodes.models import InstanceConfig
        instance_obj: InstanceConfig = InstanceConfig.objects.get(identifier=args.instance)
        instance = instance_obj.get_instance()
        context = instance.context

else:
    tracer = trace.get_tracer('load-nodes')
    init_telemetry()
    init_logging()
    with tracer.start_as_current_span('yaml-init', attributes={'config_id': args.config}) as span:
        loader = InstanceLoader.from_yaml(args.config)
        context = loader.context
        instance = loader.instance

if args.pull_datasets:
    context.pull_datasets()

if args.check:
    context.check_mode = True

if args.show_perf:
    context.perf_context.enabled = True


if args.cache_benchmark:
    from kausal_common.debugging.perf import PerfCounter

    pc = PerfCounter()
    context.skip_cache = True
    nodes = context.get_outcome_nodes()
    test_dfs = []
    for n in nodes:
        df = n.get_output_pl()
        test_dfs.append(df)
    cache = context.cache
    old_client = cache.client
    cache.client = None
    pc.display('begin')
    for i in range(1000):
        key = 'key-%d' % i
        cache.set(key, test_dfs[i % len(test_dfs)])
    pc.display('end', show_time_to_last=True)
    exit()

profile: cProfile.Profile | None
if args.profile:
    profile = cProfile.Profile()
else:
    profile = None


if args.print_graph:
    context.print_graph(include_datasets=True)

if args.skip_cache:
    context.skip_cache = True

if args.scenario:
    context.activate_scenario(context.get_scenario(args.scenario))

if args.list_params:
    context.print_all_parameters()

context.load_all_dvc_datasets()

for node_id in (args.debug_nodes or []):
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

if args.check or args.update_instance or args.update_nodes:
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

    from nodes.models import InstanceConfig
    from django.db import transaction

    ins_obj = InstanceConfig.objects.filter(identifier=instance.id).first()
    if ins_obj is None:
        print("Creating instance %s" % instance.id)
        instance_obj = InstanceConfig.create_for_instance(instance)
        # TODO: Create InstanceHostname somewhere?
    else:
        instance_obj = ins_obj

    with transaction.atomic():
        if args.update_instance:
            instance_obj.update_from_instance(instance, overwrite=True)
            instance_obj.save()
        instance_obj.sync_nodes(update_existing=args.update_nodes, delete_stale=args.delete_stale_nodes, overwrite=args.overwrite)
        instance_obj.sync_dimensions(update_existing=True, delete_stale=args.delete_stale_nodes)
        instance_obj.refresh_from_db()
        instance_obj.create_default_content()

for param_arg in (args.param or []):
    param_id, val = param_arg.split('=')
    context.set_parameter_value(param_id, val)

if args.normalize:
    norm = context.normalizations[args.normalize]
    context.active_normalization = norm

all_filters = []
for line in args.filter or []:
    for f in line.split(','):
        all_filters.append(f)


if args.generate_result_excel:
    from nodes.excel_results import create_result_excel
    excel_path = Path(args.generate_result_excel)
    existing_wb: Path | None
    if excel_path.exists():
        existing_wb = excel_path
        context.log.info("Excel workbook '%s' exists; opening it as the base" % excel_path)
    else:
        context.log.info("Excel workbook '%s' does not exist; creating new" % excel_path)
        existing_wb = None
    out = create_result_excel(context, existing_wb=existing_wb)
    with open(excel_path, 'wb') as f:
        f.write(out.getvalue())


for node_id in (args.node or []):
    node = context.get_node(node_id)
    with context.run():
        node.print_output(filters=all_filters or None)
        #node.plot_output(filters=all_filters or None)


    if isinstance(node, ActionNode):
        output_nodes = node.output_nodes
        for n in output_nodes:
            print("Impact of %s on OUTPUT node %s" % (node, n))
            node.print_impact(n)

        for n in context.get_outcome_nodes():
            print("Impact of action %s on OUTCOME node %s" % (node, n))
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


if args.print_action_efficiencies:
    def print_action_efficiencies():
        pc = PerfCounter("Action efficiencies")
        for aep in context.action_efficiency_pairs:
            title = '%s / %s' % (aep.cost_node.id, aep.impact_node.id)
            pc.display('%s starting' % title)
            table = Table(title=title)
            table.add_column("Action")
            table.add_column("Cumulative efficiency")
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

            console = Console()
            rows = sorted(rows, key=lambda x: x[1].m if x[1] is not None else 1e100)
            for row in rows:
                table.add_row(row[0], str(row[1]))
            console.print(table)

    def print_impacts():
        pc = PerfCounter("Action impacts")
        for outcome_node in context.get_outcome_nodes():
            title = outcome_node.id
            pc.display('%s starting' % title)
            table = Table(title=title)
            years = []
            table.add_column("Action")
            table.add_column("Impact %s" % context.target_year)
            years.append(context.target_year)
            if context.model_end_year != context.target_year:
                table.add_column("Impact %s" % context.model_end_year)
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
            rows = sorted(rows, key=lambda x: x[1])
            for row in rows:
                table.add_row(row[0], *[str(x) for x in row[1:]])
            console.print(table)

    if profile is not None:
        profile.enable()

    with context.run():
        if context.action_efficiency_pairs:
            print_action_efficiencies()
        else:
            print_impacts()
    if profile is not None:
        profile.disable()
        profile.dump_stats('action_efficiencies_profile.out')


if False:
    loader.context.dataset_repo.pull_datasets()
    loader.context.print_all_parameters()
    loader.context.generate_baseline_values()
    # for sector in page.get_sectors():
    #    print(sector)

