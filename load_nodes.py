#!/usr/bin/env python3
import argparse
from dotenv import load_dotenv
from nodes.actions.action import ActionNode
import sys
from nodes.instance import Instance, InstanceLoader
from common.perf import PerfCounter
import rich.traceback


if True:
    # Print traceback for warnings
    import traceback
    import warnings

    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
        log = sys.stderr
        traceback.print_stack()
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = warn_with_traceback
    # Pretty tracebacks
    rich.traceback.install()

load_dotenv()

django_initialized = False


def init_django():
    global django_initialized
    if django_initialized:
        return
    import os
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "paths.settings")
    django.setup()
    django_initialized = True


parser = argparse.ArgumentParser(description='Execute the computational graph')
parser.add_argument('--instance', type=str, help='instance identifier')
parser.add_argument('--config', type=str, help='config yaml file')
parser.add_argument('--baseline', action='store_true', help='generate baseline scenario values')
parser.add_argument('--scenario', type=str, help='select scenario')
parser.add_argument('--param', action='append', type=str, help='set a parameter')
parser.add_argument('--list-params', action='store_true', help='list parameters')
parser.add_argument('--debug', action='store_true', help='enable debug messages and disable cache')
parser.add_argument('--check', action='store_true', help='perform sanity checking')
parser.add_argument('--skip-cache', action='store_true', help='skip caching')
parser.add_argument('--node', type=str, nargs='+', help='compute node')
parser.add_argument('--pull-datasets', action='store_true', help='refresh all datasets')
parser.add_argument('--print-graph', action='store_true', help='print the graph')
# parser.add_argument('--sync', action='store_true', help='sync db to node contents')
args = parser.parse_args()

if (args.instance and args.config) or (not args.instance and not args.config):
    print('Specify either "--instance" or "--config"')
    exit(1)

if args.instance:
    init_django()
    from nodes.models import InstanceConfig
    instance_obj: InstanceConfig = InstanceConfig.objects.get(identifier=args.instance)
    instance = instance_obj.get_instance()
    context = instance.context

if args.config:
    loader = InstanceLoader.from_yaml(args.config)
    context = loader.context
    instance = loader.instance

if args.pull_datasets:
    context.pull_datasets()


def print_metric(metric):
    print(metric)

    print('Historical:')
    vals = metric.get_historical_values(context)
    for val in vals:
        print(val)

    print('\nBaseline forecast:')
    vals = metric.get_baseline_forecast_values(context)
    for val in vals:
        print(val)

    print('\nRoadmap scenario:')
    vals = metric.get_forecast_values(context)
    for val in vals:
        print(val)


if args.print_graph:
    context.print_graph()

if args.skip_cache:
    context.skip_cache = True

if args.scenario:
    context.activate_scenario(context.get_scenario(args.scenario))

if args.list_params:
    context.print_all_parameters()

if args.debug:
    for node_id in (args.node or []):
        node = context.get_node(node_id)
        node.debug = True

if args.baseline:
    pc = PerfCounter('Baseline')
    pc.display('generating')
    context.generate_baseline_values()
    pc.display('done')

if args.check:
    for node_id, node in context.nodes.items():
        df = node.get_output()
        na_count = df.isna().sum().sum()
        if na_count:
            print('Node %s has NaN values:' % node.id)
            node.print_output(context)

        if node.baseline_values is not None:
            na_count = node.baseline_values.isna().sum().sum()
            if na_count:
                print('Node %s baseline forecast has NaN values:' % node.id)
                node.print_pint_df(node.baseline_values)

    init_django()
    from nodes.models import InstanceConfig

    instance_obj = InstanceConfig.objects.filter(identifier=instance.id).first()
    if instance_obj is None:
        print("Creating instance %s" % instance.id)
        instance_obj = InstanceConfig.create_for_instance(instance)

    instance_obj.sync_nodes()
    instance_obj.create_default_content()

for param_arg in (args.param or []):
    param_id, val = param_arg.split('=')
    context.set_parameter_value(param_id, val)

for node_id in (args.node or []):
    node = context.get_node(node_id)
    node.print_output()
    if isinstance(node, ActionNode):
        node.print_impact(context, context.get_node('net_emissions'))

if False:
    loader.context.dataset_repo.pull_datasets()
    loader.context.print_all_parameters()
    loader.context.generate_baseline_values()
    # for sector in page.get_sectors():
    #    print(sector)
