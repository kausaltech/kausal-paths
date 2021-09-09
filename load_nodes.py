import argparse
from dotenv import load_dotenv
from nodes.actions.action import ActionNode
import sys
from nodes.instance import InstanceLoader
from common.perf import PerfCounter


if True:
    # Print traceback for warnings
    import traceback
    import warnings

    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
        log = sys.stderr
        traceback.print_stack()
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = warn_with_traceback

load_dotenv()

parser = argparse.ArgumentParser(description='Execute the computational graph')
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
args = parser.parse_args()


loader = InstanceLoader.from_yaml(args.config or 'configs/tampere.yaml')
context = loader.context

page = list(loader.instance.pages.values())[0]


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

if args.pull_datasets:
    context.pull_datasets()

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
        df = node.get_output(context)
        na_count = df.isna().sum().sum()
        if na_count:
            print('Node %s has NaN values:' % node.id)
            node.print_output(context)

        if node.baseline_values is not None:
            na_count = node.baseline_values.isna().sum().sum()
            if na_count:
                print('Node %s baseline forecast has NaN values:' % node.id)
                node.print_pint_df(node.baseline_values)

    import os
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "paths.settings")
    django.setup()
    from pages.models import NodeContent
    for nc in NodeContent.objects.all():
        if nc.node_id not in context.nodes:
            print('NodeContent exists for missing node %s' % nc.node_id)


for param_arg in (args.param or []):
    param_id, val = param_arg.split('=')
    context.set_parameter_value(param_id, val)

for node_id in (args.node or []):
    node = context.get_node(node_id)
    node.print_output(context)
    if isinstance(node, ActionNode):
        node.print_impact(context, context.get_node('net_emissions'))


if False:
    loader.context.dataset_repo.pull_datasets()
    loader.context.print_all_parameters()
    loader.context.generate_baseline_values()
    #for sector in page.get_sectors():
    #    print(sector)
