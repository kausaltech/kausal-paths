import argparse
import sys
from nodes.instance import InstanceLoader


if True:
    # Print traceback for warnings
    import traceback
    import warnings

    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
        log = sys.stderr
        traceback.print_stack()
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = warn_with_traceback


loader = InstanceLoader('configs/tampere.yaml')
loader.print_graph()
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


parser = argparse.ArgumentParser(description='Execute the computational graph')
parser.add_argument('--param', action='append', type=str, help='set a parameter')
parser.add_argument('--list-params', action='store_true', help='list parameters')
parser.add_argument('--node', type=str, nargs='+', help='Compute node')
args = parser.parse_args()

if args.list_params:
    context.print_params()

for param_arg in (args.param or []):
    param_id, val = param_arg.split('=')
    context.set_param_value(param_id, val)

for node_id in (args.node or []):
    node = context.get_node(node_id)
    node.print_output()

if False:
    loader.context.dataset_repo.pull_datasets()
    loader.context.print_params()
    loader.context.generate_baseline_values()
    #for sector in page.get_sectors():
    #    print(sector)
