import sys
from nodes.instance import InstanceLoader

loader = InstanceLoader('configs/tampere.yaml')
loader.print_graph()
context = loader.context

page = list(loader.pages.values())[0]


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


if len(sys.argv) > 1:
    node_id = sys.argv[1]
    node = context.get_node(node_id)
    print(node.get_output())
    exit()

if True:
    import dvc_pandas
    dvc_pandas.pull_datasets(repo_url=loader.context.dataset_repo_url)

    loader.context.generate_baseline_values()
    for sector in page.get_sectors():
        print(sector)
