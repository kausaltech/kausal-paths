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


if True:
    for sector in page.get_sectors():
        print(sector)
