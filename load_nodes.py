from nodes.instance import InstanceLoader

loader = InstanceLoader('configs/tampere.yaml')
loader.print_graph()
context = loader.context

page = list(loader.pages.values())[0]
card = page.cards[0]

metric = card.metrics[0]

print(metric)
print('Historical:')
vals = metric.get_historical_values(context)
for val in vals:
    print(val)

print('\nForecast:')
vals = metric.get_forecast_values(context)
for val in vals:
    print(val)

print('\nRoadmap scenario:')

context.activate_scenario(context.get_scenario('roadmap'))
page.refresh()

print('Historical:')
vals = metric.get_historical_values(context)
for val in vals:
    print(val)

print('\nForecast:')
vals = metric.get_forecast_values(context)
for val in vals:
    print(val)
