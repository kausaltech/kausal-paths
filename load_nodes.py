from nodes.instance import InstanceLoader

print(1)
loader = InstanceLoader('configs/tampere.yaml')
loader.print_graph()
print(2)


page = list(loader.pages.values())[0]
card = page.cards[0]
print(card)
metric = card.metrics[0]
print(metric)

print('Historical:')
vals = metric.get_historical_values(loader.context)
for val in vals:
    print(val)

print('\nForecast:')
vals = metric.get_forecast_values(loader.context)
for val in vals:
    print(val)
