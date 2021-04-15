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
print(metric.get_historical_values(loader.context))
