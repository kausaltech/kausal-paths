from __future__ import annotations
from typing import Dict, List
from dataclasses import dataclass

import pandas as pd
from nodes import Node, Context
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN


@dataclass
class YearlyValue:
    year: int
    value: float


@dataclass
class Metric:
    id: str
    name: str
    nodes: List[Node] = None

    def __post_init__(self):
        self.values = None

    def df_to_yearly(self, df: pd.DataFrame) -> Dict[str, List[YearlyValue]]:
        df.index.name = 'year'
        df = df.reset_index()
        df = df.rename(columns={VALUE_COLUMN: 'value'})
        forecast = df.loc[df[FORECAST_COLUMN], ['year', 'value']].to_dict('records')
        historical = df.loc[~df[FORECAST_COLUMN], ['year', 'value']].to_dict('records')
        return dict(
            historical=[YearlyValue(**r) for r in historical],
            forecast=[YearlyValue(**r) for r in forecast],
        )

    def _get_values(self, context):
        node = context.get_node(self.id)
        df = node.get_output()
        self.values = self.df_to_yearly(df)

    def get_historical_values(self, context: Context) -> List[YearlyValue]:
        if not self.values:
            self._get_values(context)
        return self.values['historical']

    def get_forecast_values(self, context: Context) -> List[YearlyValue]:
        if not self.values:
            self._get_values(context)
        return self.values['forecast']

    def get_baseline_forecast_values(self, context: Context) -> List[YearlyValue]:
        node = context.get_node(self.id)
        df = node.baseline_values
        vals = self.df_to_yearly(df)
        return vals['forecast']

    def refresh(self):
        self.values = None


@dataclass
class Card:
    id: str
    name: str
    type: str = None  # 'stacked'
    metrics: List[Metric] = None
    upstream_cards: List[Card] = None
    downstream_cards: List[Card] = None

    def __post_init__(self):
        self.metrics = []
        self.upstream_cards = []
        self.downstream_cards = []

    def add_metrics(self, metrics, context: Context):
        for m in metrics:
            if m.get('only_inputs'):
                node = context.get_node(m['id'])
                input_metrics = []
                for input_node in node.input_nodes:
                    input_metrics.append(dict(id=input_node.id, name=input_node.name))
                self.add_metrics(input_metrics, context)
                continue

            if not m.get('name'):
                node = context.get_node(m['id'])
                m['name'] = node.name
            self.metrics.append(Metric(**m))

    def refresh(self):
        for m in self.metrics:
            m.refresh()


@dataclass
class Page:
    id: str
    name: str
    path: str
    cards: List[Card] = None

    def add_cards(self, cards: List[dict], context: Context):
        all_cards = {}
        for c in cards:
            metrics = c.pop('metrics', [])

            if 'id' not in c:
                # A little bit of trickery where we set the card id
                # automatically from metrics if there is only one metric listed.
                if len(metrics) == 1:
                    c['id'] = metrics[0]['id']

            upstream_ids = c.pop('upstream_cards', [])
            card = Card(**c)
            card.add_metrics(metrics, context)
            card.upstream_ids = upstream_ids
            assert card.id not in all_cards
            all_cards[card.id] = card

        # Setup edges
        for c in all_cards.values():
            for uc_id in c.upstream_ids:
                assert uc_id in all_cards
                uc = all_cards[uc_id]
                uc.upstream_cards.append(c)
                c.downstream_cards.append(uc)

        self.cards = list(all_cards.values())

    def refresh(self):
        """Resets the stored values for all cards and metrics."""

        for card in self.cards:
            card.refresh()
