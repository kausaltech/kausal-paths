from __future__ import annotations
from nodes.simple import SectorEmissions
from typing import Dict, List, Optional
from dataclasses import dataclass

import pandas as pd
from nodes import Node, Context
from nodes.actions import ActionNode
from nodes.constants import BASELINE_VALUE_COLUMN, FORECAST_COLUMN, VALUE_COLUMN


@dataclass
class YearlyValue:
    year: int
    value: float


@dataclass
class Metric:
    id: str
    name: str
    df: pd.DataFrame

    def split_df(self) -> Dict[str, List[YearlyValue]]:
        if hasattr(self, 'split_values'):
            return self.split_values

        df = self.df.copy()

        if df is None or VALUE_COLUMN not in df.columns:
            self.split_values = None
            return None

        df.index.name = 'year'
        df = df.reset_index()
        if df.isnull().sum().sum():
            raise Exception('Metric %s contains NaN values' % self.id)

        df = df.rename(columns={VALUE_COLUMN: 'value'})
        if hasattr(df.value, 'pint'):
            df.value = df.value.pint.m
        is_forecast = df[FORECAST_COLUMN]

        if BASELINE_VALUE_COLUMN in df.columns:
            bs = df[BASELINE_VALUE_COLUMN]
            if hasattr(bs, 'pint'):
                bs = bs.pint.m
            bdf = df[['year']].copy()
            bdf['value'] = bs
            baseline = bdf.loc[is_forecast, ['year', 'value']].to_dict('records')
        else:
            baseline = None

        forecast = df.loc[is_forecast, ['year', 'value']].to_dict('records')
        historical = df.loc[~is_forecast, ['year', 'value']].to_dict('records')
        out = dict(
            historical=[YearlyValue(**r) for r in historical],
            forecast=[YearlyValue(**r) for r in forecast],
            baseline=[YearlyValue(**r) for r in baseline] if baseline else None
        )
        self.split_values = out
        return out

    def get_historical_values(self) -> List[YearlyValue]:
        vals = self.split_df()
        if not vals:
            return None
        return vals['historical']

    def get_forecast_values(self) -> List[YearlyValue]:
        vals = self.split_df()
        if not vals:
            return None
        return vals['forecast']

    def get_baseline_forecast_values(self) -> List[YearlyValue]:
        vals = self.split_df()
        if not vals:
            return None
        return vals['baseline']

'''
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
'''

@dataclass
class Page:
    id: str
    name: str
    path: str


'''
@dataclass
class CardPage(Page):
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
'''


@dataclass
class EmissionSector:
    node: Node
    parent: Optional[EmissionSector]
    color: str = None
    metric: Metric = None

    @property
    def id(self) -> str:
        return self.node.id

    @property
    def name(self) -> str:
        return self.node.name

    def __post_init__(self):
        self.metric = Metric(id=self.id, name=self.name, df=self.node.get_output())


@dataclass
class EmissionPage(Page):
    node: Node

    def _get_node_sectors(self, node: Node, parent: EmissionSector = None) -> List[EmissionSector]:
        sectors = []
        sector = EmissionSector(node=node, parent=parent)
        if node.color:
            sector.color = node.color
        elif parent is not None and parent.color:
            sector.color = parent.color
        sectors.append(sector)
        for input_node in node.input_nodes:
            if input_node.quantity != 'emissions' or isinstance(input_node, ActionNode):
                continue
            sectors += self._get_node_sectors(input_node, sector)
        return sectors

    def get_sectors(self):
        return self._get_node_sectors(self.node, None)


@dataclass
class ActionPage(Page):
    action: ActionNode
