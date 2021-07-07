from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, InitVar

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
    node: Optional[Node] = None

    def split_df(self) -> Dict[str, List[YearlyValue]]:
        if hasattr(self, 'split_values'):
            return self.split_values

        if self.df is None or VALUE_COLUMN not in self.df.columns:
            self.split_values = None
            return None

        df = self.df.copy()
        for col in df.columns:
            if hasattr(df[col], 'pint'):
                df[col] = df[col].pint.m

        hist = []
        forecast = []
        baseline = []
        for row in df.itertuples():
            is_fc = getattr(row, FORECAST_COLUMN)
            val = getattr(row, VALUE_COLUMN)
            year = row.Index
            if np.isnan(val):
                raise Exception("Metric %s contains NaN values" % self.id)
            if not is_fc:
                hist.append(YearlyValue(year=year, value=val))
            else:
                bl_val = getattr(row, BASELINE_VALUE_COLUMN, None)
                if bl_val is not None:
                    if np.isnan(bl_val):
                        raise Exception("Metric %s baseline contains NaN values" % self.id)
                    baseline.append(YearlyValue(year=year, value=bl_val))
                forecast.append(YearlyValue(year=year, value=val))

        out = dict(
            historical=hist,
            forecast=forecast,
            baseline=baseline if baseline else None
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

    @property
    def unit(self):
        return self.node.unit if self.node else None

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
    context: InitVar[Context]
    node: Node
    parent: Optional[EmissionSector]
    color: Optional[str] = None
    metric: Optional[Metric] = None

    @property
    def id(self) -> str:
        return self.node.id

    @property
    def name(self) -> str:
        return str(self.node.name)

    def __post_init__(self, context):
        self.metric = Metric(
            id=self.id, name=self.name, df=self.node.get_output(context),
        )


@dataclass
class EmissionPage(Page):
    node: Node

    def _get_node_sectors(self, context: Context, node: Node, parent: EmissionSector = None) -> List[EmissionSector]:
        sectors = []
        sector = EmissionSector(context=context, node=node, parent=parent)
        if node.color:
            sector.color = node.color
        elif parent is not None and parent.color:
            sector.color = parent.color
        sectors.append(sector)
        for input_node in node.input_nodes:
            if input_node.quantity != 'emissions' or isinstance(input_node, ActionNode):
                continue
            sectors += self._get_node_sectors(context, input_node, sector)
        return sectors

    def get_sectors(self, context):
        return self._get_node_sectors(context, self.node, None)


@dataclass
class ActionPage(Page):
    action: ActionNode
