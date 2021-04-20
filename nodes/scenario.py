from typing import List, Any
from dataclasses import dataclass
from nodes.actions import Action


@dataclass
class Scenario:
    id: str
    name: str
    # Dict of actions and their parameters
    actions: list[tuple[Action, dict[str, Any]]] = None

    def __post_init__(self):
        self.actions = []

    def activate(self):
        for action, params in self.actions:
            action.set_params(params)
