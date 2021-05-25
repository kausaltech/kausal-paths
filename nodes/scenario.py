from typing import List, Any, Tuple
from dataclasses import dataclass
from params import Parameter


@dataclass
class Scenario:
    id: str
    name: str
    default: bool = False
    # Dict of params and their values in the scenario
    params: List[Tuple[Parameter, Any]] = None

    def __post_init__(self):
        self.params = []

    def activate(self):
        for param, val in self.params:
            param.set(val)
