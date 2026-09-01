"""
Small in-memory fixtures for testing the node input contract.

These helpers intentionally do not construct a Django ``Context`` or an
``InstanceGraph``.  They exercise the runtime binding registry at the same
boundary used by real nodes.
"""

from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

import polars as pl

from common.polars import DataFrameMeta, PathsDataFrame, to_ppdf
from nodes.constants import VALUE_COLUMN, YEAR_COLUMN
from nodes.node import Node
from nodes.runtime_input import RuntimeInputBinding
from nodes.units import unit_registry

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from nodes.datasets import Dataset
    from nodes.defs.port_def import InputPortDeclaration


def frame(values: Iterable[float], *, unit: str = 'kWh') -> PathsDataFrame:
    values = list(values)
    raw = pl.DataFrame({YEAR_COLUMN: list(range(2020, 2020 + len(values))), VALUE_COLUMN: values})
    return to_ppdf(raw, DataFrameMeta(units={VALUE_COLUMN: unit_registry.parse_units(unit)}, primary_keys=[YEAR_COLUMN]))


def node_case(*declarations: InputPortDeclaration) -> Node:
    """Return a minimally initialised Node suitable for input accessor tests."""
    node_class = type('InputTestNode', (Node,), {'input_port_declarations': declarations})
    node: Node = object.__new__(node_class)
    node.id = 'test_node'
    return node


def binding(
    role: str,
    value: PathsDataFrame | None = None,
    *,
    position: int = 0,
    source_kind: Literal['node', 'dataset'] = 'dataset',
    source_id: str | None = None,
    source: Node | Dataset | None = None,
    target_port_id: UUID | None = None,
    value_loader: Callable[[], PathsDataFrame] | None = None,
    binding_id: UUID | None = None,
) -> RuntimeInputBinding:
    if value_loader is None:
        assert value is not None

        def load_value() -> PathsDataFrame:
            return value

        value_loader = load_value
    return RuntimeInputBinding(
        id=binding_id or uuid4(),
        port_role=role,
        position=position,
        source_kind=source_kind,
        source=source,
        value_loader=value_loader,
        source_id=source_id,
        target_port_id=target_port_id,
    )


def bind(node: Node, bindings: Iterable[RuntimeInputBinding]) -> Node:
    node.bind_runtime_inputs(tuple(bindings))
    return node


def add_binding(node: Node, runtime_binding: RuntimeInputBinding) -> None:
    """Append one binding while retaining stable positional ordering."""
    node.bind_runtime_inputs((*node.runtime_input_bindings, runtime_binding))
