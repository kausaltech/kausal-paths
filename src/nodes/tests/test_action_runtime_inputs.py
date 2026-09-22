"""Focused regression tests for the action-class port migrations."""

from pathlib import Path
from typing import Literal
from uuid import NAMESPACE_URL, uuid3

import polars as pl
import pytest

from common.polars import DataFrameMeta, PathsDataFrame, to_ppdf
from nodes.actions.linear import DatasetDifferenceAction
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.costs import SelectiveNode
from nodes.exceptions import NodeError
from nodes.instance_loader import InstanceLoader, InstanceYAMLConfig
from nodes.instance_parser import parse_instance_snapshot
from nodes.tests.node_input_harness import bind, binding, node_case
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


def _series(values: list[float], *, forecast: bool | None = None, metric: str = VALUE_COLUMN) -> PathsDataFrame:
    """Build a one-metric frame, optionally without a Forecast column (the dataset shape)."""
    columns: dict[str, list[float] | list[int] | list[bool]] = {
        YEAR_COLUMN: list(range(2020, 2020 + len(values))),
        metric: values,
    }
    if forecast is not None:
        columns[FORECAST_COLUMN] = [forecast] * len(values)
    return to_ppdf(
        pl.DataFrame(columns),
        DataFrameMeta(units={metric: unit_registry.parse_units('%')}, primary_keys=[YEAR_COLUMN]),
    )


def test_dataset_difference_action_declares_baseline_and_goal() -> None:
    assert DatasetDifferenceAction.baseline_port.role == 'baseline'
    assert DatasetDifferenceAction.goal_port.role == 'goal'
    assert DatasetDifferenceAction.baseline_port.required is True
    assert DatasetDifferenceAction.goal_port.required is True
    assert DatasetDifferenceAction.consumes_all_inputs_through_ports is True


@pytest.mark.parametrize('source_kind', ['node', 'dataset'])
def test_dataset_difference_action_accepts_either_source_kind(source_kind: Literal['node', 'dataset']) -> None:
    """The port contract is the same whichever kind of source fills it."""
    baseline = _series([10.0, 12.0], forecast=False)
    goal = _series([20.0], forecast=True)
    node = bind(
        node_case(DatasetDifferenceAction.baseline_port, DatasetDifferenceAction.goal_port),
        [
            binding('baseline', baseline, source_kind=source_kind),
            binding('goal', goal, source_kind=source_kind),
        ],
    )
    assert node.require_input(DatasetDifferenceAction.baseline_port) is baseline
    assert node.require_input(DatasetDifferenceAction.goal_port) is goal


def test_dataset_difference_action_missing_goal_is_a_binding_error() -> None:
    node = bind(
        node_case(DatasetDifferenceAction.baseline_port, DatasetDifferenceAction.goal_port),
        [binding('baseline', _series([10.0], forecast=False))],
    )
    with pytest.raises(NodeError, match=r"Required input role 'goal' has no bindings"):
        node.require_input(DatasetDifferenceAction.goal_port)


def test_dataset_difference_action_rejects_a_second_binding_on_a_single_port() -> None:
    node = bind(
        node_case(DatasetDifferenceAction.baseline_port),
        [
            binding('baseline', _series([10.0], forecast=False), position=0),
            binding('baseline', _series([11.0], forecast=False), position=1),
        ],
    )
    with pytest.raises(NodeError, match=r"Input role 'baseline' has 2 bindings, expected one"):
        node.require_input(DatasetDifferenceAction.baseline_port)


def test_selective_node_declares_one_role_per_conditional_cost() -> None:
    assert SelectiveNode.co2_cost_port.role == 'co2_cost'
    assert SelectiveNode.capacity_cost_port.role == 'capacity_cost'
    assert SelectiveNode.health_cost_port.role == 'health_cost'
    # Every conditional role is declared, and each names the global parameter that gates it.
    declared = {str(port.role) for port in SelectiveNode.input_port_declarations}
    assert set(SelectiveNode.conditional_cost_roles) <= declared
    assert set(SelectiveNode.conditional_cost_roles.values()) <= set(SelectiveNode.global_parameters)
    assert all(not port.required for port in SelectiveNode.input_port_declarations if str(port.role) != 'additive')


def test_selective_node_sums_in_stored_position_order_across_roles() -> None:
    """Roles are gathered separately but must be summed in the original binding order."""
    node = bind(
        node_case(*SelectiveNode.input_port_declarations),
        [
            binding('additive', _series([1.0]), position=0, source_kind='node'),
            binding('co2_cost', _series([2.0]), position=1, source_kind='node'),
            binding('additive', _series([4.0]), position=2, source_kind='node'),
            binding('health_cost', _series([8.0]), position=3, source_kind='node'),
        ],
    )
    selected = []
    for port in SelectiveNode.input_port_declarations:
        selected.extend(node.iter_input_bindings(port))
    selected.sort(key=lambda b: b.position)
    assert [b.position for b in selected] == [0, 1, 2, 3]
    assert [b.port_role for b in selected] == ['additive', 'co2_cost', 'additive', 'health_cost']


def test_gronlogik_yaml_projects_source_node_tags_to_cost_roles() -> None:
    """These tags sit on the source cost nodes, not on the edges that read them."""
    data = InstanceYAMLConfig.load_for_entrypoint(Path('configs/gronlogik.yaml').resolve()).data
    assert data is not None
    snapshot = parse_instance_snapshot(data, instance_uuid=uuid3(NAMESPACE_URL, 'paths:gronlogik-cost-input-test'))
    loader = object.__new__(InstanceLoader)
    loader.instance_config = None
    loader._stash_snapshot_bindings(snapshot)

    node = next(n for n in loader._instance_graph.nodes if n.identifier == 'cost_per_area_single_family')
    assert {node.role_for_input_port(port) for port in node.spec.input_ports} == {
        'additive',
        'co2_cost',
        'capacity_cost',
        'health_cost',
    }


def test_nzc_yaml_projects_legacy_difference_tags_to_semantic_roles() -> None:
    data = InstanceYAMLConfig.load_for_entrypoint(Path('configs/nzc.yaml').resolve()).data
    assert data is not None
    snapshot = parse_instance_snapshot(data, instance_uuid=uuid3(NAMESPACE_URL, 'paths:nzc-action-input-test'))
    loader = object.__new__(InstanceLoader)
    loader.instance_config = None
    loader._stash_snapshot_bindings(snapshot)

    action = next(node for node in loader._instance_graph.nodes if node.identifier == 'a51_increase_waste_recycling')
    assert {action.role_for_input_port(port) for port in action.spec.input_ports} == {'baseline', 'goal'}
