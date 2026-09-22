"""Focused regression tests for the per-source-metric port migrations in nodes.buildings."""

from pathlib import Path
from uuid import NAMESPACE_URL, uuid3

import polars as pl
import pytest

from common.polars import DataFrameMeta, PathsDataFrame, to_ppdf
from nodes.buildings import CfNode, FloorAreaNode
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.exceptions import NodeError
from nodes.instance_loader import InstanceLoader, InstanceYAMLConfig
from nodes.instance_parser import parse_instance_snapshot
from nodes.node import Node
from nodes.simple import AnnuityNode
from nodes.tests.node_input_harness import bind, binding
from nodes.units import unit_registry

pytestmark = pytest.mark.django_db


def _source(node_id: str) -> Node:
    """Build a bare runtime source node; only its id takes part in pairing the metric roles."""
    node: Node = object.__new__(Node)
    node.id = node_id
    return node


def _share(value: float, *, metric: str = VALUE_COLUMN) -> PathsDataFrame:
    raw = pl.DataFrame({YEAR_COLUMN: [2020], metric: [value], FORECAST_COLUMN: [False]})
    return to_ppdf(
        raw,
        DataFrameMeta(units={metric: unit_registry.parse_units('dimensionless')}, primary_keys=[YEAR_COLUMN]),
    )


def test_floor_area_node_declares_one_role_per_source_metric() -> None:
    assert FloorAreaNode.floor_area_port.role == 'floor_area'
    assert FloorAreaNode.triggered_port.role == 'triggered'
    assert FloorAreaNode.compliant_port.role == 'compliant'
    assert FloorAreaNode.triggered_port.multi is True
    assert FloorAreaNode.compliant_port.multi is True
    # Aggregation stays absent: the shares of different actions are never summed.
    assert FloorAreaNode.triggered_port.aggregation is None
    assert FloorAreaNode.legacy_input_port_roles_by_source_metric == {
        'triggered': 'triggered',
        'compliant': 'compliant',
        VALUE_COLUMN: 'floor_area',
    }


def test_cf_node_reads_a_different_metric_of_the_same_actions() -> None:
    assert CfNode.improvement_port.role == 'improvement'
    assert CfNode.baseline_port.role == 'baseline'
    assert CfNode.baseline_port.required is False
    assert CfNode.legacy_input_port_roles_by_source_metric == {
        'improvement': 'improvement',
        VALUE_COLUMN: 'baseline',
    }


def test_metric_roles_are_paired_back_up_by_source_node() -> None:
    """Each action arrives as one binding per metric; the class rejoins them by source."""
    first, second = _source('action_one'), _source('action_two')
    node: FloorAreaNode = object.__new__(FloorAreaNode)
    node.id = 'affected_floor_area'
    bind(
        node,
        [
            binding('triggered', _share(0.1), position=0, source_kind='node', source=first, source_id=first.id),
            binding('compliant', _share(0.2), position=1, source_kind='node', source=second, source_id=second.id),
            binding('triggered', _share(0.3), position=2, source_kind='node', source=second, source_id=second.id),
            binding('compliant', _share(0.4), position=3, source_kind='node', source=first, source_id=first.id),
        ],
    )
    triggered = node._metric_by_source(FloorAreaNode.triggered_port, 'triggered')
    compliant = node._metric_by_source(FloorAreaNode.compliant_port, 'compliant')

    # Keys follow binding position, which is the order the old single pass used.
    assert list(triggered) == ['action_one', 'action_two']
    assert triggered['action_one']['triggered'].to_list() == [0.1]
    assert triggered['action_two']['triggered'].to_list() == [0.3]
    # Pairing is by source, not by position: compliant arrived in the opposite order.
    assert compliant['action_one']['compliant'].to_list() == [0.4]
    assert compliant['action_two']['compliant'].to_list() == [0.2]


def test_annuity_node_pairs_cost_and_lifetime_by_source() -> None:
    """Currency and term reach the node as separate bindings and must rejoin per source."""
    assert AnnuityNode.currency_port.role == 'currency'
    assert AnnuityNode.term_port.role == 'term'
    assert AnnuityNode.discount_rate_port.role == 'discount_rate'
    assert AnnuityNode.legacy_input_port_roles_by_tag == {'discount_rate': 'discount_rate'}
    assert AnnuityNode.legacy_input_port_roles_by_source_metric == {'currency': 'currency', 'term': 'term'}


def test_annuity_node_rejects_a_source_missing_one_of_its_two_metrics() -> None:
    node: AnnuityNode = object.__new__(AnnuityNode)
    node.id = 'annuitized_costs'
    source = _source('an_action')
    bind(
        node,
        [binding('currency', _share(1000.0), position=0, source_kind='node', source=source, source_id=source.id)],
    )
    with pytest.raises(NodeError, match='must supply both a currency and a term'):
        node._cost_frames()


def test_longmont_yaml_splits_multi_metric_actions_into_per_metric_roles() -> None:
    """One legacy input_nodes entry against a 3-metric action is already several ports."""
    data = InstanceYAMLConfig.load_for_entrypoint(Path('configs/longmont-old.yaml').resolve()).data
    assert data is not None
    snapshot = parse_instance_snapshot(data, instance_uuid=uuid3(NAMESPACE_URL, 'paths:longmont-buildings-input-test'))
    loader = object.__new__(InstanceLoader)
    loader.instance_config = None
    loader._stash_snapshot_bindings(snapshot)

    floor_area = next(n for n in loader._instance_graph.nodes if n.identifier == 'affected_square_footage')
    roles = [floor_area.role_for_input_port(port) for port in floor_area.spec.input_ports]
    assert roles == ['floor_area', 'triggered', 'compliant', 'triggered', 'compliant', 'triggered', 'compliant']
    assert None not in roles

    cf = next(n for n in loader._instance_graph.nodes if n.identifier == 'electricity_eui_improvement')
    cf_roles = [cf.role_for_input_port(port) for port in cf.spec.input_ports]
    # The same three actions, read through their improvement metric instead.
    assert cf_roles == ['baseline', 'improvement', 'improvement', 'improvement']
