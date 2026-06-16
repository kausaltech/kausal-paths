"""
Tests for the draft-model fault-tolerance behavior.

See ``docs/architecture/fault-tolerance.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from factory import Sequence

from nodes.edges import Edge
from nodes.exceptions import NodeError
from nodes.node import NodeErrorPhase, NodeStatus, NodeStatusError
from nodes.simple import AdditiveNode, SimpleNode
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeFactory

if TYPE_CHECKING:
    from nodes.context import Context

pytestmark = pytest.mark.django_db


class _BoomNode(SimpleNode):
    """A node whose computation always fails, counting how often it is attempted."""

    def compute(self):
        self.compute_calls = getattr(self, 'compute_calls', 0) + 1
        raise NodeError(self, 'boom')


class AdditiveNodeFactory(NodeFactory):
    class Meta:
        model = AdditiveNode

    # Distinct id namespace: each factory subclass has its own Sequence counter, so a shared
    # 'node{i}' prefix would let two nodes collide on id (and thus on cache key) across factories.
    id = Sequence(lambda i: f'additive{i}')


class BoomNodeFactory(NodeFactory):
    class Meta:
        model = _BoomNode

    id = Sequence(lambda i: f'boom{i}')


def _leaf(context: Context) -> AdditiveNode:
    """Make an additive leaf fed by the factory's fixed dataset (computes cleanly)."""
    return cast('AdditiveNode', AdditiveNodeFactory.create(context=context))


def _sum_of(context: Context, inputs) -> AdditiveNode:
    """Make an additive node with no dataset that sums the given input nodes."""
    node = cast('AdditiveNode', AdditiveNodeFactory.create(context=context, input_datasets=[]))
    for inp in inputs:
        # Register the shared edge on both nodes, as the instance loader does.
        edge = Edge(input_node=inp, output_node=node, tags=[])
        node.add_edge(edge)
        inp.add_edge(edge)
    context.finalize_nodes()
    return node


# --- mark_status (monotonic, never recovers within a run) -------------------


def test_mark_status_only_moves_toward_failure(node):
    assert node.status is None
    node.mark_status(NodeStatus.OK)
    assert node.status is NodeStatus.OK
    node.mark_status(NodeStatus.DEGRADED)
    assert node.status is NodeStatus.DEGRADED
    # A less-severe status does not override a more-severe one.
    node.mark_status(NodeStatus.OK)
    assert node.status is NodeStatus.DEGRADED
    node.mark_status(NodeStatus.FAILED)
    assert node.status is NodeStatus.FAILED
    node.mark_status(NodeStatus.OK)
    assert node.status is NodeStatus.FAILED


def test_mark_status_records_error(node):
    err = NodeStatusError(phase=NodeErrorPhase.COMPUTATION, message='kaboom')
    node.mark_status(NodeStatus.FAILED, err)
    assert node.status is NodeStatus.FAILED
    assert node.status_errors == [err]


# --- compute-time status ----------------------------------------------------


def test_clean_node_marked_ok(context):
    leaf = _leaf(context)
    leaf.get_output_pl()
    assert leaf.status is NodeStatus.OK
    assert leaf.status_errors == []


def test_failed_node_records_own_error(context):
    boom = BoomNodeFactory.create(context=context)
    with pytest.raises(NodeError):
        boom.get_output_pl()
    assert boom.status is NodeStatus.FAILED
    assert len(boom.status_errors) == 1
    assert boom.status_errors[0].phase is NodeErrorPhase.COMPUTATION


def test_failure_is_memoized_not_recomputed(context):
    boom = cast('_BoomNode', BoomNodeFactory.create(context=context))
    for _ in range(3):
        with pytest.raises(NodeError):
            boom.get_output_pl()
    # The failing compute() ran only once; later pulls short-circuit on FAILED status.
    assert boom.compute_calls == 1


def test_downstream_cascade_has_no_own_error(context):
    boom = BoomNodeFactory.create(context=context)
    consumer = _sum_of(context, [boom])
    with pytest.raises(NodeError):
        consumer.get_output_pl()
    assert consumer.status is NodeStatus.FAILED
    # The error originated upstream, so the consumer carries status but no own error.
    assert consumer.status_errors == []
    assert boom.status is NodeStatus.FAILED
    assert len(boom.status_errors) == 1


# --- non-tolerant (default) keeps failing fast ------------------------------


def test_default_mode_propagates_failure(context):
    boom = BoomNodeFactory.create(context=context)
    good = _leaf(context)
    summed = _sum_of(context, [good, boom])
    assert context.tolerate_node_failures is False
    with pytest.raises(NodeError):
        summed.get_output_pl()


def test_default_mode_unwired_additive_raises(context):
    node = AdditiveNodeFactory.create(context=context, input_datasets=[])
    context.finalize_nodes()
    with pytest.raises(NodeError):
        node.get_output_pl()


# --- activation: enter_instance_context threads the flag onto the context ----


def _instance_config():
    instance = InstanceFactory.create()
    return InstanceConfigFactory.create(identifier=instance.id, instance=instance)


def test_enter_instance_context_sets_tolerant_flag():
    ic = _instance_config()
    with ic.enter_instance_context(tolerate_node_failures=True) as instance:
        assert instance.context.tolerate_node_failures is True


def test_enter_instance_context_defaults_to_fail_fast():
    ic = _instance_config()
    with ic.enter_instance_context() as instance:
        assert instance.context.tolerate_node_failures is False


# --- editor status field: opt-in compute ------------------------------------


def test_maybe_compute_status_is_noop_without_opt_in(context):
    from nodes.graphql.types.node import _maybe_compute_status

    boom = BoomNodeFactory.create(context=context)
    _maybe_compute_status(boom, compute=False)
    assert boom.status is None  # not computed; cheap default path


def test_maybe_compute_status_computes_and_swallows_failure(context):
    from nodes.graphql.types.node import _maybe_compute_status

    boom = BoomNodeFactory.create(context=context)
    _maybe_compute_status(boom, compute=True)  # must not raise
    assert boom.status is NodeStatus.FAILED
    assert len(boom.status_errors) == 1


def test_maybe_compute_status_marks_clean_node_ok(context):
    from nodes.graphql.types.node import _maybe_compute_status

    leaf = _leaf(context)
    _maybe_compute_status(leaf, compute=True)
    assert leaf.status is NodeStatus.OK


# --- tolerant mode ----------------------------------------------------------


def test_tolerant_additive_skips_failed_input(context):
    context.tolerate_node_failures = True
    good = _leaf(context)
    boom = BoomNodeFactory.create(context=context)
    summed = _sum_of(context, [good, boom])

    df = summed.get_output_pl()
    expected = good.get_output_pl()

    assert summed.status is NodeStatus.DEGRADED
    assert boom.status is NodeStatus.FAILED
    # The partial sum equals the one surviving input.
    assert df['Value'].to_list() == expected['Value'].to_list()


def test_tolerant_unwired_additive_is_incomplete(context):
    context.tolerate_node_failures = True
    node = AdditiveNodeFactory.create(context=context, input_datasets=[])
    context.finalize_nodes()

    df = node.get_output_pl()
    assert node.status is NodeStatus.INCOMPLETE
    assert len(df) == 0
    assert 'Value' in df.columns
    assert df.has_unit('Value')


def test_tolerant_all_inputs_failed_is_incomplete(context):
    context.tolerate_node_failures = True
    boom = BoomNodeFactory.create(context=context)
    summed = _sum_of(context, [boom])

    df = summed.get_output_pl()
    assert summed.status is NodeStatus.INCOMPLETE
    assert len(df) == 0


def test_incomplete_input_skipped_by_consumer(context):
    context.tolerate_node_failures = True
    good = _leaf(context)
    unwired = AdditiveNodeFactory.create(context=context, input_datasets=[])
    summed = _sum_of(context, [good, unwired])

    df = summed.get_output_pl()
    expected = good.get_output_pl()
    # The unwired (INCOMPLETE) input is treated as absent, not summed as an empty frame.
    assert unwired.status is NodeStatus.INCOMPLETE
    assert summed.status is NodeStatus.DEGRADED
    assert df['Value'].to_list() == expected['Value'].to_list()
