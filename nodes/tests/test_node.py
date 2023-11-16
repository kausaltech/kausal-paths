import pytest
from nodes.context import Context

from nodes.tests.factories import NodeFactory

pytestmark = pytest.mark.django_db


def test_node_get_downstream_nodes(context: Context, node):
    output_node = NodeFactory(context=context)
    node.add_output_node(output_node)
    context.finalize_nodes()
    expected = [output_node]
    assert node.get_downstream_nodes() == expected


def test_node_get_upstream_nodes(context, node):
    input_node = NodeFactory(context=context)
    node.add_input_node(input_node)
    expected = [input_node]
    assert node.get_upstream_nodes() == expected
