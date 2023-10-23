import pytest

from nodes.tests.factories import NodeFactory

pytestmark = pytest.mark.django_db


def test_node_get_downstream_nodes(context, node):
    output_node = NodeFactory(context=context)
    node.add_output_node(output_node)
    expected = [output_node]
    assert node.get_downstream_nodes() == expected


def test_node_get_upstream_nodes(context, node):
    input_node = NodeFactory(context=context)
    node.add_input_node(input_node)
    expected = [input_node]
    assert node.get_upstream_nodes() == expected
