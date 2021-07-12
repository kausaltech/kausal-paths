import pytest

from nodes.tests.factories import NodeFactory


@pytest.mark.parametrize('proper', [True, False])
def test_node_get_descendant_nodes(context, node, proper):
    output_node = NodeFactory()
    node.add_output_node(output_node)
    if proper:
        expected = [output_node]
    else:
        expected = [node, output_node]
    assert node.get_descendant_nodes(proper) == expected
