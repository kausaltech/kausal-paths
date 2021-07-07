import pytest

from params.tests.factories import ParameterFactory
from nodes.tests.factories import ContextFactory, NodeFactory

pytestmark = pytest.mark.django_db


def test_context_get_param_global():
    context = ContextFactory()
    param = ParameterFactory()
    context.add_global_parameter(param)
    assert param.global_id == param.local_id
    assert context.get_param(param.global_id) == param


def test_context_get_param_local():
    context = ContextFactory()
    node = NodeFactory()
    context.add_node(node)
    param = ParameterFactory()
    node.add_parameter(param)
    assert param.global_id != param.local_id
    assert context.get_param(param.global_id) == param
