from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import strawberry as sb

import pytest

from paths.graphql_helpers import pass_context

from nodes.context import Context


def _context(marker: str) -> Context:
    context = Context.__new__(Context)
    context.marker = marker  # type: ignore[attr-defined]
    return context


@sb.type
class BoundItem:
    """A root object bound to a runtime, the way a node or a scenario is."""

    context: sb.Private[Any]

    @sb.field
    @staticmethod
    @pass_context
    def marker(root: Any, context: Any) -> str:
        return context.marker


@pytest.mark.django_db
def test_pass_context_takes_the_context_of_the_root():
    @sb.type
    class Query:
        @sb.field
        def first(self) -> BoundItem:
            return BoundItem(context=_context('from root'))

    # What the request names must not be what the resolver sees.
    request_context = SimpleNamespace(instance=SimpleNamespace(context=_context('from request')))
    result = sb.Schema(query=Query).execute_sync('{ first { marker } }', context_value=request_context)

    assert result.errors is None
    assert result.data == {'first': {'marker': 'from root'}}


@pytest.mark.django_db
def test_pass_context_refuses_a_root_without_a_runtime():
    @sb.type
    class Query:
        @sb.field
        @pass_context
        def marker(self, context: Any) -> str:
            return context.marker

    result = sb.Schema(query=Query).execute_sync('{ marker }', root_value=object())

    assert result.errors is not None
    assert 'is not bound to a model runtime' in result.errors[0].message


@pytest.mark.django_db
def test_pass_context_requires_a_root_parameter():
    with pytest.raises(TypeError, match='take the runtime from their root'):

        @pass_context  # type: ignore[arg-type]  # pyright: ignore[reportCallIssue, reportArgumentType]
        def resolver(context: Any) -> str:  # pyright: ignore[reportUnusedFunction]
            return context.marker


@pytest.mark.django_db
def test_pass_context_rejects_strawberry_field_objects():
    with pytest.raises(TypeError, match=r'pass_context must wrap the resolver function before @sb\.field'):

        @sb.type
        class Query:  # pyright: ignore[reportUnusedClass]
            @pass_context  # type: ignore[arg-type]
            @sb.field
            def wrong_order(self, context: Any) -> str:
                return context.marker
