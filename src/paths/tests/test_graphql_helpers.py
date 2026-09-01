from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import strawberry as sb

import pytest

from paths.graphql_helpers import pass_context


@pytest.mark.django_db
def test_pass_context_allows_strawberry_resolvers_without_info_or_root():
    @sb.type
    class Query:
        @sb.field
        @pass_context
        def instance_field(self, context: Any) -> str:
            return context.marker

        @sb.field
        @staticmethod
        @pass_context
        def static_field(context: Any) -> str:
            return context.marker

    schema = sb.Schema(query=Query)
    context = SimpleNamespace(instance=SimpleNamespace(context=SimpleNamespace(marker='ok')))
    result = schema.execute_sync('{ instanceField staticField }', context_value=context)

    assert result.errors is None
    assert result.data == {
        'instanceField': 'ok',
        'staticField': 'ok',
    }


@pytest.mark.django_db
def test_pass_context_rejects_strawberry_field_objects():
    with pytest.raises(TypeError, match=r'pass_context must wrap the resolver function before @sb\.field'):

        @sb.type
        class Query:  # pyright: ignore[reportUnusedClass]
            @pass_context  # type: ignore[arg-type]
            @sb.field
            def wrong_order(self, context: Any) -> str:
                return context.marker
