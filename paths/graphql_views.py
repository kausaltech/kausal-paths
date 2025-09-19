from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, cast

from django.conf import settings
from graphql.execution import ExecutionContext
from graphql.language.ast import VariableNode

from loguru import logger

from kausal_common.deployment import env_bool
from kausal_common.strawberry.views import GraphQLView, GraphQLWSConsumer, SyncGraphQLHTTPConsumer

from paths.schema_context import PathsGraphQLContext

from .graphql_helpers import GraphQLPerfNode

if TYPE_CHECKING:
    from django.http import HttpRequest
    from django.http.response import HttpResponse
    from graphql import (
        GraphQLOutputType,
    )
    from graphql.error import GraphQLError
    from graphql.language import FieldNode
    from graphql.pyutils import AwaitableOrValue, Path
    from graphql.type import GraphQLObjectType
    from strawberry.channels import (
        ChannelsRequest,
    )
    from strawberry.http.temporal_response import TemporalResponse

    from kausal_common.users import UserOrAnon

    from paths.types import GQLInstanceContext


SUPPORTED_LANGUAGES = {x[0] for x in settings.LANGUAGES}


def _arg_value(arg, variable_vals) -> Any:
    if isinstance(arg.value, VariableNode):
        return variable_vals.get(arg.value.name.value)
    return arg.value.value


logger = logger.bind(markup=True)

GRAPHQL_CAPTURE_QUERIES = env_bool('GRAPHQL_CAPTURE_QUERIES', default=False)


# FIXME: Not used anywhere; any code worth keeping?
class PathsExecutionContext(ExecutionContext):
    context_value: GQLInstanceContext

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def handle_field_error(
        self,
        error: GraphQLError,
        return_type: GraphQLOutputType,
    ) -> None:
        if settings.DEBUG and error.original_error is not None and not getattr(error, '_was_printed', False):
            exc = error.original_error
            logger.opt(exception=exc).error('GraphQL field error at {path}', path=error.path)
            setattr(error, '_was_printed', True)  # noqa: B010
        return super().handle_field_error(error, return_type)

    def execute_fields(
        self,
        parent_type: GraphQLObjectType,
        source_value: Any,
        path: Path | None,
        fields: dict[str, list[FieldNode]],
    ) -> AwaitableOrValue[dict[str, Any]]:
        path_parts = path.as_list() if path else []
        span_cm: AbstractContextManager
        str_path = '.'.join([str(x) for x in path_parts])
        if path_parts:
            node = GraphQLPerfNode(str_path)
            span_cm = self.context_value.graphql_perf.exec_node(node)
        else:
            span_cm = nullcontext()
        with span_cm:
            _rich_traceback_omit = True
            ret = super().execute_fields(parent_type, source_value, path, fields)
        return ret


class PathsGraphQLWSConsumer(GraphQLWSConsumer[PathsGraphQLContext]):
    async def get_context(self, request: GraphQLWSConsumer, response: GraphQLWSConsumer) -> PathsGraphQLContext:
        base_ctx = await self.get_base_context(request, response)
        return PathsGraphQLContext(
            **base_ctx,
        )


class PathsGraphQLHTTPConsumer(SyncGraphQLHTTPConsumer[PathsGraphQLContext]):
    def get_context(self, request: ChannelsRequest, response: TemporalResponse) -> PathsGraphQLContext:
        base_ctx = self.get_base_context(request, response)
        return PathsGraphQLContext(
            **base_ctx,
        )


class PathsGraphQLView(GraphQLView[PathsGraphQLContext]):
    context_class: type[PathsGraphQLContext] = PathsGraphQLContext

    def __init__(self):
        from .schema import schema

        super().__init__(schema=schema)

    def get_context(self, request: HttpRequest, response: HttpResponse) -> PathsGraphQLContext:
        from paths.context import PathsObjectCache

        base_ctx = super().get_base_context(request, response)
        context = PathsGraphQLContext(
            **base_ctx,
        )
        user = cast('UserOrAnon', request.user)
        context.cache = PathsObjectCache(user=user)
        return context
