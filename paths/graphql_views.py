from __future__ import annotations

import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, cast

from django.conf import settings
from django.http import HttpRequest
from graphql.execution import ExecutionContext
from strawberry.channels import (
    ChannelsRequest,
)

from loguru import logger

from kausal_common.deployment import env_bool
from kausal_common.strawberry.views import GraphQLView, GraphQLWSConsumer, SyncGraphQLHTTPConsumer
from kausal_common.testing.graphql import capture_query

from paths.schema_context import PathsGraphQLContext

from .graphql_helpers import GraphQLPerfNode

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from django.http.response import HttpResponse
    from graphql import (
        GraphQLOutputType,
    )
    from graphql.error import GraphQLError
    from graphql.language import FieldNode
    from graphql.pyutils import AwaitableOrValue, Path
    from graphql.type import GraphQLObjectType
    from strawberry.http import GraphQLRequestData
    from strawberry.http.temporal_response import TemporalResponse
    from strawberry.types import ExecutionResult

    from kausal_common.users import UserOrAnon

    from paths.types import GQLInstanceContext


SUPPORTED_LANGUAGES = {x[0] for x in settings.LANGUAGES}

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
        span_cm: AbstractContextManager[Any]
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


def graphql_capture_query(
    view: PathsGraphQLHTTPConsumer | PathsGraphQLView,
    request: ChannelsRequest | HttpRequest,
    context: PathsGraphQLContext,
    request_data: GraphQLRequestData,
    response: ExecutionResult,
    exec_time: float,
):
    from paths.const import INSTANCE_HOSTNAME_HEADER, INSTANCE_IDENTIFIER_HEADER, WILDCARD_DOMAINS_HEADER
    if not env_bool('GRAPHQL_CAPTURE_QUERIES', default=False):
        return
    headers = [INSTANCE_IDENTIFIER_HEADER, INSTANCE_HOSTNAME_HEADER, WILDCARD_DOMAINS_HEADER]
    http_headers = context.get_request_headers()
    instance_id = http_headers.get(INSTANCE_IDENTIFIER_HEADER.lower())
    if isinstance(view, PathsGraphQLHTTPConsumer):
        assert isinstance(request, ChannelsRequest)
        processed_response = view.process_result(request, response)
        capture_query(context, headers, request_data, processed_response, exec_time, instance_id=instance_id)
    else:
        assert isinstance(request, HttpRequest)
        processed_response = view.process_result(request, response)  # type: ignore[arg-type]
        capture_query(context, headers, request_data, processed_response, exec_time, instance_id=instance_id)


class PathsGraphQLWSConsumer(GraphQLWSConsumer[PathsGraphQLContext]):
    async def get_context(self, request: GraphQLWSConsumer, response: GraphQLWSConsumer) -> PathsGraphQLContext:
        base_ctx = await self.get_base_context(request, response)
        return PathsGraphQLContext(
            **base_ctx,
        )


class PathsGraphQLHTTPConsumer(SyncGraphQLHTTPConsumer[PathsGraphQLContext]):
    def execute_single(
        self,
        request: ChannelsRequest,
        request_adapter: Any,
        sub_response: Any,
        context: PathsGraphQLContext,
        root_value: Any,
        request_data: GraphQLRequestData,
    ) -> ExecutionResult:
        start = time.time()
        response = super().execute_single(request, request_adapter, sub_response, context, root_value, request_data)
        graphql_capture_query(self, request, context, request_data, response, (time.time() - start) * 1000)
        return response
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

    def execute_single(
        self,
        request: HttpRequest,
        request_adapter: Any,
        sub_response: Any,
        context: PathsGraphQLContext,
        root_value: Any,
        request_data: GraphQLRequestData,
    ) -> ExecutionResult:
        start = time.time()
        response = super().execute_single(request, request_adapter, sub_response, context, root_value, request_data)
        graphql_capture_query(self, request, context, request_data, response, (time.time() - start) * 1000)
        return response

    def get_context(self, request: HttpRequest, response: HttpResponse) -> PathsGraphQLContext:
        from paths.context import PathsObjectCache

        base_ctx = super().get_base_context(request, response)
        context = PathsGraphQLContext(
            **base_ctx,
        )
        user = cast('UserOrAnon', request.user)
        context.cache = PathsObjectCache(user=user)
        return context
