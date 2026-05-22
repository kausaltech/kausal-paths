from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

from django.conf import settings
from django.http import HttpRequest
from strawberry.channels import (
    ChannelsRequest,
)

from loguru import logger

from kausal_common.deployment import env_bool
from kausal_common.strawberry.views import GraphQLView, GraphQLWSConsumer, SyncGraphQLHTTPConsumer
from kausal_common.testing.graphql import capture_query

from paths.schema_context import PathsGraphQLContext

if TYPE_CHECKING:
    from django.http.response import HttpResponse
    from strawberry.http import GraphQLRequestData
    from strawberry.http.temporal_response import TemporalResponse
    from strawberry.types import ExecutionResult

    from kausal_common.users import UserOrAnon


GRAPHQL_CAPTURE_QUERIES = env_bool('GRAPHQL_CAPTURE_QUERIES', default=False)

SUPPORTED_LANGUAGES = {x[0] for x in settings.LANGUAGES}

logger = logger.bind(markup=True)


def graphql_capture_query(
    view: PathsGraphQLHTTPConsumer | PathsGraphQLView,
    request: ChannelsRequest | HttpRequest,
    context: PathsGraphQLContext,
    request_data: GraphQLRequestData,
    response: ExecutionResult,
    exec_time: float,
):
    from paths.const import INSTANCE_HOSTNAME_HEADER, INSTANCE_IDENTIFIER_HEADER, WILDCARD_DOMAINS_HEADER

    if not GRAPHQL_CAPTURE_QUERIES:
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
