from typing import Optional
from django.core.handlers.wsgi import WSGIRequest
from graphql.execution import ResolveInfo
from graphql.language.ast import OperationDefinition
from graphql.error import GraphQLError

from nodes.instance import Instance


# Helper classes for typing
class GQLContext(WSGIRequest):
    instance: Optional[Instance]
    graphql_query_language: str


class GQLInfo(ResolveInfo):
    context: GQLContext
    operation: OperationDefinition


def ensure_instance(method):
    """Wrap a class method to ensure instance is specified when the method is called."""

    def method_wrapper(self, info: GQLInfo, *args, **kwargs):
        if info.context.instance is None:
            raise GraphQLError(
                "Unable to determine Paths instance for the request. Use the 'instance' directive or HTTP headers.",
                [info]
            )
        return method(self, info, *args, **kwargs)

    return method_wrapper
