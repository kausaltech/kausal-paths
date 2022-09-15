from django.core.handlers.wsgi import WSGIRequest
from graphql.type import GraphQLResolveInfo
from graphql.language.ast import OperationDefinitionNode
from graphql.error import GraphQLError

from nodes.instance import Instance


# Helper classes for typing
class GQLContext(WSGIRequest):
    graphql_query_language: str


class GQLInstanceContext(GQLContext):
    instance: Instance


class GQLInfo(GraphQLResolveInfo):
    context: GQLContext
    operation: OperationDefinitionNode


class GQLInstanceInfo(GQLInfo):
    context: GQLInstanceContext


def ensure_instance(method):
    """Wrap a class method to ensure instance is specified when the method is called."""

    def method_wrapper(self, info: GQLInstanceInfo, *args, **kwargs):
        if info.context.instance is None:
            raise GraphQLError(
                "Unable to determine Paths instance for the request. Use the 'instance' directive or HTTP headers.",
                [info]
            )
        return method(self, info, *args, **kwargs)

    return method_wrapper
