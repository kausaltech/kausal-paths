import strawberry
from strawberry.tools import merge_types
from nodes.ser.schema import NodesQuery


Query = merge_types("Query", (NodesQuery,))
schema = strawberry.Schema(
    query=Query,
)
