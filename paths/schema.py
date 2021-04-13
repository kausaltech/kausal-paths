import graphene


class Query(graphene.ObjectType):
    graph = graphene.String()

    def resolve_graph(root, info):
        return 'TODO'


schema = graphene.Schema(
    query=Query,
)
