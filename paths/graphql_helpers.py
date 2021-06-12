from nodes.instance import Instance


# Helper classes for typing
class GQLContext:
    instance: Instance


class GQLInfo:
    context: GQLContext
