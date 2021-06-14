from typing import Dict
from nodes.instance import Instance


class GQLSession(Dict):
    modified: bool


# Helper classes for typing
class GQLContext:
    instance: Instance
    session: GQLSession


class GQLInfo:
    context: GQLContext
