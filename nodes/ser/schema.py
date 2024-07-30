import inspect
from typing import Any, TypeGuard
from pydantic import BaseModel
import strawberry

from .i18n_ser import I18nString
from nodes.models import InstanceConfig
from . import node_ser
from .node_ser import EdgeNodePort, EdgeSer, PathsModelConfig, NodeSer



"""
@strawberry.experimental.pydantic.type(model=I18nString, all_fields=True)
class I18nStringType:
    @strawberry.field
    @staticmethod
    def i18n(root: I18nString) -> list[tuple[str, str]]:
        return [(lang, val) for lang, val in root.i18n.items()]
"""

I18nStringType = strawberry.scalar(
    I18nString,
    serialize=lambda x: x.get_value(),
)

I18nString._strawberry_type = I18nStringType  # type: ignore


def should_register(val: Any) -> TypeGuard[type[BaseModel]]:
    if not inspect.isclass(val) or val is BaseModel or inspect.isabstract(val):
        return False
    if not issubclass(val, BaseModel):
        return False
    if val is PathsModelConfig:
        return False
    if val in (I18nString, NodeSer, EdgeNodePort, EdgeSer):
        return False
    return True


for key, val in vars(node_ser).items():
    if not should_register(val):
        continue
    if key.endswith('Ser'):
        key = key.removesuffix('Ser')
    kls = type(key, (), {})
    globals()[key] = strawberry.experimental.pydantic.type(model=val, all_fields=True)(kls)



@strawberry.experimental.pydantic.type(model=NodeSer, all_fields=True)
class NodeType:
    pass


@strawberry.experimental.pydantic.type(model=EdgeNodePort, all_fields=True)
class EdgeNodePortType:
    @strawberry.field
    @staticmethod
    def node(root: EdgeNodePort) -> NodeType:
        return NodeType.from_pydantic(root._node)


@strawberry.experimental.pydantic.type(model=EdgeSer, all_fields=True)
class EdgeType:
    pass


@strawberry.experimental.pydantic.type(model=PathsModelConfig, all_fields=True)
class PathsModelConfigType:
    pass


_configs: dict[str, PathsModelConfigType] = {}


@strawberry.type
class NodesQuery:
    @strawberry.field
    def model_config(self, id: strawberry.ID) -> PathsModelConfigType:
        if id in _configs:
            return _configs[id]

        ic = InstanceConfig.objects.get(identifier=id)
        pmc = PathsModelConfig.from_instance(ic.get_instance())
        return PathsModelConfigType.from_pydantic(pmc)  # type: ignore
