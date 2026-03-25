from __future__ import annotations

import json
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Self, cast

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from kausal_common.i18n.pydantic import I18nString, TranslatedString

from common.types import ParameterLocalId
from nodes.exceptions import ParameterError
from nodes.node import Node
from nodes.units import Unit

from .registry import register_parameter_type

if TYPE_CHECKING:
    from nodes.context import Context  # noqa: TC004
    from nodes.instance import Instance
    from nodes.scenario import Scenario


def parameter[PT: Parameter[Any, Any]](cls: type[PT]) -> type[PT]:
    register_parameter_type(cls)
    return cls


class Parameter[ValueT = Any, SetValueT = ValueT](BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    local_id: ParameterLocalId
    type: str = ''  # Discriminator — overridden with a Literal in each subclass

    label: I18nString | None = None
    description: I18nString | None = None

    is_customized: bool = False
    is_customizable: bool = True
    is_visible: bool | None = None

    # Runtime-only fields (excluded from serialization)
    _node: Node | None = PrivateAttr(default=None)
    """Set if this parameter is bound to a specific node."""

    _subscription_nodes: list[Node] = PrivateAttr(default_factory=list)
    """Nodes that should be notified when the parameter changes value."""

    _subscription_params: list[Parameter[Any, Any]] = PrivateAttr(default_factory=list)
    """Parameters that should be notified when the parameter changes value."""

    _context: Context | None = PrivateAttr(default=None)
    """The context to which this parameter is bound."""

    _hash: str | None = PrivateAttr(default=None)

    @property
    def context(self) -> Context | None:
        return self._context

    @property
    def instance(self) -> Instance:
        if self.context is None:
            raise ParameterError(self, 'Parameter is not bound to a context')
        return self.context.instance

    @property
    def node(self) -> Node | None:
        return self._node

    def _convert_lazy_strings(self) -> None:
        from django.utils.functional import Promise

        if self.label is not None and isinstance(self.label, Promise):
            self.label = TranslatedString.from_lazy_string(self.label)
        if self.description is not None and isinstance(self.description, Promise):
            self.description = TranslatedString.from_lazy_string(self.description)

    def model_post_init(self, /, __context: Any) -> None:
        assert '.' not in self.local_id
        if self.is_visible is None:
            if self.is_customizable:
                self.is_visible = True
            else:
                self.is_visible = False

        self._convert_lazy_strings()

    def copy(self, **kwargs: Any) -> Self:
        return self.model_copy(update=kwargs)

    def subscribe_changes(self, target: Parameter[Any] | Node) -> None:
        if isinstance(target, Parameter):
            param = target
            if param in self._subscription_params:
                raise ParameterError(self, f'Parameter {param.global_id} already subscribed to changes')
            self._subscription_params.append(cast('Self', param))
        elif isinstance(target, Node):
            node = target
            if node in self._subscription_nodes:
                raise ParameterError(self, f'Node {node.id} already subscribed to changes')
            self._subscription_nodes.append(node)
        else:
            raise TypeError(f'Unknown target type: {type(target)}')

    def notify_change(self) -> None:
        self._hash = None
        if self._node:
            self._node.notify_parameter_change(self)
        for node in self._subscription_nodes:
            node.notify_parameter_change(self)
        for param in self._subscription_params:
            param.notify_change()

    @contextmanager
    def override(self, value: SetValueT):
        prev_val = self.value
        self.set(value)
        yield
        self.set(prev_val)

    def set(self, value: SetValueT, notify: bool = True):
        prev_val = getattr(self, 'value', None)
        self.value = self.clean(value)
        if notify and not self.is_value_equal(value=prev_val):
            self.notify_change()

    def reset_to_scenario_setting(self, scenario: Scenario, value: SetValueT):  # pyright: ignore[reportUnusedParameter]
        self.set(value)
        self.is_customized = False

    def get(self) -> ValueT:
        return self.value

    def serialize_value(self) -> Any:
        if isinstance(self.value, BaseModel):
            return self.value.model_dump(mode='json')
        return self.value

    def is_value_equal(self, value: Any) -> bool:
        return self.value == value

    def calculate_hash(self) -> str:
        h = self._hash
        if h is not None:
            return h
        if isinstance(self.value, str):
            v = self.value.encode('unicode_escape').decode('ascii')
        elif isinstance(self.value, bool | int | float):
            v = str(self.value)
        else:
            v = json.dumps({'value': self.serialize_value()}, ensure_ascii=True)

        h = '%s:%s' % (self.global_id, v)
        self._hash = h
        return h

    def clean(self, value: Any) -> Any:  # pyright: ignore[reportUnusedParameter]
        raise NotImplementedError('Implement in subclass')

    @property
    def global_id(self):
        if self._node is None:
            return self.local_id
        return f'{self._node.id}.{self.local_id}'

    def set_node(self, node: Node):
        if self._node is not None:
            msg = f'Node for parameter {self.global_id} already set'
            raise Exception(msg)
        self._node = node
        if self._context is not None:
            assert self._context == node.context
            assert self not in self._context.global_parameters.values()
        else:
            self.set_context(node.context)

    def set_context(self, context: Context):
        if self._context is not None:
            msg = f'Context for parameter {self.global_id} already set'
            raise Exception(msg)
        self._context = context

    def has_unit(self) -> bool:
        if isinstance(self, ParameterWithUnit) and self.unit is not None:
            return True
        return False

    def get_unit(self) -> Unit:
        raise ParameterError(self, 'Parameter does not have units' % self.global_id)


class ParameterWithUnit[ValueT, SetValueT = ValueT](Parameter[ValueT, SetValueT]):
    unit: Unit | None = Field(default=None, exclude=False)
    unit_str: str | None = Field(default=None, exclude=True, repr=False)

    def _init_unit(self) -> None:
        if self.unit_str is not None:
            from nodes.units import unit_registry

            self.unit = unit_registry.parse_units(self.unit_str)

        if self.unit is not None and not isinstance(self.unit, Unit):
            raise Exception('str given for unit for parameter %s' % self.local_id)

    def get_unit(self) -> Unit:
        if not self.has_unit():
            msg = f'Parameter {self.global_id} does not have units'
            raise Exception(msg)
        return cast('Unit', self.unit)

    def model_post_init(self, /, __context: Any) -> None:
        self._init_unit()
        super().model_post_init(__context)
