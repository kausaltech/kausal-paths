# ruff: noqa: UP007
from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Union  # pyright: ignore[reportDeprecated]

import strawberry as sb
from graphql.error import GraphQLError

from paths import gql
from paths.graphql_helpers import get_instance_context

from . import BoolParameter, NumberParameter, Parameter, StringParameter, ValidationError

if TYPE_CHECKING:
    from paths.graphql_types import UnitType

    from nodes.node import Node
    from nodes.scenario import Scenario
    from nodes.schema import NodeInterface, ScenarioType


def _resolve_default_value(info: gql.Info, root: Parameter[Any, Any]) -> Any:
    context = get_instance_context(info)
    scenario: Scenario = context.get_default_scenario()
    if not scenario.has_parameter(root):
        return None
    return scenario.get_parameter_value(root)


def _get_parameter_type_name(instance: Parameter[Any, Any]) -> str:
    type_map: dict[type[Parameter[Any, Any]], str] = {
        BoolParameter: 'BoolParameterType',
        NumberParameter: 'NumberParameterType',
        StringParameter: 'StringParameterType',
    }
    for param_type in type(instance).mro():
        if param_type in type_map:
            return type_map[param_type]
    return 'UnknownParameterType'


def _get_parameter_or_error(info: gql.Info, id: str) -> Parameter[Any, Any]:
    context = get_instance_context(info)
    try:
        return context.get_parameter(id)
    except KeyError:
        raise GraphQLError(f'Parameter {id} does not exist', info.field_nodes) from None


def _get_parameter_value_for_mutation(
    info: gql.Info,
    param: Parameter[Any, Any],
    *,
    number_value: float | None,
    bool_value: bool | None,
    string_value: str | None,
) -> Any:
    parameter_values = {
        NumberParameter: (number_value, 'numberValue'),
        BoolParameter: (bool_value, 'boolValue'),
        StringParameter: (string_value, 'stringValue'),
    }
    param_type = type(param)
    for klass, (value, attr_name) in parameter_values.items():  # noqa: B007
        if issubclass(param_type, klass):
            break
    else:
        msg = f'Attempting to mutate an unsupported parameter class: {type(param)}'
        raise Exception(msg)

    if value is None:
        raise GraphQLError(f"You must specify '{attr_name}' for '{param.global_id}'", info.field_nodes)

    del parameter_values[klass]
    for other_value, _ in parameter_values.values():
        if other_value is not None:
            raise GraphQLError('Only one type of value allowed', info.field_nodes)

    try:
        return param.clean(value)
    except ValidationError as e:
        raise GraphQLError(str(e), info.field_nodes) from e


@sb.interface
class ParameterInterface:
    local_id: sb.ID | None
    is_customized: bool
    is_customizable: bool

    @sb.field(description='Global ID for the parameter in the instance')
    @staticmethod
    def id(root: Parameter[Any, Any]) -> sb.ID:
        return sb.ID(root.global_id)

    @sb.field
    @staticmethod
    def label(root: Parameter[Any, Any]) -> str | None:
        if root.label is None:
            return None
        return str(root.label)

    @sb.field
    @staticmethod
    def description(root: Parameter[Any, Any]) -> str | None:
        if root.description is None:
            return None
        return str(root.description)

    @sb.field(description="ID of parameter in the node's namespace")
    @staticmethod
    def node_relative_id(root: Parameter[Any, Any]) -> sb.ID | None:
        return sb.ID(root.local_id)

    @sb.field(graphql_type=Union[Annotated['NodeInterface', sb.lazy('nodes.schema')], None])  # pyright: ignore[reportDeprecated]
    @staticmethod
    def node(root: Parameter[Any, Any]) -> 'Node | None':
        return root.node

    @staticmethod
    def resolve_type(instance: Parameter[Any, Any], _info: gql.Info, _abstract_type: Any) -> str:
        return _get_parameter_type_name(instance)


@sb.type(name='BoolParameterType')
class BoolParameterType(ParameterInterface):
    @classmethod
    def is_type_of(cls, obj: Any, _info: gql.Info) -> bool:
        return isinstance(obj, BoolParameter)

    value: bool | None

    @sb.field
    @staticmethod
    def default_value(root: BoolParameter, info: gql.Info) -> bool | None:
        return _resolve_default_value(info, root)


@sb.type(name='NumberParameterType')
class NumberParameterType(ParameterInterface):
    @classmethod
    def is_type_of(cls, obj: Any, _info: gql.Info) -> bool:
        return isinstance(obj, NumberParameter)

    value: float | None
    min_value: float | None
    max_value: float | None
    step: float | None
    unit: Annotated['UnitType', sb.lazy('paths.graphql_types')] | None

    @sb.field
    @staticmethod
    def default_value(root: NumberParameter, info: gql.Info) -> float | None:
        return _resolve_default_value(info, root)


@sb.type(name='StringParameterType')
class StringParameterType(ParameterInterface):
    @classmethod
    def is_type_of(cls, obj: Any, _info: gql.Info) -> bool:
        return isinstance(obj, StringParameter)

    value: str | None

    @sb.field
    @staticmethod
    def default_value(root: StringParameter, info: gql.Info) -> str | None:
        return _resolve_default_value(info, root)


@sb.type(name='UnknownParameterType')
class UnknownParameterType(ParameterInterface):
    @classmethod
    def is_type_of(cls, obj: Any, _info: gql.Info) -> bool:
        return isinstance(obj, Parameter) and _get_parameter_type_name(obj) == 'UnknownParameterType'


@sb.type
class SetParameterResult:
    ok: bool
    parameter: Parameter[Any] | None = sb.field(
        graphql_type=Union[Annotated['ParameterInterface', sb.lazy('params.schema')], None]  # pyright: ignore[reportDeprecated]
    )


@sb.type
class ResetParameterResult:
    ok: bool


@sb.type
class ActivateScenarioResult:
    ok: bool
    active_scenario: 'Scenario' = sb.field(graphql_type=Annotated['ScenarioType', sb.lazy('nodes.schema')])


@sb.type
class SBMutation:
    @sb.mutation(description='Set the value of a parameter. Customized parameters are saved in a session.')
    def set_parameter(
        self,
        info: gql.Info,
        id: sb.ID,
        number_value: float | None = None,
        bool_value: bool | None = None,
        string_value: str | None = None,
    ) -> SetParameterResult:
        context = get_instance_context(info)
        param = _get_parameter_or_error(info, str(id))

        if not param.is_customizable:
            raise GraphQLError(f'Parameter {id} is not customizable', info.field_nodes)

        value = _get_parameter_value_for_mutation(
            info,
            param,
            number_value=number_value,
            bool_value=bool_value,
            string_value=string_value,
        )

        setting_storage = context.setting_storage
        assert setting_storage is not None
        setting_storage.set_param(str(id), value)
        setting_storage.set_active_scenario(context.custom_scenario.id)
        context.activate_scenario(context.custom_scenario)

        return SetParameterResult(ok=True, parameter=param)

    @sb.mutation
    def reset_parameter(self, info: gql.Info, id: sb.ID | None = None) -> ResetParameterResult:
        context = get_instance_context(info)
        storage = context.setting_storage
        assert storage is not None
        if id is None:
            storage.reset()
        else:
            storage.reset_param(str(id))

        customized_params = storage.get_customized_param_values()
        if not customized_params:
            default_scenario_id = context.get_default_scenario().id
            active_scenario_id = storage.get_active_scenario()
            if active_scenario_id is not None and active_scenario_id != default_scenario_id:
                storage.set_active_scenario(None)

        return ResetParameterResult(ok=True)

    @sb.mutation
    def activate_scenario(self, info: gql.Info, id: sb.ID) -> ActivateScenarioResult:
        context = get_instance_context(info).instance.context
        scenario = context.scenarios.get(str(id))
        if scenario is None:
            raise GraphQLError(f"Scenario '{id}' not found", info.field_nodes)

        assert context.setting_storage is not None

        default_scenario_id = context.get_default_scenario().id
        if scenario.id == default_scenario_id:
            val = None
        else:
            val = scenario.id
        context.setting_storage.set_active_scenario(val)
        context.activate_scenario(scenario)

        return ActivateScenarioResult(ok=True, active_scenario=scenario)


@sb.type
class SBQuery:
    @sb.field(graphql_type=list[ParameterInterface])
    def parameters(self, info: gql.Info) -> list[Parameter[Any, Any]]:
        context = get_instance_context(info)
        return [param for param in context.global_parameters.values() if param.is_visible]

    @sb.field(graphql_type=Union[Annotated['ParameterInterface', sb.lazy('params.schema')], None])  # pyright: ignore[reportDeprecated]
    def parameter(self, info: gql.Info, id: sb.ID) -> Parameter[Any, Any] | None:
        param = _get_parameter_or_error(info, str(id))
        if not param.is_visible:
            return None
        return param


types = [BoolParameterType, NumberParameterType, StringParameterType, UnknownParameterType]
