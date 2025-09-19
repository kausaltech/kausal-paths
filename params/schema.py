from __future__ import annotations

from typing import TYPE_CHECKING, Any

import graphene
from graphql.error import GraphQLError

from paths.graphql_helpers import ensure_instance

from . import BoolParameter, NumberParameter, Parameter, PercentageParameter, StringParameter, ValidationError

if TYPE_CHECKING:
    from kausal_common.graphene import GQLInfo

    from paths.graphql_helpers import GQLInstanceInfo


class ResolveDefaultValueMixin:
    @staticmethod
    def resolve_default_value(root: Parameter, info: GQLInstanceInfo) -> Any:
        context = info.context.instance.context
        scenario = context.get_default_scenario()
        if not scenario.has_parameter(root):
            return None
        return scenario.get_parameter_value(root)


class ParameterInterface(graphene.Interface):
    id = graphene.ID(required=True, description='Global ID for the parameter in the instance')
    local_id = graphene.ID(required=False, description="ID of parameter in the node's namespace")
    label = graphene.String(required=False)
    description = graphene.String(required=False)
    node_relative_id = graphene.ID(required=False)  # can be null if node is null
    node = graphene.Field('nodes.schema.NodeInterface', required=False)  # can be null for global parameters
    is_customized = graphene.Boolean(required=True)
    is_customizable = graphene.Boolean(required=True)

    # TODO: Use the proper field names instead of defining this alias?
    @staticmethod
    def resolve_id(root: Parameter, info) -> str:
        return root.global_id

    # TODO: Use the proper field names instead of defining this alias?
    @staticmethod
    def resolve_node_relative_id(root: Parameter, info) -> str:
        return root.local_id

    @classmethod
    def resolve_type(cls, instance: Parameter, info: GQLInfo) -> type[graphene.ObjectType]:  # noqa: ARG003
        type_map = {
            BoolParameter: BoolParameterType,
            NumberParameter: NumberParameterType,
            StringParameter: StringParameterType,
            PercentageParameter: NumberParameterType,
        }
        # Try to find the parameter type by going through the superclasses
        # of the parameter instance.
        for param_type in type(instance).mro():
            if param_type in type_map:
                return type_map[param_type]
        return UnknownParameterType


class BoolParameterType(ResolveDefaultValueMixin, graphene.ObjectType):
    class Meta:
        interfaces = (ParameterInterface,)

    value = graphene.Boolean()
    default_value = graphene.Boolean()


class NumberParameterType(ResolveDefaultValueMixin, graphene.ObjectType):
    class Meta:
        interfaces = (ParameterInterface,)

    value = graphene.Float()
    default_value = graphene.Float()
    min_value = graphene.Float()
    max_value = graphene.Float()
    step = graphene.Float()
    unit = graphene.Field('paths.schema.UnitType')


class StringParameterType(ResolveDefaultValueMixin, graphene.ObjectType):
    class Meta:
        interfaces = (ParameterInterface,)

    value = graphene.String()
    default_value = graphene.String()


class UnknownParameterType(graphene.ObjectType):
    class Meta:
        interfaces = (ParameterInterface,)


class SetParameterMutation(graphene.Mutation):
    class Arguments:
        id = graphene.ID(required=True)
        number_value = graphene.Float()
        bool_value = graphene.Boolean()
        string_value = graphene.String()

    ok = graphene.Boolean()
    parameter = graphene.Field(ParameterInterface)

    @ensure_instance
    @staticmethod
    def mutate(
        root,
        info: GQLInstanceInfo,
        id: str,
        number_value: float | None = None,
        bool_value: bool | None = None,
        string_value: str | None = None,
    ) -> SetParameterMutation:
        context = info.context.instance.context
        try:
            param = context.get_parameter(id)
        except KeyError:
            raise GraphQLError('Parameter %s does not exist', info.field_nodes) from None

        if not param.is_customizable:
            raise GraphQLError('Parameter %s is not customizable', info.field_nodes)

        parameter_values = {
            NumberParameter: (number_value, 'numberValue'),
            BoolParameter: (bool_value, 'boolValue'),
            StringParameter: (string_value, 'stringValue'),
        }
        param_type = type(param)
        for klasses, (value, attr_name) in parameter_values.items():  # noqa: B007
            # if isinstance(klasses, tuple):
            #     found = False
            #     for k in klasses:
            #         if issubclass(param_type, k):
            #             found = True
            #             break
            #     if found:
            #         break
            #     continue
            if issubclass(param_type, klasses):
                break
        else:
            raise Exception('Attempting to mutate an unsupported parameter class: %s' % type(param))

        if value is None:
            raise GraphQLError("You must specify '%s' for '%s'" % (attr_name, param.global_id), info.field_nodes)

        del parameter_values[klasses]
        for v, _ in parameter_values.values():
            if v is not None:
                raise GraphQLError('Only one type of value allowed', info.field_nodes)

        try:
            value = param.clean(value)
        except ValidationError as e:
            raise GraphQLError(str(e), info.field_nodes)  # noqa: B904

        setting_storage = info.context.instance.context.setting_storage
        assert setting_storage is not None
        setting_storage.set_param(id, value)
        setting_storage.set_active_scenario(context.custom_scenario.id)
        context.activate_scenario(context.custom_scenario)

        return SetParameterMutation(ok=True, parameter=param)


class ResetParameterMutation(graphene.Mutation):
    class Arguments:
        id = graphene.ID()

    ok = graphene.Boolean()

    @staticmethod
    @ensure_instance
    def mutate(root, info: GQLInstanceInfo, id: str | None = None) -> ResetParameterMutation:
        context = info.context.instance.context
        storage = context.setting_storage
        assert storage is not None
        if id is None:
            # Reset all parameters to defaults
            storage.reset()
        else:
            storage.reset_param(id)

        customized_params = storage.get_customized_param_values()
        if not customized_params:
            # If we no longer have customized parameters, activate the default scenario
            default_scenario_id = context.get_default_scenario().id
            active_scenario_id = storage.get_active_scenario()
            if active_scenario_id is not None and active_scenario_id != default_scenario_id:
                storage.set_active_scenario(None)

        return ResetParameterMutation(ok=True)


class ActivateScenarioMutation(graphene.Mutation):
    class Arguments:
        id = graphene.ID(required=True)

    ok = graphene.Boolean()
    active_scenario = graphene.Field('nodes.schema.ScenarioType')

    @staticmethod
    @ensure_instance
    def mutate(root, info: GQLInstanceInfo, id: str) -> dict[str, Any]:
        context = info.context.instance.context
        scenario = context.scenarios.get(id)
        if scenario is None:
            raise GraphQLError("Scenario '%s' not found" % id, info.field_nodes)

        assert context.setting_storage is not None

        default_scenario_id = context.get_default_scenario().id
        if scenario.id == default_scenario_id:
            val = None
        else:
            val = scenario.id
        context.setting_storage.set_active_scenario(val)
        context.activate_scenario(scenario)

        return dict(ok=True, active_scenario=scenario)


class Mutations(graphene.ObjectType):
    set_parameter = SetParameterMutation.Field()
    reset_parameter = ResetParameterMutation.Field()
    activate_scenario = ActivateScenarioMutation.Field()


class Query(graphene.ObjectType):
    parameters = graphene.List(graphene.NonNull(ParameterInterface), required=True)
    parameter = graphene.Field(ParameterInterface, id=graphene.ID(required=True))

    @staticmethod
    @ensure_instance
    def resolve_parameters(root, info: GQLInstanceInfo) -> list[Parameter[Any]]:
        instance = info.context.instance
        params = [param for param in instance.context.global_parameters.values() if param.is_visible]
        return params

    @staticmethod
    @ensure_instance
    def resolve_parameter(root, info: GQLInstanceInfo, id: str) -> Parameter[Any] | None:
        instance = info.context.instance
        try:
            param = instance.context.get_parameter(id)
            if not param.is_visible:
                return None
        except KeyError:
            raise GraphQLError(f'Parameter {id} does not exist', info.field_nodes) from None
        return param


types = [BoolParameterType, NumberParameterType, StringParameterType, UnknownParameterType]
