from typing import Any
import graphene
from graphql.error import GraphQLError

from paths.graphql_helpers import GQLInfo

from . import (
    BoolParameter, NumberParameter, Parameter, PercentageParameter, StringParameter,
    ValidationError
)


class ParameterInterface(graphene.Interface):
    id = graphene.ID()  # global id
    name = graphene.String()
    description = graphene.String()
    node_relative_id = graphene.ID()  # can be null if node is null
    node = graphene.Field('nodes.schema.NodeType')  # can be null for global params
    is_customized = graphene.Boolean()
    is_customizable = graphene.Boolean()

    @classmethod
    def resolve_type(cls, parameter, info):
        type_map = {
            BoolParameter: BoolParameterType,
            NumberParameter: NumberParameterType,
            StringParameter: StringParameterType,
            PercentageParameter: NumberParameterType,
        }
        # Try to find the parameter type by going through the superclasses
        # of the parameter instance.
        for param_type in type(parameter).mro():
            if param_type in type_map:
                return type_map[param_type]
        raise Exception(f"{parameter} has invalid type")

    def resolve_default_value(root: Parameter, info: GQLInfo) -> Any:
        context = info.context.instance.context
        scenario = context.get_default_scenario()
        param = scenario.params.get(root.id)
        if param is None:
            return None
        return param.value


class BoolParameterType(graphene.ObjectType):
    class Meta:
        interfaces = (ParameterInterface,)

    value = graphene.Boolean()
    default_value = graphene.Boolean()


class NumberParameterType(graphene.ObjectType):
    class Meta:
        interfaces = (ParameterInterface,)

    value = graphene.Float()
    default_value = graphene.Float()
    min_value = graphene.Float()
    max_value = graphene.Float()
    step = graphene.Float()
    unit = graphene.Field('paths.schema.UnitType')


class StringParameterType(graphene.ObjectType):
    class Meta:
        interfaces = (ParameterInterface,)

    value = graphene.String()
    default_value = graphene.String()


class SetParameterMutation(graphene.Mutation):
    class Arguments:
        id = graphene.ID(required=True)
        number_value = graphene.Float()
        bool_value = graphene.Boolean()
        string_value = graphene.String()

    ok = graphene.Boolean()
    parameter = graphene.Field(ParameterInterface)

    def mutate(root, info: GQLInfo, id, number_value=None, bool_value=None, string_value=None):
        context = info.context.instance.context
        try:
            param = context.params[id]
        except KeyError:
            raise GraphQLError("Parameter %s does not exist", [info])

        if not param.is_customizable:
            raise GraphQLError("Parameter %s is not customizable", [info])

        parameter_values = {
            NumberParameter: (number_value, 'numberValue'),
            BoolParameter: (bool_value, 'boolValue'),
            StringParameter: (string_value, 'stringValue'),
        }
        p = parameter_values.pop(type(param), None)
        if p is None:
            raise Exception("Attempting to mutate an unsupported parameter class: %s" % type(param))
        value, attr_name = p
        if value is None:
            raise GraphQLError("You must specify '%s' for '%s'" % (attr_name, param.id))

        for v, _ in parameter_values.values():
            if v is not None:
                raise GraphQLError("Only one type of value allowed", [info])

        try:
            value = param.clean(value)
        except ValidationError as e:
            raise GraphQLError(str(e), [info])

        session = info.context.session
        session_params = session.setdefault('params', {})
        session_params[id] = value

        custom_scenario = context.scenarios['custom']
        custom_scenario.set_session(session)
        context.activate_scenario(custom_scenario)
        session['active_scenario'] = 'custom'
        # Explicitly mark session as modified because we might only have modified `session['params']`, not `session`
        session.modified = True

        return SetParameterMutation(ok=True, parameter=param)


class ResetParameterMutation(graphene.Mutation):
    class Arguments:
        id = graphene.ID()

    ok = graphene.Boolean()

    def mutate(root, info: GQLInfo, id=None):
        context = info.context.instance.context
        session = info.context.session
        if id is None:
            # Reset all parameters to defaults
            session.pop('params', None)
        else:
            params = session.get('params', {})
            params.pop(id, None)

        params = session.get('params', {})
        if not params:
            session['active_scenario'] = context.get_default_scenario().id

        info.context.session.modified = True
        return ResetParameterMutation(ok=True)


class Mutations(graphene.ObjectType):
    set_parameter = SetParameterMutation.Field()
    reset_parameter = ResetParameterMutation.Field()


class Query(graphene.ObjectType):
    parameters = graphene.List(ParameterInterface)
    parameter = graphene.Field(ParameterInterface, id=graphene.ID(required=True))

    def resolve_parameters(root, info: GQLInfo):
        instance = info.context.instance
        return instance.context.params.values()

    def resolve_parameter(root, info: GQLInfo, id):
        instance = info.context.instance
        try:
            return instance.context.params[id]
        except KeyError:
            raise GraphQLError(f"Parameter {id} does not exist")


types = [
    BoolParameterType,
    NumberParameterType,
    StringParameterType,
]
