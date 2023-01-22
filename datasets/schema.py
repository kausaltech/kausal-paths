from graphene import NonNull
import graphene


class Category(graphene.ObjectType):
    id = graphene.String()
    label = graphene.String()
    values = graphene.List(NonNull(graphene.Float), required=True)


class Dimension(graphene.ObjectType):
    id = graphene.ID()
    categories = graphene.List(NonNull(Category), required=True)


class Metric(graphene.ObjectType):
    id = graphene.ID()
    dimensions = graphene.List(NonNull(Dimension), required=True)
    years = graphene.List(NonNull(graphene.Int), required=True)
    is_forecast = graphene.List(NonNull(graphene.Boolean), required=True)
    values = graphene.List(NonNull(graphene.Float), required=True, normalize_by=graphene.ID(required=False))
    unit = graphene.Field('paths.schema.UnitType')
    yearly_cumulative_unit = graphene.Field('paths.schema.UnitType')
