import graphene
from graphene_django.types import DjangoObjectType
from kausal_common.graphene import DjangoNode
from kausal_common.organizations.schema import OrganizationNode as BaseOrganizationNode
from orgs.models import Organization


class OrganizationNode(BaseOrganizationNode):
    class Meta:
        model = Organization
        abstract = False
        fields = [
            'id', 'abbreviation', 'name', 'description', 'url', 'email', 'classification', 'distinct_name', 'location',
        ]

class Query:
    organization = graphene.Field(OrganizationNode, id=graphene.ID(required=True))

    @staticmethod
    def resolve_organization(root, info, id: str) -> Organization:
        return Organization.objects.get(id=id)