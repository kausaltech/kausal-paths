from __future__ import annotations

import graphene

from kausal_common.organizations.schema import (
    OrganizationNode as BaseOrganizationNode,
)

from paths.graphql_helpers import AdminButtonsMixin

from orgs.models import Organization


class OrganizationNode(BaseOrganizationNode, AdminButtonsMixin):

    class Meta:
        model = Organization
        abstract = False
        fields = [
            'id', 'abbreviation', 'name', 'description', 'url', 'email', 'classification', 'distinct_name',
            # 'location',  # commented out as this causes an exception:
            # Don't know how to convert the Django field orgs.Organization.location (<class
            # 'django.contrib.gis.db.models.fields.PointField'>)
        ]

class Query:
    organization = graphene.Field(OrganizationNode, id=graphene.ID(required=True))

    @staticmethod
    def resolve_organization(root, info, id: str) -> Organization:
        return Organization.objects.get(id=id)
