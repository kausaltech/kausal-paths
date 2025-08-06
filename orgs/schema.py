from __future__ import annotations

import graphene

from kausal_common.organizations.schema import (
    CreateOrganizationMutation as BaseCreateOrganizationMutation,
    DeleteOrganizationMutation as BaseDeleteOrganizationMutation,
    Mutation as BaseMutation,
    OrganizationForm as BaseOrganizationForm,
    OrganizationNode as BaseOrganizationNode,
    UpdateOrganizationMutation as BaseUpdateOrganizationMutation,
)

from paths.graphql_helpers import AdminButtonsMixin

from orgs.models import Organization


class OrganizationForm(BaseOrganizationForm):
    class Meta:
        model = Organization
        fields = ['parent', 'name', 'classification', 'abbreviation', 'founding_date', 'dissolution_date']


class OrganizationNode(BaseOrganizationNode, AdminButtonsMixin):

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

class CreateOrganizationMutation(BaseCreateOrganizationMutation):
    class Meta:
        form_class = OrganizationForm

class UpdateOrganizationMutation(BaseUpdateOrganizationMutation):
    class Meta:
        form_class = OrganizationForm

class DeleteOrganizationMutation(BaseDeleteOrganizationMutation):
    class Meta:
        model = Organization

class Mutation(BaseMutation):
    create_organization = CreateOrganizationMutation.Field()
    update_organization = UpdateOrganizationMutation.Field()
    delete_organization = DeleteOrganizationMutation.Field()
