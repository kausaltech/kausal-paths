import pytest

from orgs.models import Organization
from orgs.tests.factories import OrganizationFactory

pytestmark = pytest.mark.django_db


def test_organization_queryset_available_for_instance(instance_config):
    assert instance_config.organization
    instance_org = instance_config.organization
    org = OrganizationFactory()
    # Creating a new root org might've changed the path of instance_org
    instance_org.refresh_from_db()
    sub_org1 = OrganizationFactory(parent=instance_config.organization)
    OrganizationFactory(parent=org)  # sub_org2
    result = list(Organization.objects.qs.available_for_instance(instance_config))
    assert result == [instance_org, sub_org1]
    # instance.related_organizations.add(org) # TODO: Add this if we implement related_organizations
    # result = set(Organization.objects.available_for_instance(instance_config))
    # assert result == {instance_org, sub_org1, org, sub_org2}
