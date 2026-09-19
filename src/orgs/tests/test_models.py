from django.contrib.gis.geos import Point

import pytest

from orgs.models import Organization
from orgs.tests.factories import OrganizationFactory

pytestmark = pytest.mark.django_db


def test_organization_coordinates_follow_location():
    organization = OrganizationFactory.create(location=Point(24.9384, 60.1699, srid=4326))

    assert organization.latitude == 60.1699
    assert organization.longitude == 24.9384

    organization.location = Point(25.0, 61.0, srid=4326)
    organization.save(update_fields={'location'})
    organization.refresh_from_db()

    assert organization.latitude == 61.0
    assert organization.longitude == 25.0

    organization.location = None
    organization.save(update_fields={'location'})
    organization.refresh_from_db()

    assert organization.latitude is None
    assert organization.longitude is None


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
