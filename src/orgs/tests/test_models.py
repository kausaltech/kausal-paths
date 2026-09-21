from django.db import IntegrityError, transaction

import pytest

from kausal_common.organizations.forms import OrganizationLocationField, OrganizationLocationFormMixin

from orgs.models import Organization
from orgs.tests.factories import OrganizationFactory

pytestmark = pytest.mark.django_db


class OrganizationLocationForm(OrganizationLocationFormMixin):
    coordinate_location = OrganizationLocationField()

    class Meta:
        model = Organization
        fields = ('coordinate_location',)


def test_organization_location_form_maps_widget_value_to_coordinates():
    organization = OrganizationFactory.create()
    form = OrganizationLocationForm(
        data={'coordinate_location': 'SRID=4326;POINT(24.9384 60.1699)'},
        instance=organization,
    )

    assert form.is_valid(), form.errors
    organization = form.save()
    assert organization.latitude == 60.1699
    assert organization.longitude == 24.9384

    initial_form = OrganizationLocationForm(instance=organization)
    assert initial_form.initial['coordinate_location'] == 'SRID=4326;POINT(24.9384 60.1699)'


def test_organization_coordinates_are_stored_as_a_pair():
    organization = OrganizationFactory.create(latitude=60.1699, longitude=24.9384)

    organization.refresh_from_db()
    assert organization.latitude == 60.1699
    assert organization.longitude == 24.9384
    assert organization.location is not None
    assert organization.location.x == 24.9384
    assert organization.location.y == 60.1699

    organization.latitude = 61.0
    organization.save(update_fields={'latitude'})
    organization.refresh_from_db()

    assert organization.location is not None
    assert organization.location.x == 24.9384
    assert organization.location.y == 61.0

    with pytest.raises(IntegrityError), transaction.atomic():
        Organization.objects.filter(pk=organization.pk).update(latitude=None)


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
