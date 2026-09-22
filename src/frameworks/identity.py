"""The jurisdiction a framework instance reports for, as an organization with official identifiers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from django.db import transaction

from loguru import logger

from orgs.models import Namespace, Organization, OrganizationIdentifier

if TYPE_CHECKING:
    from nodes.models import InstanceConfig


AGS_NAMESPACE: Final = 'ags'
AGS_NAMESPACE_NAME: Final = 'Amtlicher Gemeindeschlüssel'
AGS_PARAMETER: Final = 'ags_number'
_AGS_RE = re.compile(r'^\d{8}$')


class IdentityError(ValueError):
    pass


def ags_namespace() -> Namespace:
    namespace, _ = Namespace.objects.get_or_create(identifier=AGS_NAMESPACE, defaults={'name': AGS_NAMESPACE_NAME})
    return namespace


def ags_from_parameters(ic: InstanceConfig) -> str | None:
    """Read the AGS from the instance's `ags_number` parameter, the only place it lives before provisioning."""
    spec = ic.ensure_spec()
    for param in spec.params:
        if param.local_id == AGS_PARAMETER and param.value:
            return str(param.value)
    return None


def ensure_municipal_organization(ic: InstanceConfig, ags: str | None = None) -> Organization | None:
    """
    Make `ic` belong to the organization identified by its AGS, creating it if needed.

    The organization is found by its `ags` identifier, so provisioning several
    instances for one municipality converges on one organization. A new one is
    named after the instance owner. The denormalized organization fields on the
    instance's `FrameworkConfig` are kept in step, since the data studio already
    reads them. Returns None when the instance has no AGS.

    The organization is created as a root. The Land / Kreis hierarchy is not
    modelled yet; the Land is recoverable from the first two AGS digits.
    """
    ags = ags or ags_from_parameters(ic)
    if ags is None:
        return None
    if not _AGS_RE.match(ags):
        raise IdentityError(f'{ic.identifier}: {ags!r} is not an eight-digit AGS')

    with transaction.atomic():
        namespace = ags_namespace()
        existing = (
            OrganizationIdentifier.objects.select_related('organization').filter(namespace=namespace, identifier=ags).first()
        )
        if existing is not None:
            org = existing.organization
        else:
            name = ic.owner or ic.name
            if not name:
                raise IdentityError(f'{ic.identifier}: no owner name to name the organization after')
            org = Organization.add_root(name=name)
            OrganizationIdentifier.objects.create(organization=org, namespace=namespace, identifier=ags)
            logger.info('Created organization %s for AGS %s' % (org.name, ags))

        if ic.organization_id != org.pk:
            ic.organization = org
            ic.save(update_fields=['organization'])
        if ic.has_framework_config():
            fwc = ic.framework_config
            if (fwc.organization_name, fwc.organization_identifier) != (org.name, ags):
                fwc.organization_name = org.name
                fwc.organization_identifier = ags
                fwc.save(update_fields=['organization_name', 'organization_identifier'])
    return org
