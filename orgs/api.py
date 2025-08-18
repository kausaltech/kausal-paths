from __future__ import annotations

from typing import TYPE_CHECKING

from rest_framework import exceptions, serializers

from kausal_common.api.bulk import BulkListSerializer, BulkModelViewSet
from kausal_common.api.exceptions import HandleProtectedErrorMixin
from kausal_common.api.tree import TreebeardModelSerializerMixin
from kausal_common.api.utils import RegisteredAPIView, register_view
from kausal_common.models.general import public_fields

from paths import permissions

from nodes.models import InstanceConfig
from orgs.models import Organization

if TYPE_CHECKING:
    from rest_framework.permissions import BasePermission

all_views: list[RegisteredAPIView] = []

class OrganizationSerializer(TreebeardModelSerializerMixin, serializers.ModelSerializer):
    uuid = serializers.UUIDField(required=False)

    class Meta:  # type: ignore[override]
        model = Organization
        list_serializer_class = BulkListSerializer
        fields = public_fields(Organization)

    def create(self, validated_data):
        # from paths.context import realm_context
        instance = super().create(validated_data)
        # # Add instance to active instance's related organizations
        # request: PathsAdminRequest = self.context.get('request')
        # ic = realm_context.get().realm
        # ic.related_organizations.add(instance)
        return instance

@register_view
class OrganizationViewSet(HandleProtectedErrorMixin, BulkModelViewSet):
    queryset = Organization.objects.all()
    serializer_class = OrganizationSerializer
    filterset_fields = {
        'name': ('exact', 'in'),
    }

    # This view set is not registered with a "bulk router" (see BulkRouter or NestedBulkRouter), so we need to define
    # patch and put ourselves
    def patch(self, request, *args, **kwargs):
        return self.partial_bulk_update(request, *args, **kwargs)

    def put(self, request, *args, **kwargs):
        return self.bulk_update(request, *args, **kwargs)

    def get_permissions(self):
        permission_classes: list[type[BasePermission]]
        if self.action == 'list':
            permission_classes = [permissions.ReadOnly]
        else:
            permission_classes = [permissions.OrganizationPermission]
        return [permission() for permission in permission_classes]

    def get_queryset(self):
        queryset = super().get_queryset()
        instance_identifier = self.request.query_params.get('instance', None)
        if instance_identifier is None:
            return queryset
        try:
            instance = InstanceConfig.objects.get(identifier=instance_identifier)
        except InstanceConfig.DoesNotExist as e:
            raise exceptions.NotFound(detail="Instance not found") from e
        available_organizations = Organization.objects.qs.available_for_instance(instance)
        return available_organizations
