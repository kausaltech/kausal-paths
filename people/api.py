from __future__ import annotations

from rest_framework import exceptions

from kausal_common.api.bulk import BulkModelViewSet
from kausal_common.api.utils import register_view
from kausal_common.model_images import ModelWithImageViewMixin
from kausal_common.people.api import PersonSerializer

from paths import permissions

from nodes.models import InstanceConfig
from people.models import Person

all_views = []

@register_view
class PersonViewSet(ModelWithImageViewMixin, BulkModelViewSet):
    queryset = Person.objects.all()
    serializer_class = PersonSerializer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # This view set is not registered with a "bulk router" (see BulkRouter or NestedBulkRouter), so we need to define
    # patch and put ourselves
    def patch(self, request, *args, **kwargs):
        return self.partial_bulk_update(request, *args, **kwargs)

    def put(self, request, *args, **kwargs):
        return self.bulk_update(request, *args, **kwargs)

    def perform_destroy(self, instance):
        # FIXME: Duplicated in people.wagtail_admin.PersonDeleteView.delete_instance()
        acting_admin_user = self.request.user
        instance.delete_and_deactivate_corresponding_user(acting_admin_user)

    def get_permissions(self):
        if self.action == 'list':
            permission_classes = [permissions.ReadOnly]
        else:
            permission_classes = [permissions.PersonPermission]
        return [permission() for permission in permission_classes]

    def get_instance(self):
        instance_identifier = self.request.query_params.get('instanceIdentifier', None)

        if instance_identifier is None:
            return None
        try:
            return InstanceConfig.objects.get(identifier=instance_identifier)
        except InstanceConfig.DoesNotExist as e:
            raise exceptions.NotFound(detail="InstanceConfig not found") from e

    def user_is_authorized_for_instance(self, instance):
        user = self.request.user

        return (
            user is not None
            and user.is_authenticated
            # and hasattr(user, 'is_general_admin_for_instance')
            and user.user_is_admin_for_instance(instance)
        )

    def get_serializer_context(self):
        context = super().get_serializer_context()
        instance = self.get_instance()
        if instance is None:
            return context
        if self.user_is_authorized_for_instance(instance):
            context.update({'authorized_for_instance': instance})
        return context

    def get_queryset(self):
        queryset = super().get_queryset()
        instance = self.get_instance()
        if instance is None:
            return queryset
        if not self.user_is_authorized_for_instance(instance):
            raise exceptions.PermissionDenied(detail="Not authorized")
        return queryset.available_for_instance(instance)
