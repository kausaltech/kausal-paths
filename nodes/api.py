from django.shortcuts import get_object_or_404
from rest_framework import viewsets, serializers, settings, permissions, response, generics
from rest_framework_nested import routers, relations

from paths.api_router import router
from paths.types import APIRequest
from .models import InstanceConfig


all_routers = []


class InstanceSerializer(serializers.ModelSerializer):
    datasets = relations.NestedHyperlinkedIdentityField(
        view_name='instance-datasets-list',
        read_only=True,
        lookup_url_kwarg='instance_pk'
    )
    dimensions = relations.NestedHyperlinkedIdentityField(
        view_name='instance-dimensions-list',
        read_only=True,
        lookup_url_kwarg='instance_pk'
    )
    data_sources = relations.NestedHyperlinkedIdentityField(
        view_name='instance-data-sources-list',
        read_only=True,
        lookup_url_kwarg='instance_pk'
    )

    class Meta:
        model = InstanceConfig
        fields = ['id', 'identifier', 'name', 'datasets', 'dimensions', 'data_sources']


class ReadOnly(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.method in permissions.SAFE_METHODS


class InstanceViewSet(viewsets.ViewSet, generics.GenericAPIView):
    permission_classes = [ReadOnly, *settings.api_settings.DEFAULT_PERMISSION_CLASSES]  # type: ignore
    serializer_class = InstanceSerializer

    def get_queryset(self):
        return InstanceConfig.objects.all()

    def list(self, request: APIRequest):
        qs = self.get_queryset()
        serializer = self.get_serializer(qs, many=True)
        return response.Response(serializer.data)

    def retrieve(self, request: APIRequest, pk: str | None = None):
        qs = self.get_queryset()
        obj = get_object_or_404(qs, pk=pk)
        serializer = self.get_serializer(obj)
        return response.Response(serializer.data)


router.register(r'instances', InstanceViewSet, basename='instance')
instance_router = routers.NestedSimpleRouter(router, r'instances', lookup='instance')
all_routers.append(instance_router)
