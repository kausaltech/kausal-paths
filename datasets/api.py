# pylint: disable=abstract-method
from django.shortcuts import get_object_or_404
from rest_framework import serializers, viewsets, exceptions, permissions, generics
from rest_framework.response import Response

from paths.types import APIRequest
from nodes.api import instance_router
from .models import Dataset, DatasetComment, DatasetMetric, Dimension, DimensionCategory


all_routers = []


class DatasetSchemaFieldSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=200)
    type = serializers.CharField(max_length=50)
    unit = serializers.CharField(required=False)
    format = serializers.CharField(required=False)


class DatasetSchemaSerializer(serializers.Serializer):
    fields_ = DatasetSchemaFieldSerializer(many=True)
    primary_key = serializers.ListField(
        child=serializers.CharField(max_length=200)
    )
    pandas_version = serializers.CharField(max_length=20)

    def get_fields(self):
        ret = super().get_fields()
        f = ret.pop('primary_key')
        ret['primaryKey'] = f
        f = ret.pop('fields_')
        ret['fields'] = f
        return ret


class UserSerializer(serializers.Serializer):
    full_name = serializers.SerializerMethodField()

    def get_full_name(self):
        return 'Esko'


class DatasetCommentSerializer(serializers.ModelSerializer):
    created_by = UserSerializer()
    text = serializers.CharField()

    class Meta:
        model = DatasetComment


class DatasetTableSerializer(serializers.Serializer):
    schema = DatasetSchemaSerializer()
    data_ = serializers.ListSerializer(  # type: ignore
        child=serializers.DictField(),
    )

    def get_fields(self):
        ret = super().get_fields()
        f = ret.pop('data_')
        ret['data'] = f
        return ret


class DatasetMetricSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatasetMetric
        fields = ['id', 'identifier', 'uuid', 'label', 'unit']


class DatasetSerializer(serializers.ModelSerializer):
    table = DatasetTableSerializer()
    metrics = DatasetMetricSerializer(many=True)

    class Meta:
        model = Dataset
        fields = ['id', 'identifier', 'uuid', 'name', 'dimensions', 'metrics', 'table', 'comments']


class DatasetViewSet(viewsets.ViewSet, generics.GenericAPIView):
    serializer_class = DatasetSerializer
    permission_classes = (
        permissions.DjangoModelPermissions,
    )

    def get_queryset(self):
        return Dataset.objects.all()

    def list(self, request: APIRequest, instance_pk: str | None = None):
        qs = self.get_queryset().filter(instance=instance_pk)
        serializer = self.get_serializer(qs, many=True)
        return Response(serializer.data)

    def retrieve(self, request: APIRequest, pk: str | None = None, instance_pk: str | None = None):
        qs = self.get_queryset()
        obj = get_object_or_404(qs, pk=pk, instance=instance_pk)
        serializer = self.get_serializer(obj)
        return Response(serializer.data)


class DimensionCategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = DimensionCategory
        fields = ['id', 'identifier', 'uuid', 'label', 'order']


class DimensionSerializer(serializers.ModelSerializer):
    categories = DimensionCategorySerializer(many=True)

    class Meta:
        model = Dimension
        fields = ['id', 'identifier', 'uuid', 'label', 'categories']


class DimensionViewSet(viewsets.ViewSet, generics.GenericAPIView):
    serializer_class = DimensionSerializer
    permission_classes = (
        permissions.DjangoModelPermissions,
    )

    def get_queryset(self):
        return Dimension.objects.all()

    def list(self, request: APIRequest, instance_pk: str | None = None):
        qs = self.get_queryset().filter(instance=instance_pk)
        serializer = self.get_serializer(qs, many=True)
        return Response(serializer.data)

    def retrieve(self, request: APIRequest, pk: str | None = None, instance_pk: str | None = None):
        qs = self.get_queryset()
        obj = get_object_or_404(qs, pk=pk, instance=instance_pk)
        serializer = self.get_serializer(obj)
        return Response(serializer.data)


instance_router.register(r'datasets', DatasetViewSet, basename='instance-datasets')
instance_router.register(r'dimensions', DimensionViewSet, basename='instance-dimensions')
