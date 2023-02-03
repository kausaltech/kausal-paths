# pylint: disable=abstract-method
from typing import Any, List, Sequence, Union

import pandas as pd
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.urls import reverse
from django.db.transaction import atomic
from rest_framework import serializers, viewsets, exceptions, permissions, generics, mixins
from rest_framework.response import Response
from rest_framework_nested import routers, relations
from drf_spectacular.utils import extend_schema_field, extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes

from paths.types import APIRequest
from nodes.api import instance_router
from nodes.models import InstanceConfig
from nodes.constants import YEAR_COLUMN, FORECAST_COLUMN
from paths.utils import validate_unit
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

    def get_full_name(self, instance):
        return instance.get_full_name()


class DatasetCommentSerializer(serializers.ModelSerializer):
    created_by = UserSerializer(required=False)
    text = serializers.CharField()
    id = serializers.IntegerField(required=False)
    dataset = serializers.PrimaryKeyRelatedField(queryset=Dataset.objects.all())
    type = serializers.CharField(required=False, allow_null=True)
    state = serializers.CharField(required=False, allow_null=True)
    cell_path = serializers.CharField(required=False, allow_null=True)

    def create(self, validated_data: dict):
        request = self.context.get('request')
        if request is None:
            raise exceptions.NotAuthenticated()
        user = request.user
        validated_data['created_by'] = user
        return super().create(validated_data)

    class Meta:
        model = DatasetComment
        fields = ('id', 'text', 'created_by', 'created_at', 'dataset', 'type', 'state', 'cell_path')
        read_only_fields = ('id',)


@extend_schema(
    parameters=[
        OpenApiParameter("instance_id", OpenApiTypes.INT, location='path'),
    ]
)
class DatasetCommentViewSet(viewsets.ModelViewSet):
    serializer_class = DatasetCommentSerializer

    def get_queryset(self):
        dataset_id = self.kwargs.get('dataset_pk', 0)
        return DatasetComment.objects.filter(dataset_id=dataset_id)


class DatasetTableSerializer(serializers.Serializer):
    schema = DatasetSchemaSerializer()
    data_ = serializers.ListSerializer(  # type: ignore
        child=serializers.DictField(),
    )

    def validate(self, attrs: dict) -> dict:
        # FIXME: Move some of the schema validation logic here
        return super().validate(attrs)

    def validate_empty_values(self, data: Any) -> tuple[bool, Any]:
        if data is serializers.empty:
            # We allow it to be null in incoming data
            raise serializers.SkipField
        return super().validate_empty_values(data)

    def get_fields(self):
        ret = super().get_fields()
        f = ret.pop('data_')
        ret['data'] = f
        return ret


class DatasetMetricSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatasetMetric
        fields = ['id', 'identifier', 'uuid', 'label', 'unit']


class InstanceRelatedField(serializers.PrimaryKeyRelatedField):
    def get_queryset(self):
        qs = super().get_queryset()
        parent = self.parent
        while parent is not None:
            if isinstance(parent, serializers.Serializer):
                break
            parent = parent.parent
        if parent is None:
            raise Exception("No Serializer as parent")
        ic = parent.context.get('instance_config')
        if ic is None:
            raise Exception("Serializer didn't have 'instance_config' in context")
        return qs.filter(instance=ic)


class DatasetSerializer(serializers.ModelSerializer):
    table = DatasetTableSerializer(required=True, allow_null=False)
    metrics = DatasetMetricSerializer(many=True)
    dimensions = InstanceRelatedField(many=True, queryset=Dimension.objects.all())
    comments_url = serializers.SerializerMethodField()

    instance: Dataset | None

    class Meta:
        model = Dataset
        fields = [
            'id', 'identifier', 'uuid', 'name', 'years', 'dimensions', 'metrics', 'table', 'comments_url',
            'created_at', 'created_by', 'updated_at', 'updated_by',
        ]
        extra_kwargs = dict(
            created_at=dict(read_only=True),
            updated_at=dict(read_only=True),
            created_by=dict(read_only=True),
            updated_by=dict(read_only=True),
        )

    @extend_schema_field(OpenApiTypes.URI)
    def get_comments_url(self, obj):
        return reverse('dataset-comments-list', kwargs=dict(instance_pk=obj.instance.pk, dataset_pk=obj.pk))

    def validate_table(self, table: dict):
        #print(table)
        return table

    def validate_table_cols(self, table: dict, metric_cols: list[str], dim_cols: dict[str, Dimension]):
        data = table['data']
        required_cols = set(metric_cols + list(dim_cols.keys()) + [YEAR_COLUMN, 'uuid'])
        allowed_cols = set([FORECAST_COLUMN]) | required_cols
        all_row_errors = {}
        cols_present = set()
        for row in data:
            row_errors: List[Union[str, dict]] = []
            uuid = row.get('uuid')
            if uuid is None:
                raise exceptions.ValidationError(dict(table="A row is missing the 'uuid' field"))
            cols = set(row.keys())
            cols_present.update(cols)
            extra = cols - allowed_cols
            if extra:
                row_errors.append('Unknown columns: %s' % ', '.join(extra))
            missing = required_cols - cols
            if missing:
                row_errors.append('Missing columns: %s' % ', '.join(missing))

            for col, val in row.items():
                if col == YEAR_COLUMN:
                    if not isinstance(val, int):
                        row_errors.append({col: 'Integer expected'})
                elif col == FORECAST_COLUMN:
                    if not isinstance(val, bool):
                        row_errors.append({col: 'Bool expected'})
                elif col in metric_cols:
                    if val is not None and not isinstance(val, (float, int)):
                        row_errors.append({col: 'Float or int expected'})
                elif col in dim_cols:
                    dim = dim_cols[col]
                    if val not in dim._cat_map:  # type: ignore
                        row_errors.append({col: 'Unknown category'})

            if row_errors:
                all_row_errors[uuid] = row_errors

        if all_row_errors:
            raise exceptions.ValidationError(dict(table=dict(data=all_row_errors)))

        return cols_present

    def validate_table_schema(self, table: dict, metric_cols: list[str], dim_cols: dict[str, Dimension], cols_present: set[str]):
        fields = table['schema']['fields']
        names = set([f['name'] for f in fields])
        fields_errors = []
        schema_errors = {}
        if names != cols_present:
            fields_errors.append("Fields do not correspond to rows")
        for f in fields:
            fn = f['name']
            if fn in metric_cols:
                if f['type'] != 'number':
                    fields_errors.append("Expecting a type 'number' for field %s" % f['name'])
                unit = f.get('unit')
                try:
                    validate_unit(unit)
                except:  # noqa
                    fields_errors.append("Invalid unit for field %s" % f['name'])
            elif fn in dim_cols or fn in ['uuid']:
                if f['type'] != 'string':
                    fields_errors.append("Expecting a type 'string' for field %s" % f['name'])
            elif fn == YEAR_COLUMN:
                if f['type'] != 'integer':
                    fields_errors.append("Expecting a type 'integer' for field %s" % f['name'])

        if fields_errors:
            schema_errors['fields'] = fields_errors

        if schema_errors:
            raise exceptions.ValidationError(dict(table=dict(schema=schema_errors)))

    def validate(self, attrs: dict):
        table = attrs.get('table')
        if table is not None:
            metrics = attrs.get('metrics')
            if metrics is not None:
                metric_cols = [m['identifier'] for m in metrics]
            else:
                assert self.instance is not None
                metric_cols = [m.identifier for m in self.instance.metrics.all()]

            dims = attrs.get('dimensions')
            if dims is None:
                assert self.instance is not None
                dims = list(self.instance.dimensions.all())

            for dim in dims:
                dim._cat_map = {cat.identifier: cat for cat in dim.categories.all()}  # type: ignore
            dim_cols = {dim.identifier: dim for dim in dims}
            cols_present = self.validate_table_cols(table, metric_cols, dim_cols)
            self.validate_table_schema(table, metric_cols, dim_cols, cols_present)

        return attrs

    def inject_common_data(self, validated_data: dict, is_create: bool):
        request = self.context.get('request')
        if request is not None:
            user = request.user
        else:
            user = None
        validated_data['updated_at'] = timezone.now()
        validated_data['updated_by'] = user
        if is_create:
            validated_data['created_by'] = validated_data['updated_by']
            validated_data['created_at'] = validated_data['updated_at']

    @atomic
    def update(self, instance: Dataset, validated_data: dict) -> Dataset:
        self.inject_common_data(validated_data=validated_data, is_create=False)

        try:
            metrics = validated_data.pop('metrics')
            metric_s = [DatasetMetricSerializer(data=m) for m in metrics]
            for s in metric_s:
                s.is_valid(raise_exception=True)
        except KeyError:
            metric_s = []

        ds: Dataset = super().update(instance, validated_data)

        for s in metric_s:
            s.save(dataset=ds)

        if ds.table is None:
            ds.table = ds.generate_empty_table()

        if not ds.years:
            ds.years = ds.generate_years_from_data()

        return ds

    @atomic
    def create(self, validated_data: dict):
        validated_data['instance'] = self.context['instance_config']

        self.inject_common_data(validated_data=validated_data, is_create=True)

        metrics = validated_data.pop('metrics')
        metric_s = [DatasetMetricSerializer(data=m) for m in metrics]
        for s in metric_s:
            s.is_valid(raise_exception=True)

        if Dataset.objects.filter(identifier=validated_data['identifier']).exists():
            raise exceptions.ValidationError(dict(identifier='identifier already exists'))
        ds: Dataset = super().create(validated_data)

        for s in metric_s:
            s.save(dataset=ds)

        if ds.table is None:
            ds.table = ds.generate_empty_table()  # type: ignore
            ds.save()

        return ds


class DatasetViewSet(viewsets.ModelViewSet):
    serializer_class = DatasetSerializer
    permission_classes = (
        permissions.DjangoModelPermissions,
    )

    def get_serializer_context(self) -> dict[str, Any]:
        ret = super().get_serializer_context()
        ipk = self.kwargs.get('instance_pk')
        if ipk:
            ic = InstanceConfig.objects.get(pk=ipk)
        else:
            obj = InstanceConfig.objects.first()
            assert obj is not None
            ic = obj
        ret['instance_config'] = ic
        return ret

    def get_queryset(self):
        instance_pk = self.kwargs.get('instance_pk', 0)
        return Dataset.objects.filter(instance=instance_pk)


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

dataset_router = routers.NestedSimpleRouter(instance_router, r'datasets', lookup='dataset')
dataset_router.register(r'comments', DatasetCommentViewSet, basename='dataset-comments')

all_routers.append(dataset_router)
