from __future__ import annotations

from typing import TYPE_CHECKING, override

from django.contrib.contenttypes.models import ContentType
from django.db.models import ObjectDoesNotExist, Q
from rest_framework import exceptions, serializers
from rest_framework.routers import DefaultRouter

from rest_framework_nested.routers import NestedSimpleRouter

from kausal_common.api.permissions import NestedResourcePermissionPolicyDRFPermission, PermissionPolicyDRFPermission
from kausal_common.datasets.api import (
    DataPointCommentViewSet as BaseDataPointCommentViewSet,
    DataPointSourceReferenceViewSet as BaseDataPointSourceReferenceViewSet,
    DataPointViewSet as BaseDataPointViewSet,
    DatasetCommentsViewSet as BaseDatasetCommentsViewSet,
    DatasetMetricViewSet as BaseDatasetMetricViewSet,
    DatasetSchemaViewSet as BaseDatasetSchemaViewSet,
    DatasetSourceReferenceViewSet as BaseDatasetSourceReferenceViewSet,
    DatasetViewSet as BaseDatasetViewSet,
    DataSourceViewSet as BaseDataSourceViewSet,
    DimensionCategoryViewSet as BaseDimensionCategoryViewSet,
    DimensionViewSet as BaseDimensionViewSet,
)
from kausal_common.datasets.models import (
    DataPoint,
    DataPointComment,
    Dataset,
    DatasetMetric,
    DatasetSchema,
    DatasetSourceReference,
    DataSource,
)
from kausal_common.users import user_or_bust

from nodes.dataset_materialization import dataset_change, datasets_change, refresh_dataset_materialization
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from rest_framework.response import Response
    from rest_framework.routers import SimpleRouter
    from rest_framework.views import APIView


class DataPointCommentPermission(NestedResourcePermissionPolicyDRFPermission[DataPointComment, DataPoint, DataPoint]):
    class Meta:
        model = DataPointComment
        view_kwargs_parent_key = 'datapoint_uuid'
        nested_parent_model = DataPoint
        nested_parent_key_field = 'uuid'

    @override
    def get_create_context_from_api_view(self, view: APIView) -> DataPoint:
        data_point_uuid = view.kwargs['datapoint_uuid']
        return DataPoint.objects.get(uuid=data_point_uuid)


class DataPointCommentViewSet(BaseDataPointCommentViewSet):
    @override
    def get_permissions(self):
        return [DataPointCommentPermission()]

    @override
    def perform_create(self, serializer):
        dataset = Dataset.objects.get(uuid=self.kwargs['dataset_uuid'])
        with dataset_change(dataset, user=user_or_bust(self.request.user)):
            super().perform_create(serializer)

    @override
    def perform_update(self, serializer):
        dataset = Dataset.objects.get(uuid=self.kwargs['dataset_uuid'])
        with dataset_change(dataset, user=user_or_bust(self.request.user)):
            super().perform_update(serializer)

    @override
    def perform_destroy(self, instance):
        dataset = Dataset.objects.get(uuid=self.kwargs['dataset_uuid'])
        with dataset_change(dataset, user=user_or_bust(self.request.user)):
            super().perform_destroy(instance)


class DatasetSourceReferencePermission(NestedResourcePermissionPolicyDRFPermission[DatasetSourceReference, Dataset, Dataset]):
    class Meta:
        model = DatasetSourceReference
        view_kwargs_parent_key = 'dataset_uuid'
        nested_parent_model = Dataset
        nested_parent_key_field = 'uuid'

    @override
    def get_create_context_from_api_view(self, view: APIView) -> Dataset:
        dataset_uuid = view.kwargs['dataset_uuid']
        return Dataset.objects.get(uuid=dataset_uuid)


class DataPointSourceReferencePermission(NestedResourcePermissionPolicyDRFPermission[DatasetSourceReference, Dataset, DataPoint]):
    class Meta:
        model = DatasetSourceReference
        view_kwargs_parent_key = 'datapoint_uuid'
        nested_parent_model = DataPoint
        nested_parent_key_field = 'uuid'

    @override
    def get_create_context_from_api_view(self, view: APIView) -> Dataset:
        datapoint_uuid = view.kwargs['datapoint_uuid']
        datapoint = DataPoint.objects.get(uuid=datapoint_uuid)
        return datapoint.dataset


class DataPointSourceReferenceViewSet(BaseDataPointSourceReferenceViewSet):
    @override
    def get_permissions(self):
        return [DataPointSourceReferencePermission()]

    @override
    def perform_create(self, serializer):
        dataset = Dataset.objects.get(uuid=self.kwargs['dataset_uuid'])
        with dataset_change(dataset, user=user_or_bust(self.request.user)):
            super().perform_create(serializer)

    @override
    def perform_update(self, serializer):
        dataset = Dataset.objects.get(uuid=self.kwargs['dataset_uuid'])
        with dataset_change(dataset, user=user_or_bust(self.request.user)):
            super().perform_update(serializer)

    @override
    def perform_destroy(self, instance):
        dataset = Dataset.objects.get(uuid=self.kwargs['dataset_uuid'])
        with dataset_change(dataset, user=user_or_bust(self.request.user)):
            super().perform_destroy(instance)


class DatasetSourceReferenceViewSet(BaseDatasetSourceReferenceViewSet):
    @override
    def get_permissions(self):
        return [DatasetSourceReferencePermission()]

    @override
    def perform_create(self, serializer):
        dataset = Dataset.objects.get(uuid=self.kwargs['dataset_uuid'])
        with dataset_change(dataset, user=user_or_bust(self.request.user)):
            super().perform_create(serializer)

    @override
    def perform_update(self, serializer):
        dataset = Dataset.objects.get(uuid=self.kwargs['dataset_uuid'])
        with dataset_change(dataset, user=user_or_bust(self.request.user)):
            super().perform_update(serializer)

    @override
    def perform_destroy(self, instance):
        dataset = Dataset.objects.get(uuid=self.kwargs['dataset_uuid'])
        with dataset_change(dataset, user=user_or_bust(self.request.user)):
            super().perform_destroy(instance)


class DataPointPermission(NestedResourcePermissionPolicyDRFPermission[DataPoint, Dataset, Dataset]):
    class Meta:
        model = DataPoint
        view_kwargs_parent_key = 'dataset_uuid'
        nested_parent_model = Dataset
        nested_parent_key_field = 'uuid'

    @override
    def get_create_context_from_api_view(self, view: APIView) -> Dataset:
        return Dataset.objects.get(uuid=view.kwargs['dataset_uuid'])


class DataPointViewSet(BaseDataPointViewSet):
    @override
    def get_permissions(self):
        return [DataPointPermission()]

    @override
    def _with_locked_dataset(self, callback: Callable[[], Response]) -> Response:
        if hasattr(self, '_locked_dataset'):
            return callback()
        dataset = Dataset.objects.get(uuid=self.kwargs['dataset_uuid'])
        with dataset_change(dataset, user=user_or_bust(self.request.user)) as locked_dataset:
            self._locked_dataset = locked_dataset
            try:
                return callback()
            finally:
                del self._locked_dataset


class DatasetMetricPermission(NestedResourcePermissionPolicyDRFPermission[DatasetMetric, None, DatasetSchema]):
    class Meta:
        model = DatasetMetric
        view_kwargs_parent_key = 'datasetschema_uuid'
        nested_parent_model = DatasetSchema
        nested_parent_key_field = 'uuid'
        allowed_actions = {'view'}

    def get_create_context_from_api_view(self, view: APIView) -> None:
        return None


class DatasetMetricViewSet(BaseDatasetMetricViewSet):
    @override
    def get_permissions(self):
        return [DatasetMetricPermission()]


class DatasetSchemaPermission(PermissionPolicyDRFPermission[DatasetSchema, None]):
    class Meta:
        model = DatasetSchema

    def get_create_context_from_api_view(self, view: APIView) -> None:
        return None


class DatasetSchemaViewSet(BaseDatasetSchemaViewSet):
    @override
    def get_permissions(self):
        return [DatasetSchemaPermission()]

    @override
    def permission_denied(self, request, message=None, code=None):
        if request.authenticators and not request.successful_authenticator:
            raise exceptions.NotAuthenticated()
        if code == 'not_found':
            # Try to avoid revealing existence of object when not desired
            raise exceptions.NotFound(detail=message, code=code)
        raise exceptions.PermissionDenied(detail=message, code=code)

    @override
    def perform_update(self, serializer):
        datasets = Dataset.objects.filter(schema=serializer.instance).order_by('pk')
        with datasets_change(datasets, user=user_or_bust(self.request.user)):
            super().perform_update(serializer)


class DatasetPermission(PermissionPolicyDRFPermission[Dataset, DatasetSchema]):
    class Meta:
        model = Dataset

    def get_create_context_from_api_view(self, view: APIView) -> DatasetSchema:
        data = view.request.data
        if not isinstance(data, dict):
            raise serializers.ValidationError('Expected an object')
        schema_uuid = data['schema']
        try:
            return DatasetSchema.objects.get(uuid=schema_uuid)
        except DatasetSchema.DoesNotExist as e:
            raise serializers.ValidationError('DatasetSchema not found') from e


class DatasetViewSet(BaseDatasetViewSet):
    @override
    def get_permissions(self):
        return [DatasetPermission()]

    @override
    def perform_create(self, serializer):
        from django.db import transaction

        user = user_or_bust(self.request.user)
        with transaction.atomic():
            dataset = serializer.save(created_by=user, last_modified_by=user)
            refresh_dataset_materialization(dataset, user=user)

    @override
    def perform_update(self, serializer):
        dataset = serializer.instance
        with dataset_change(dataset, user=user_or_bust(self.request.user)):
            super().perform_update(serializer)


class DatasetCommentPermission(NestedResourcePermissionPolicyDRFPermission[DataPointComment, None, Dataset]):
    class Meta:
        model = DataPointComment
        view_kwargs_parent_key = 'dataset_uuid'
        nested_parent_model = Dataset
        nested_parent_key_field = 'uuid'
        allowed_actions = {'view'}

    def get_create_context_from_api_view(self, view: APIView) -> None:
        return None


class DatasetCommentsViewSet(BaseDatasetCommentsViewSet):
    @override
    def get_permissions(self):
        return [DatasetCommentPermission()]


class DataSourcePermission(PermissionPolicyDRFPermission[DataSource, InstanceConfig]):
    class Meta:
        model = DataSource

    def get_create_context_from_api_view(self, view: APIView) -> InstanceConfig:
        data = view.request.data
        if not isinstance(data, dict):
            raise serializers.ValidationError('Expected an object')
        content_type_app = data.get('content_type_app')
        content_type_model = data.get('content_type_model')
        object_id = data.get('object_id')

        try:
            content_type = ContentType.objects.get(app_label=content_type_app, model=content_type_model)
        except ContentType.DoesNotExist as e:
            raise serializers.ValidationError('Scope object not found') from e
        model = content_type.model_class()
        assert isinstance(model, type(InstanceConfig))
        if model is None or object_id is None:
            raise serializers.ValidationError('Scope object not found')
        try:
            instance_config = model.objects.get(pk=int(object_id))
        except ObjectDoesNotExist as e:
            raise serializers.ValidationError('Scope object not found') from e
        return instance_config


class DataSourceViewSet(BaseDataSourceViewSet):
    @override
    def get_permissions(self):
        return [DataSourcePermission()]

    @override
    def perform_update(self, serializer):
        datasets = (
            Dataset.objects
            .filter(
                Q(source_references__data_source=serializer.instance)
                | Q(data_points__source_references__data_source=serializer.instance)
            )
            .distinct()
            .order_by('pk')
        )
        with datasets_change(datasets, user=user_or_bust(self.request.user)):
            super().perform_update(serializer)


class DimensionCategoryViewSet(BaseDimensionCategoryViewSet):
    pass


class DimensionViewSet(BaseDimensionViewSet):
    pass


router = DefaultRouter()
router.register(r'dataset_schemas', DatasetSchemaViewSet, basename='datasetschema')
router.register(r'datasets', DatasetViewSet, basename='dataset')
router.register(r'dimensions', DimensionViewSet, basename='dimension')
router.register(r'data_sources', DataSourceViewSet, basename='datasource')

dataset_router = NestedSimpleRouter(router, r'datasets', lookup='dataset')
datasetschema_router = NestedSimpleRouter(router, r'dataset_schemas', lookup='datasetschema')
dimension_router = NestedSimpleRouter(router, r'dimensions', lookup='dimension')

dataset_router.register(r'comments', DatasetCommentsViewSet, basename='datasetcomment')
dataset_router.register(r'data_points', DataPointViewSet, basename='datapoint')
dataset_router.register(r'sources', DatasetSourceReferenceViewSet, basename='datasetsource')
datasetschema_router.register(r'metrics', DatasetMetricViewSet, basename='datasetmetric')
dimension_router.register(r'categories', DimensionCategoryViewSet, basename='category')

datapoint_router = NestedSimpleRouter(dataset_router, r'data_points', lookup='datapoint')
datapoint_router.register(r'comments', DataPointCommentViewSet, basename='datapointcomment')
datapoint_router.register(r'sources', DataPointSourceReferenceViewSet, basename='datapointsource')

nested_routers: list[SimpleRouter] = []
nested_routers.append(dataset_router)
nested_routers.append(dimension_router)
nested_routers.append(datasetschema_router)
nested_routers.append(datapoint_router)
