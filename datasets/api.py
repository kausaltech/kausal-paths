from __future__ import annotations

from typing import TYPE_CHECKING, override

from rest_framework import serializers
from rest_framework.exceptions import MethodNotAllowed, NotFound
from rest_framework.routers import DefaultRouter, SimpleRouter

from rest_framework_nested.routers import NestedSimpleRouter

from kausal_common.api.permissions import PermissionPolicyDRFPermission
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
)
from kausal_common.users import user_or_none

if TYPE_CHECKING:
    from rest_framework.request import Request
    from rest_framework.views import APIView


class DataPointCommentPermission(PermissionPolicyDRFPermission[DataPointComment, DataPoint]):
    class Meta:
        model = DataPointComment

    def get_create_context_from_api_view(self, view: APIView) -> DataPoint:
        data_point_uuid = view.kwargs['datapoint_uuid']
        return DataPoint.objects.get(uuid=data_point_uuid)


class DataPointCommentViewSet(BaseDataPointCommentViewSet):
    @override
    def get_permissions(self):
        return [DataPointCommentPermission()]


class DatasetSourceReferencePermission(PermissionPolicyDRFPermission[DatasetSourceReference, Dataset]):
    class Meta:
        model = DatasetSourceReference

    def get_create_context_from_api_view(self, view: APIView) -> Dataset:
        dataset_uuid = view.kwargs['dataset_uuid']
        return Dataset.objects.get(uuid=dataset_uuid)


class DataPointSourceReferenceViewSet(BaseDataPointSourceReferenceViewSet):
    @override
    def get_permissions(self):
        return [DatasetSourceReferencePermission()]


class DatasetSourceReferenceViewSet(BaseDatasetSourceReferenceViewSet):
    @override
    def get_permissions(self):
        return [DatasetSourceReferencePermission()]


class DataPointPermission(PermissionPolicyDRFPermission[DataPoint, Dataset]):
    class Meta:
        model = DataPoint

    @override
    def get_create_context_from_api_view(self, view: APIView) -> Dataset:
        return Dataset.objects.get(uuid=view.kwargs['dataset_uuid'])

class DataPointViewSet(BaseDataPointViewSet):
    @override
    def get_permissions(self):
        return [DataPointPermission()]


class DatasetMetricPermission(PermissionPolicyDRFPermission[DatasetMetric, None]):
    class Meta:
        model = DatasetMetric

    def has_permission(self, request: Request, view: APIView):
        schema_uuid = view.kwargs['datasetschema_uuid']
        schema = DatasetSchema.objects.get(uuid=schema_uuid)
        pp = schema.permission_policy()
        user = user_or_none(request.user)
        if not user:
            return False
        action = self.http_method_to_django_action(request.method)
        if request.method is None:
            raise ValueError('No method supplied')
        if action != 'view':
            raise MethodNotAllowed(request.method)
        if not super().has_permission(request, view):
            return False
        if not pp.user_has_any_permission_for_instance(user, ['view'], schema):
            raise NotFound()
        return True

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


class DatasetPermission(PermissionPolicyDRFPermission[Dataset, DatasetSchema]):
    class Meta:
        model = Dataset

    def get_create_context_from_api_view(self, view: APIView) -> DatasetSchema:
        schema_uuid = view.request.data['schema']
        try:
            return DatasetSchema.objects.get(uuid=schema_uuid)
        except DatasetSchema.DoesNotExist as e:
            raise serializers.ValidationError('DatasetSchema not found') from e


class DatasetViewSet(BaseDatasetViewSet):
    @override
    def get_permissions(self):
        return [DatasetPermission()]


class DatasetCommentPermission(PermissionPolicyDRFPermission[DataPointComment, None]):
    class Meta:
        model = DataPoint

    def has_permission(self, request: Request, view: APIView):
        dataset_uuid = view.kwargs['dataset_uuid']
        dataset = Dataset.objects.get(uuid=dataset_uuid)
        pp = dataset.permission_policy()
        user = user_or_none(request.user)
        if not user:
            return False
        action = self.http_method_to_django_action(request.method)
        if action != 'view':
            return True  # We want to return 405 instead of 403
        return pp.user_has_any_permission_for_instance(user, ['view'], dataset)

    def get_create_context_from_api_view(self, view: APIView) -> None:
        return None


class DatasetCommentsViewSet(BaseDatasetCommentsViewSet):
    @override
    def get_permissions(self):
        return[DatasetCommentPermission()]


class DataSourceViewSet(BaseDataSourceViewSet):
    pass


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

nested_routers: list[SimpleRouter]  = []
nested_routers.append(dataset_router)
nested_routers.append(dimension_router)
nested_routers.append(datasetschema_router)
nested_routers.append(datapoint_router)
