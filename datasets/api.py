from __future__ import annotations

from typing import TYPE_CHECKING, override

from rest_framework import serializers
from rest_framework.routers import DefaultRouter, SimpleRouter

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
)

if TYPE_CHECKING:
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


class DatasetSourceReferenceViewSet(BaseDatasetSourceReferenceViewSet):
    @override
    def get_permissions(self):
        return [DatasetSourceReferencePermission()]


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


class DatasetMetricPermission(NestedResourcePermissionPolicyDRFPermission[DatasetMetric, None, DatasetSchema]):
    class Meta:
        model = DatasetMetric
        view_kwargs_parent_key = 'datasetschema_uuid'
        nested_parent_model = DatasetSchema
        nested_parent_key_field = 'uuid'
        allowed_actions = { 'view' }

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
