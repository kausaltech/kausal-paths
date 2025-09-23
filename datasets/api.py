from __future__ import annotations

from typing import override

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
from kausal_common.datasets.models import DataPointComment


class DataPointCommentViewSet(BaseDataPointCommentViewSet):
    @override
    def get_permissions(self):
        """Instantiate and return the list of permissions that this view requires."""
        return [PermissionPolicyDRFPermission(model=DataPointComment)]


class DataPointSourceReferenceViewSet(BaseDataPointSourceReferenceViewSet):
    pass


class DataPointViewSet(BaseDataPointViewSet):
    pass


class DatasetCommentsViewSet(BaseDatasetCommentsViewSet):
    pass


class DatasetMetricViewSet(BaseDatasetMetricViewSet):
    pass


class DatasetSchemaViewSet(BaseDatasetSchemaViewSet):
    pass


class DatasetSourceReferenceViewSet(BaseDatasetSourceReferenceViewSet):
    pass


class DatasetViewSet(BaseDatasetViewSet):
    pass


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
