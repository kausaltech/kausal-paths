from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NoReturn, TypeGuard, cast
from uuid import UUID

import strawberry as sb
from django.core.exceptions import PermissionDenied, ValidationError
from django.utils import timezone
from strawberry import Maybe
from strawberry_django.fields.types import OperationInfo

from kausal_common.datasets.api import DataPointSerializer
from kausal_common.datasets.models import DataPoint, Dataset
from kausal_common.strawberry.helpers import get_or_error
from kausal_common.users import user_or_bust

from paths import gql

from nodes.change_ops import gql_change_operation, record_change
from nodes.models import InstanceConfig

from .types import DataPointType

if TYPE_CHECKING:
    from strawberry import Some


@sb.input
class CreateDataPointInput:
    date: date
    value: float | None
    metric_id: UUID
    dimension_category_ids: list[UUID] | None = None


@sb.input
class UpdateDataPointInput:
    date: Maybe[date]
    value: Maybe[float | None]
    metric_id: Maybe[UUID]
    dimension_category_ids: Maybe[list[UUID]]


def _is_maybe_set[T](maybe: Some[T] | None) -> TypeGuard[Some[T]]:
    return maybe is not None and maybe is not sb.UNSET


def _stringify_errors(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _stringify_errors(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_stringify_errors(item) for item in value]
    return str(value)


def _raise_serializer_errors(serializer: DataPointSerializer) -> NoReturn:
    raise ValidationError(_stringify_errors(serializer.errors))


@sb.type
class DatasetEditorMutation:
    dataset: sb.Private[Dataset]
    instance: sb.Private[InstanceConfig]
    type Me = DatasetEditorMutation

    @staticmethod
    def _serializer_context(root: Me) -> dict[str, Any]:
        return {'view': SimpleNamespace(kwargs={'dataset_uuid': str(root.dataset.uuid)})}

    @staticmethod
    def _serialize_input(input: CreateDataPointInput | UpdateDataPointInput) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if isinstance(input, UpdateDataPointInput):
            if _is_maybe_set(input.date):
                data['date'] = input.date.value.isoformat()
            if _is_maybe_set(input.value):
                data['value'] = input.value.value
            if _is_maybe_set(input.metric_id):
                data['metric'] = str(input.metric_id.value)
            if _is_maybe_set(input.dimension_category_ids):
                data['dimension_categories'] = [str(category_id) for category_id in input.dimension_category_ids.value]
            return data

        if input.date is not None:
            data['date'] = input.date.isoformat()
        data['value'] = input.value
        data['metric'] = str(input.metric_id)
        data['dimension_categories'] = [str(category_id) for category_id in input.dimension_category_ids or []]
        return data

    @staticmethod
    def _save_dataset(root: Me, info: gql.Info) -> None:
        try:
            user = user_or_bust(info.context.user)
        except ValueError as exc:
            raise PermissionDenied('Permission denied') from exc
        root.dataset.last_modified_by = user
        root.dataset.last_modified_at = timezone.now()
        root.dataset.save(update_fields=['last_modified_by', 'last_modified_at'])
        root.dataset.clear_scope_instance_cache()

    @staticmethod
    def _data_point_snapshot(dp: DataPoint) -> dict[str, Any]:
        """Lightweight snapshot for change tracking."""
        # Decimal → float: JSONField can't serialize Decimal natively and
        # DataPoint values don't need cents-grade precision.
        return {
            'uuid': str(dp.uuid),
            'dataset_uuid': str(dp.dataset.uuid),
            'date': dp.date.isoformat() if dp.date else None,
            'value': float(dp.value) if dp.value is not None else None,
            'metric_uuid': str(dp.metric.uuid) if dp.metric else None,
            'dimension_category_uuids': [str(cat.uuid) for cat in dp.dimension_categories.all()],
        }

    @gql.mutation(description='Create a data point', graphql_type=DataPointType)
    @staticmethod
    def create_data_point(info: gql.Info, root: sb.Parent[Me], input: CreateDataPointInput) -> DataPointType:
        dataset = root.dataset
        if not DataPoint.gql_create_allowed(info, cast('Any', dataset)):
            raise PermissionDenied('Permission denied for create')

        serializer = DataPointSerializer(
            data=DatasetEditorMutation._serialize_input(input),
            context=DatasetEditorMutation._serializer_context(root),
        )
        if not serializer.is_valid():
            _raise_serializer_errors(serializer)

        try:
            user = user_or_bust(info.context.user)
        except ValueError as exc:
            raise PermissionDenied('Permission denied') from exc
        with gql_change_operation(info, root.instance, action='dataset.datapoint.create'):
            data_point = serializer.save(dataset=dataset, last_modified_by=user)
            DatasetEditorMutation._save_dataset(root, info)
            record_change(
                data_point,
                action='dataset.datapoint.create',
                before=None,
                after=DatasetEditorMutation._data_point_snapshot(data_point),
            )
        return DataPointType.from_model(data_point)

    @gql.mutation(description='Update a data point', graphql_type=DataPointType)
    @staticmethod
    def update_data_point(
        info: gql.Info,
        root: sb.Parent[Me],
        data_point_id: sb.ID,
        input: UpdateDataPointInput,
    ) -> DataPointType:
        data_point = get_or_error(info, root.dataset.data_points.get_queryset(), uuid=str(data_point_id), for_action='change')

        serializer = DataPointSerializer(
            data_point,
            data=DatasetEditorMutation._serialize_input(input),
            partial=True,
            context=DatasetEditorMutation._serializer_context(root),
        )
        if not serializer.is_valid():
            _raise_serializer_errors(serializer)

        try:
            user = user_or_bust(info.context.user)
        except ValueError as exc:
            raise PermissionDenied('Permission denied') from exc
        with gql_change_operation(info, root.instance, action='dataset.datapoint.update'):
            before = DatasetEditorMutation._data_point_snapshot(data_point)
            updated = serializer.save(last_modified_by=user)
            DatasetEditorMutation._save_dataset(root, info)
            record_change(
                updated,
                action='dataset.datapoint.update',
                before=before,
                after=DatasetEditorMutation._data_point_snapshot(updated),
            )
        return DataPointType.from_model(updated)

    @gql.mutation(description='Delete a data point', graphql_type=OperationInfo | None)
    @staticmethod
    def delete_data_point(root: sb.Parent[Me], info: gql.Info, data_point_id: sb.ID) -> OperationInfo | None:
        data_point = get_or_error(info, root.dataset.data_points.get_queryset(), uuid=str(data_point_id), for_action='delete')
        with gql_change_operation(info, root.instance, action='dataset.datapoint.delete'):
            record_change(
                data_point,
                action='dataset.datapoint.delete',
                before=DatasetEditorMutation._data_point_snapshot(data_point),
                after=None,
            )
            DatasetEditorMutation._save_dataset(root, info)
            data_point.delete()
        return None
