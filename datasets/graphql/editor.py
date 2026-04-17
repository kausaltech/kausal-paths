from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NoReturn, TypeGuard, cast
from uuid import UUID

import strawberry as sb
from django.core.exceptions import PermissionDenied, ValidationError
from strawberry import Maybe
from strawberry_django.fields.types import OperationInfo

from kausal_common.datasets.api import DataPointSerializer
from kausal_common.datasets.models import DataPoint, Dataset
from kausal_common.strawberry.helpers import get_or_error
from kausal_common.users import user_or_bust

from paths import gql

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
        root.dataset.save(update_fields=['last_modified_by'])
        root.dataset.clear_scope_instance_cache()

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
        data_point = serializer.save(dataset=dataset, last_modified_by=user)
        DatasetEditorMutation._save_dataset(root, info)
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
        updated = serializer.save(last_modified_by=user)
        DatasetEditorMutation._save_dataset(root, info)
        return DataPointType.from_model(updated)

    @gql.mutation(description='Delete a data point', graphql_type=OperationInfo | None)
    @staticmethod
    def delete_data_point(root: sb.Parent[Me], info: gql.Info, data_point_id: sb.ID) -> OperationInfo | None:
        data_point = get_or_error(info, root.dataset.data_points.get_queryset(), uuid=str(data_point_id), for_action='delete')
        DatasetEditorMutation._save_dataset(root, info)
        data_point.delete()
        return None
