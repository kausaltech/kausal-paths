from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NoReturn, TypeGuard, cast
from uuid import UUID

import strawberry as sb
from django.core.exceptions import PermissionDenied, ValidationError
from django.db import transaction
from django.db.models import Q
from django.utils import timezone
from strawberry import Maybe
from strawberry_django.fields.types import OperationInfo

from kausal_common.datasets.api import DataPointSerializer
from kausal_common.datasets.models import (
    DataPoint,
    DataPointComment,
    DataPointCommentReviewState,
    Dataset,
    DatasetSourceReference,
    DataSource,
)
from kausal_common.strawberry.helpers import get_or_error
from kausal_common.users import user_or_bust

from paths import gql

from datasets.validation_rules import ValidationRule, ValidationRuleSpecInput
from nodes.change_ops import gql_change_operation, record_change
from nodes.dataset_materialization import refresh_dataset_materialization
from nodes.graphql.types.problems import DatasetValidationViolationType
from nodes.models import InstanceConfig

from .types import DataPointCommentType, DataPointType, DatasetSourceReferenceType, MetricValidationRuleType

if TYPE_CHECKING:
    from strawberry import Some

    from kausal_common.datasets.models import DatasetMetric, DatasetMetricValidationRule


@sb.input
class CreateDatasetSourceReferenceInput:
    """Create a source reference. Exactly one of data_point_id or to_dataset must be set."""

    data_source_id: UUID
    data_point_id: UUID | None = None
    to_dataset: bool = False


@sb.input
class CreateDataPointCommentInput:
    text: str
    is_sticky: bool = False
    is_review: bool = False
    review_state: DataPointCommentReviewState | None = None


@sb.input
class UpdateDataPointCommentInput:
    text: Maybe[str]
    is_sticky: Maybe[bool]
    is_review: Maybe[bool]
    review_state: Maybe[DataPointCommentReviewState | None]


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


@sb.input
class UpdateDataPointItemInput:
    data_point_id: sb.ID
    input: UpdateDataPointInput


@sb.type
class DataPointsMutationResult:
    data_points: list[DataPointType]
    violations: list[DatasetValidationViolationType] = sb.field(
        description=(
            'Validation-rule violations of the dataset after this write. These do not undo the write, but they block publication.'
        ),
    )


@sb.type
class DeleteDataPointsResult:
    deleted_data_point_ids: list[sb.ID]
    violations: list[DatasetValidationViolationType] = sb.field(
        description=(
            'Validation-rule violations of the dataset after this write. These do not undo the write, but they block publication.'
        ),
    )


@sb.input
class ValidationRuleInput:
    rule: ValidationRuleSpecInput = sb.field(
        description='The rule; set exactly one variant field.',
    )
    uuid: UUID | None = sb.field(
        default=None,
        description='Identity of an existing rule to keep; omit for a new rule.',
    )


@sb.type
class MetricValidationRulesResult:
    validation_rules: list[MetricValidationRuleType]
    violations: list[DatasetValidationViolationType] = sb.field(
        description='Validation-rule violations of the dataset under the new rule set.',
    )


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


def _replace_metric_validation_rules(
    metric: DatasetMetric,
    rules: list[tuple[UUID | None, ValidationRule]],
) -> list[DatasetMetricValidationRule]:
    """
    Reconcile the metric's rule rows against the (uuid, rule) list, recording changes.

    Entries are matched by uuid: known uuids are updated in place, entries
    without a uuid create new rows, and existing rows missing from the list
    are deleted. List position becomes the order.
    """
    from kausal_common.datasets.models import DatasetMetricValidationRule

    def rule_snapshot(row: DatasetMetricValidationRule) -> dict[str, Any]:
        return {'uuid': str(row.uuid), 'rule': row.rule, 'order': row.order}

    existing = {row.uuid: row for row in metric.validation_rules.all()}
    kept: set[UUID] = set()
    result_rows: list[DatasetMetricValidationRule] = []
    for order, (rule_uuid, rule) in enumerate(rules):
        blob = rule.model_dump(mode='json')
        if rule_uuid is not None:
            row = existing.get(rule_uuid)
            if row is None:
                raise ValidationError(f'Unknown validation rule uuid {rule_uuid}')
            before = rule_snapshot(row)
            row.rule = blob
            row.order = order
            row.save(update_fields=['rule', 'order'])
            record_change(row, action='dataset.metric.validation_rule.update', before=before, after=rule_snapshot(row))
        else:
            row = DatasetMetricValidationRule.objects.create(metric=metric, rule=blob, order=order)
            record_change(row, action='dataset.metric.validation_rule.create', before=None, after=rule_snapshot(row))
        kept.add(row.uuid)
        result_rows.append(row)
    for row in existing.values():
        if row.uuid in kept:
            continue
        record_change(row, action='dataset.metric.validation_rule.delete', before=rule_snapshot(row), after=None)
        row.delete()
    return result_rows


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
    def _save_dataset(root: Me, info: gql.Info, dataset: Dataset | None = None) -> None:
        try:
            user = user_or_bust(info.context.user)
        except ValueError as exc:
            raise PermissionDenied('Permission denied') from exc
        refresh_dataset_materialization(dataset or root.dataset, user=user, enforce_edit_rules=True)

    @staticmethod
    def _current_violations(root: Me) -> list[DatasetValidationViolationType]:
        """Read the dataset's persisted violations, as recorded by the refresh this mutation just ran."""
        from datasets.validation import load_violations
        from nodes.models import DatasetMaterialization

        materialization = DatasetMaterialization.objects.filter(dataset=root.dataset).first()
        if materialization is None:
            return []
        return [
            DatasetValidationViolationType.from_violation(violation)
            for violation in load_violations(materialization.validation_violations)
        ]

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

    @staticmethod
    def _require_batch(input: list[Any]) -> None:
        if not input:
            raise ValidationError('At least one data point is required')

    @staticmethod
    def _require_unique_data_point_ids(data_point_ids: list[sb.ID]) -> None:
        if len({str(data_point_id) for data_point_id in data_point_ids}) != len(data_point_ids):
            raise ValidationError('Each data point may occur only once in a batch')

    @staticmethod
    def _create_data_points(info: gql.Info, root: Me, input: list[CreateDataPointInput]) -> list[DataPoint]:
        DatasetEditorMutation._require_batch(input)
        dataset = root.dataset
        if not DataPoint.gql_create_allowed(info, cast('Any', dataset)):
            raise PermissionDenied('Permission denied for create')

        try:
            user = user_or_bust(info.context.user)
        except ValueError as exc:
            raise PermissionDenied('Permission denied') from exc

        created: list[DataPoint] = []
        with transaction.atomic():
            dataset = Dataset.objects.select_for_update().get(pk=dataset.pk)
            with gql_change_operation(info, root.instance, action='dataset.datapoint.create'):
                for item in input:
                    serializer = DataPointSerializer(
                        data=DatasetEditorMutation._serialize_input(item),
                        context=DatasetEditorMutation._serializer_context(root),
                    )
                    if not serializer.is_valid():
                        _raise_serializer_errors(serializer)

                    data_point = serializer.save(dataset=dataset, last_modified_by=user)
                    record_change(
                        data_point,
                        action='dataset.datapoint.create',
                        before=None,
                        after=DatasetEditorMutation._data_point_snapshot(data_point),
                    )
                    created.append(data_point)
                DatasetEditorMutation._save_dataset(root, info, dataset=dataset)
        return created

    @gql.mutation(description='Create data points', graphql_type=DataPointsMutationResult)
    @staticmethod
    def create_data_points(
        info: gql.Info,
        root: sb.Parent[Me],
        input: list[CreateDataPointInput],
    ) -> DataPointsMutationResult:
        created = DatasetEditorMutation._create_data_points(info, root, input)
        return DataPointsMutationResult(
            data_points=[DataPointType.from_model(data_point) for data_point in created],
            violations=DatasetEditorMutation._current_violations(root),
        )

    @gql.mutation(
        description='Create a data point',
        graphql_type=DataPointType,
        deprecation_reason='Use createDataPoints instead.',
    )
    @staticmethod
    def create_data_point(info: gql.Info, root: sb.Parent[Me], input: CreateDataPointInput) -> DataPointType:
        data_point = DatasetEditorMutation._create_data_points(info, root, [input])[0]
        return DataPointType.from_model(data_point)

    @staticmethod
    def _update_data_points(info: gql.Info, root: Me, input: list[UpdateDataPointItemInput]) -> list[DataPoint]:
        DatasetEditorMutation._require_batch(input)
        DatasetEditorMutation._require_unique_data_point_ids([item.data_point_id for item in input])
        try:
            user = user_or_bust(info.context.user)
        except ValueError as exc:
            raise PermissionDenied('Permission denied') from exc

        updated_data_points: list[DataPoint] = []
        with transaction.atomic():
            dataset = Dataset.objects.select_for_update().get(pk=root.dataset.pk)
            with gql_change_operation(info, root.instance, action='dataset.datapoint.update'):
                for item in input:
                    data_point = get_or_error(
                        info,
                        dataset.data_points.get_queryset(),
                        uuid=str(item.data_point_id),
                        for_action='change',
                    )
                    serializer = DataPointSerializer(
                        data_point,
                        data=DatasetEditorMutation._serialize_input(item.input),
                        partial=True,
                        context=DatasetEditorMutation._serializer_context(root),
                    )
                    if not serializer.is_valid():
                        _raise_serializer_errors(serializer)

                    before = DatasetEditorMutation._data_point_snapshot(data_point)
                    updated = serializer.save(last_modified_by=user)
                    record_change(
                        updated,
                        action='dataset.datapoint.update',
                        before=before,
                        after=DatasetEditorMutation._data_point_snapshot(updated),
                    )
                    updated_data_points.append(updated)
                DatasetEditorMutation._save_dataset(root, info, dataset=dataset)
        return updated_data_points

    @gql.mutation(description='Update data points', graphql_type=DataPointsMutationResult)
    @staticmethod
    def update_data_points(
        info: gql.Info,
        root: sb.Parent[Me],
        input: list[UpdateDataPointItemInput],
    ) -> DataPointsMutationResult:
        updated = DatasetEditorMutation._update_data_points(info, root, input)
        return DataPointsMutationResult(
            data_points=[DataPointType.from_model(data_point) for data_point in updated],
            violations=DatasetEditorMutation._current_violations(root),
        )

    @gql.mutation(
        description='Update a data point',
        graphql_type=DataPointType,
        deprecation_reason='Use updateDataPoints instead.',
    )
    @staticmethod
    def update_data_point(
        info: gql.Info,
        root: sb.Parent[Me],
        data_point_id: sb.ID,
        input: UpdateDataPointInput,
    ) -> DataPointType:
        updated = DatasetEditorMutation._update_data_points(
            info,
            root,
            [UpdateDataPointItemInput(data_point_id=data_point_id, input=input)],
        )[0]
        return DataPointType.from_model(updated)

    @staticmethod
    def _delete_data_points(info: gql.Info, root: Me, data_point_ids: list[sb.ID]) -> list[sb.ID]:
        DatasetEditorMutation._require_batch(data_point_ids)
        DatasetEditorMutation._require_unique_data_point_ids(data_point_ids)
        deleted_ids: list[sb.ID] = []
        with transaction.atomic():
            dataset = Dataset.objects.select_for_update().get(pk=root.dataset.pk)
            data_points = [
                get_or_error(
                    info,
                    dataset.data_points.get_queryset(),
                    uuid=str(data_point_id),
                    for_action='delete',
                )
                for data_point_id in data_point_ids
            ]
            with gql_change_operation(info, root.instance, action='dataset.datapoint.delete'):
                for data_point in data_points:
                    deleted_ids.append(sb.ID(str(data_point.uuid)))
                    record_change(
                        data_point,
                        action='dataset.datapoint.delete',
                        before=DatasetEditorMutation._data_point_snapshot(data_point),
                        after=None,
                    )
                    data_point.delete()
                DatasetEditorMutation._save_dataset(root, info, dataset=dataset)
        return deleted_ids

    @gql.mutation(
        description=(
            'Replace the validation rules of one metric of this dataset. '
            'Entries are matched by uuid: known uuids are updated in place (preserving '
            'rule identity for violation tracking), entries without a uuid are created, '
            'and existing rules missing from the list are deleted. Setting rules never '
            'blocks on pre-existing data; the resulting violations are returned.'
        ),
        graphql_type=MetricValidationRulesResult,
    )
    @staticmethod
    def set_metric_validation_rules(
        info: gql.Info,
        root: sb.Parent[Me],
        metric_id: UUID,
        rules: list[ValidationRuleInput],
    ) -> MetricValidationRulesResult:
        from pydantic import ValidationError as PydanticValidationError

        from kausal_common.datasets.models import DatasetMetric

        dataset = root.dataset
        if dataset.schema is None:
            raise ValidationError('Dataset has no schema')
        try:
            metric = DatasetMetric.objects.get(schema=dataset.schema, uuid=metric_id)
        except DatasetMetric.DoesNotExist as exc:
            raise ValidationError(f'Metric {metric_id} does not belong to this dataset') from exc
        converted: list[tuple[UUID | None, ValidationRule]] = []
        for item in rules:
            try:
                converted.append((item.uuid, item.rule.to_rule()))
            except (PydanticValidationError, ValueError) as error:
                raise ValidationError(f'Invalid validation rule: {error}') from error

        try:
            user = user_or_bust(info.context.user)
        except ValueError as exc:
            raise PermissionDenied('Permission denied') from exc

        with transaction.atomic():
            locked_dataset = Dataset.objects.select_for_update().get(pk=dataset.pk)
            with gql_change_operation(info, root.instance, action='dataset.metric.validation_rules.set'):
                result_rows = _replace_metric_validation_rules(metric, converted)
                # Setting rules is not a data edit: re-evaluate and persist the
                # violations, but never block on pre-existing data.
                refresh_dataset_materialization(locked_dataset, user=user)
        return MetricValidationRulesResult(
            validation_rules=[MetricValidationRuleType.from_model(row) for row in result_rows],
            violations=DatasetEditorMutation._current_violations(root),
        )

    @gql.mutation(description='Delete data points', graphql_type=DeleteDataPointsResult)
    @staticmethod
    def delete_data_points(
        root: sb.Parent[Me],
        info: gql.Info,
        data_point_ids: list[sb.ID],
    ) -> DeleteDataPointsResult:
        return DeleteDataPointsResult(
            deleted_data_point_ids=DatasetEditorMutation._delete_data_points(info, root, data_point_ids),
            violations=DatasetEditorMutation._current_violations(root),
        )

    @gql.mutation(
        description='Delete a data point',
        graphql_type=OperationInfo | None,
        deprecation_reason='Use deleteDataPoints instead.',
    )
    @staticmethod
    def delete_data_point(root: sb.Parent[Me], info: gql.Info, data_point_id: sb.ID) -> OperationInfo | None:
        DatasetEditorMutation._delete_data_points(info, root, [data_point_id])
        return None

    # ------------------------------------------------------------------
    # Data point comments
    # ------------------------------------------------------------------

    @staticmethod
    def _data_point_comment_snapshot(comment: DataPointComment) -> dict[str, Any]:
        return {
            'uuid': str(comment.uuid),
            'data_point_uuid': str(comment.data_point.uuid) if comment.data_point else None,
            'text': comment.text,
            'is_sticky': comment.is_sticky,
            'is_review': comment.is_review,
            'review_state': comment.review_state or None,
            'is_soft_deleted': comment.is_soft_deleted,
            'resolved_at': comment.resolved_at.isoformat() if comment.resolved_at else None,
            'resolved_by_uuid': str(comment.resolved_by.uuid) if comment.resolved_by else None,
        }

    @staticmethod
    def _get_data_point(info: gql.Info, root: Me, data_point_id: sb.ID) -> DataPoint:
        return get_or_error(
            info,
            root.dataset.data_points.get_queryset(),
            uuid=str(data_point_id),
            for_action='change',
        )

    @staticmethod
    def _get_comment(info: gql.Info, root: Me, comment_id: sb.ID, for_action: Any = 'change') -> DataPointComment:
        return get_or_error(
            info,
            DataPointComment.objects.filter(data_point__dataset=root.dataset),
            uuid=str(comment_id),
            for_action=for_action,
        )

    @gql.mutation(description='Create a comment on a data point', graphql_type=DataPointCommentType)
    @staticmethod
    def create_data_point_comment(
        info: gql.Info,
        root: sb.Parent[Me],
        data_point_id: sb.ID,
        input: CreateDataPointCommentInput,
    ) -> DataPointCommentType:
        data_point = DatasetEditorMutation._get_data_point(info, root, data_point_id)
        if not DataPointComment.gql_create_allowed(info, cast('Any', data_point)):
            raise PermissionDenied('Permission denied for create')

        try:
            user = user_or_bust(info.context.user)
        except ValueError as exc:
            raise PermissionDenied('Permission denied') from exc

        with gql_change_operation(info, root.instance, action='dataset.datapoint.comment.create'):
            comment = DataPointComment.objects.create(
                data_point=data_point,
                text=input.text,
                is_sticky=input.is_sticky,
                is_review=input.is_review,
                review_state=input.review_state,
                created_by=user,
                last_modified_by=user,
            )
            record_change(
                comment,
                action='dataset.datapoint.comment.create',
                before=None,
                after=DatasetEditorMutation._data_point_comment_snapshot(comment),
            )
            DatasetEditorMutation._save_dataset(root, info)
        return cast('DataPointCommentType', comment)

    @gql.mutation(description='Update a comment on a data point', graphql_type=DataPointCommentType)
    @staticmethod
    def update_data_point_comment(
        info: gql.Info,
        root: sb.Parent[Me],
        comment_id: sb.ID,
        input: UpdateDataPointCommentInput,
    ) -> DataPointCommentType:
        comment = DatasetEditorMutation._get_comment(info, root, comment_id, for_action='change')

        try:
            user = user_or_bust(info.context.user)
        except ValueError as exc:
            raise PermissionDenied('Permission denied') from exc

        with gql_change_operation(info, root.instance, action='dataset.datapoint.comment.update'):
            before = DatasetEditorMutation._data_point_comment_snapshot(comment)
            update_fields: list[str] = []
            if _is_maybe_set(input.text):
                comment.text = input.text.value
                update_fields.append('text')
            if _is_maybe_set(input.is_sticky):
                comment.is_sticky = input.is_sticky.value
                update_fields.append('is_sticky')
            if _is_maybe_set(input.is_review):
                comment.is_review = input.is_review.value
                update_fields.append('is_review')
            if _is_maybe_set(input.review_state):
                comment.review_state = input.review_state.value
                update_fields.append('review_state')
            comment.last_modified_by = user
            update_fields.append('last_modified_by')
            update_fields.append('last_modified_at')
            comment.save(update_fields=update_fields)
            record_change(
                comment,
                action='dataset.datapoint.comment.update',
                before=before,
                after=DatasetEditorMutation._data_point_comment_snapshot(comment),
            )
            DatasetEditorMutation._save_dataset(root, info)
        return cast('DataPointCommentType', comment)

    @gql.mutation(description='Soft-delete a comment on a data point', graphql_type=OperationInfo | None)
    @staticmethod
    def delete_data_point_comment(
        root: sb.Parent[Me],
        info: gql.Info,
        comment_id: sb.ID,
    ) -> OperationInfo | None:
        comment = DatasetEditorMutation._get_comment(info, root, comment_id, for_action='delete')

        try:
            user = user_or_bust(info.context.user)
        except ValueError as exc:
            raise PermissionDenied('Permission denied') from exc

        with gql_change_operation(info, root.instance, action='dataset.datapoint.comment.delete'):
            record_change(
                comment,
                action='dataset.datapoint.comment.delete',
                before=DatasetEditorMutation._data_point_comment_snapshot(comment),
                after=None,
            )
            comment.soft_delete(user)
            DatasetEditorMutation._save_dataset(root, info)
        return None

    @gql.mutation(description='Mark a review comment as resolved', graphql_type=DataPointCommentType)
    @staticmethod
    def resolve_data_point_comment(
        info: gql.Info,
        root: sb.Parent[Me],
        comment_id: sb.ID,
    ) -> DataPointCommentType:
        comment = DatasetEditorMutation._get_comment(info, root, comment_id, for_action='change')

        try:
            user = user_or_bust(info.context.user)
        except ValueError as exc:
            raise PermissionDenied('Permission denied') from exc

        with gql_change_operation(info, root.instance, action='dataset.datapoint.comment.resolve'):
            before = DatasetEditorMutation._data_point_comment_snapshot(comment)
            comment.review_state = DataPointComment.ReviewState.RESOLVED
            comment.resolved_at = timezone.now()
            comment.resolved_by = user
            comment.last_modified_by = user
            comment.save(
                update_fields=['review_state', 'resolved_at', 'resolved_by', 'last_modified_by', 'last_modified_at'],
            )
            record_change(
                comment,
                action='dataset.datapoint.comment.resolve',
                before=before,
                after=DatasetEditorMutation._data_point_comment_snapshot(comment),
            )
            DatasetEditorMutation._save_dataset(root, info)
        return cast('DataPointCommentType', comment)

    @gql.mutation(description='Mark a review comment as unresolved', graphql_type=DataPointCommentType)
    @staticmethod
    def unresolve_data_point_comment(
        info: gql.Info,
        root: sb.Parent[Me],
        comment_id: sb.ID,
    ) -> DataPointCommentType:
        comment = DatasetEditorMutation._get_comment(info, root, comment_id, for_action='change')

        try:
            user = user_or_bust(info.context.user)
        except ValueError as exc:
            raise PermissionDenied('Permission denied') from exc

        with gql_change_operation(info, root.instance, action='dataset.datapoint.comment.unresolve'):
            before = DatasetEditorMutation._data_point_comment_snapshot(comment)
            comment.review_state = DataPointComment.ReviewState.UNRESOLVED
            comment.resolved_at = None
            comment.resolved_by = None
            comment.last_modified_by = user
            comment.save(
                update_fields=['review_state', 'resolved_at', 'resolved_by', 'last_modified_by', 'last_modified_at'],
            )
            record_change(
                comment,
                action='dataset.datapoint.comment.unresolve',
                before=before,
                after=DatasetEditorMutation._data_point_comment_snapshot(comment),
            )
            DatasetEditorMutation._save_dataset(root, info)
        return cast('DataPointCommentType', comment)

    # ------------------------------------------------------------------
    # DatasetSourceReference
    # ------------------------------------------------------------------

    @staticmethod
    def _source_reference_snapshot(ref: DatasetSourceReference) -> dict[str, Any]:
        return {
            'uuid': str(ref.uuid),
            'data_source_uuid': str(ref.data_source.uuid),
            'data_point_uuid': str(ref.data_point.uuid) if ref.data_point else None,
            'dataset_uuid': str(ref.dataset.uuid) if ref.dataset else None,
        }

    @gql.mutation(
        description='Attach a DataSource to either a data point or this dataset.',
        graphql_type=DatasetSourceReferenceType,
    )
    @staticmethod
    def create_source_reference(
        info: gql.Info,
        root: sb.Parent[Me],
        input: CreateDatasetSourceReferenceInput,
    ) -> DatasetSourceReferenceType:
        dataset = root.dataset

        if (input.data_point_id is None) == (not input.to_dataset):
            raise ValidationError('Exactly one of data_point_id or to_dataset must be set.')

        data_point: DataPoint | None = None
        if input.data_point_id is not None:
            data_point = get_or_error(
                info,
                dataset.data_points.get_queryset(),
                uuid=str(input.data_point_id),
                for_action='change',
            )

        data_source = get_or_error(
            info,
            DataSource.objects.filter(scope_id=root.instance.pk),
            uuid=str(input.data_source_id),
        )

        if not DatasetSourceReference.gql_create_allowed(info, cast('Any', dataset)):
            raise PermissionDenied('Permission denied for create')

        try:
            user = user_or_bust(info.context.user)
        except ValueError as exc:
            raise PermissionDenied('Permission denied') from exc

        with gql_change_operation(info, root.instance, action='dataset.source_reference.create'):
            ref = DatasetSourceReference.objects.create(
                data_source=data_source,
                data_point=data_point,
                dataset=dataset if input.to_dataset else None,
                created_by=user,
                last_modified_by=user,
            )
            record_change(
                ref,
                action='dataset.source_reference.create',
                before=None,
                after=DatasetEditorMutation._source_reference_snapshot(ref),
            )
            DatasetEditorMutation._save_dataset(root, info)
        return cast('DatasetSourceReferenceType', ref)

    @gql.mutation(description='Remove a source reference.', graphql_type=OperationInfo | None)
    @staticmethod
    def delete_source_reference(
        info: gql.Info,
        root: sb.Parent[Me],
        reference_id: sb.ID,
    ) -> OperationInfo | None:
        ref = get_or_error(
            info,
            DatasetSourceReference.objects.filter(
                Q(dataset=root.dataset) | Q(data_point__dataset=root.dataset),
            ),
            uuid=str(reference_id),
            for_action='delete',
        )
        with gql_change_operation(info, root.instance, action='dataset.source_reference.delete'):
            record_change(
                ref,
                action='dataset.source_reference.delete',
                before=DatasetEditorMutation._source_reference_snapshot(ref),
                after=None,
            )
            ref.delete()
            DatasetEditorMutation._save_dataset(root, info)
        return None
