"""GraphQL surface for submissions: the reporting record of an instance and its lifecycle mutations."""

from datetime import datetime
from typing import TYPE_CHECKING, Annotated

import strawberry as sb

from kausal_common.strawberry.helpers import get_or_error
from kausal_common.strawberry.registry import register_strawberry_type
from kausal_common.users import user_or_none

from paths import gql

from frameworks import submissions as ops
from frameworks.models import Submission, SubmissionKind, SubmissionStatus
from nodes.graphql.types.constraints import ConstraintViolationsType
from nodes.graphql.types.problems import DatasetValidationViolationsType
from nodes.models import InstanceConfig
from users.models import User

if TYPE_CHECKING:
    from paths.types import PathsGQLInfo

    from users.schema import UserType


@register_strawberry_type
@sb.type(name='Submission')
class SubmissionType:
    """
    One balance of an instance for a period: draft, in review, final or superseded.

    A final submission is frozen: it refers to the published instance revision it
    was finalised with. A correction is a new submission that supersedes it.
    """

    id: sb.ID
    kind: SubmissionKind
    period_start: int
    period_end: int
    status: SubmissionStatus
    created_at: datetime
    created_by: User | None = sb.field(graphql_type=Annotated['UserType', sb.lazy('users.schema')] | None)
    last_modified_at: datetime
    finalised_at: datetime | None
    finalised_by: User | None = sb.field(graphql_type=Annotated['UserType', sb.lazy('users.schema')] | None)
    supersedes_id: sb.ID | None = sb.field(description='The final submission this one corrects, if any.')

    @classmethod
    def from_model(cls, obj: Submission) -> SubmissionType:
        return cls(
            id=sb.ID(str(obj.uuid)),
            kind=SubmissionKind(obj.kind),
            period_start=obj.period_start,
            period_end=obj.period_end,
            status=SubmissionStatus(obj.status),
            created_at=obj.created_at,
            created_by=obj.created_by,
            last_modified_at=obj.last_modified_at,
            finalised_at=obj.finalised_at,
            finalised_by=obj.finalised_by,
            supersedes_id=sb.ID(str(obj.supersedes.uuid)) if obj.supersedes is not None else None,
        )


def submissions_for(ic: InstanceConfig, info: gql.Info | PathsGQLInfo) -> list[SubmissionType]:
    """Open submissions are visible only to those who may edit the instance."""
    include_open = ic.gql_action_allowed(info, 'change', raise_on_denied=False)
    return [SubmissionType.from_model(obj) for obj in ops.visible_submissions(ic, include_open=include_open)]


@sb.type
class DiscardSubmissionResult:
    discarded_submission_id: sb.ID


@sb.type
class SubmissionMutations:
    instance: sb.Private[InstanceConfig]
    type Me = SubmissionMutations

    @staticmethod
    def _get(info: gql.Info, root: SubmissionMutations, submission_id: sb.ID) -> Submission:
        return get_or_error(info, Submission.objects.filter(instance_config=root.instance), uuid=str(submission_id))

    @gql.mutation(
        description=(
            'Open a submission for a period. An inventory covers one historical year (periodEnd defaults to '
            'periodStart). If the period already has a final submission, the new one corrects it.'
        ),
        graphql_type=SubmissionType,
    )
    @staticmethod
    def create(
        info: gql.Info,
        root: sb.Parent[Me],
        period_start: int,
        period_end: int | None = None,
        kind: SubmissionKind = SubmissionKind.INVENTORY,
    ) -> SubmissionType:
        obj = ops.create_submission(
            root.instance,
            period_start=period_start,
            period_end=period_end,
            kind=kind,
            user=user_or_none(info.context.user),
        )
        return SubmissionType.from_model(obj)

    @gql.mutation(description='Move a draft submission into review.', graphql_type=SubmissionType)
    @staticmethod
    def request_review(info: gql.Info, root: sb.Parent[Me], submission_id: sb.ID) -> SubmissionType:
        obj = ops.request_review(SubmissionMutations._get(info, root, submission_id), user=user_or_none(info.context.user))
        return SubmissionType.from_model(obj)

    @gql.mutation(description='Send a submission in review back to draft.', graphql_type=SubmissionType)
    @staticmethod
    def return_to_draft(info: gql.Info, root: sb.Parent[Me], submission_id: sb.ID) -> SubmissionType:
        obj = ops.return_to_draft(SubmissionMutations._get(info, root, submission_id), user=user_or_none(info.context.user))
        return SubmissionType.from_model(obj)

    @gql.mutation(
        description=(
            'Finalise a submission in review: publish the instance and pin the published revision. '
            'If publication is blocked, the blocking problems are returned and the submission stays in review.'
        ),
        graphql_type=SubmissionType | ConstraintViolationsType | DatasetValidationViolationsType,
    )
    @staticmethod
    def finalise(
        info: gql.Info, root: sb.Parent[Me], submission_id: sb.ID
    ) -> SubmissionType | ConstraintViolationsType | DatasetValidationViolationsType:
        from datasets.validation import InstanceDatasetValidationError
        from nodes.constraints.validation import InstanceConstraintError

        submission = SubmissionMutations._get(info, root, submission_id)
        try:
            obj = ops.finalise(submission, user=user_or_none(info.context.user))
        except InstanceConstraintError as error:
            return ConstraintViolationsType.from_conflicts(error.conflicts)
        except InstanceDatasetValidationError as error:
            return DatasetValidationViolationsType.from_violations(error.violations)
        return SubmissionType.from_model(obj)

    @gql.mutation(description='Delete an open submission.', graphql_type=DiscardSubmissionResult)
    @staticmethod
    def discard(info: gql.Info, root: sb.Parent[Me], submission_id: sb.ID) -> DiscardSubmissionResult:
        ops.discard(SubmissionMutations._get(info, root, submission_id))
        return DiscardSubmissionResult(discarded_submission_id=submission_id)
