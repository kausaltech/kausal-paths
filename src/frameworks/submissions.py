"""Submission lifecycle: draft → in review → final, with corrections superseding final submissions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from django.core.exceptions import PermissionDenied, ValidationError
from django.db import IntegrityError, transaction
from django.db.models import Q
from django.utils import timezone
from wagtail.actions.publish_revision import PublishPermissionError

from frameworks.models import Submission, SubmissionKind, SubmissionStatus
from frameworks.models.submission import OPEN_STATUSES

if TYPE_CHECKING:
    from django.db.models import QuerySet

    from nodes.models import InstanceConfig
    from users.models import User


class SubmissionError(ValidationError):
    pass


def visible_submissions(ic: InstanceConfig, *, include_open: bool) -> QuerySet[Submission]:
    """Return the submissions of `ic`; open ones are work in progress, the rest are its record."""
    qs = Submission.objects.filter(instance_config=ic).select_related('finalised_by', 'created_by', 'supersedes')
    if not include_open:
        qs = qs.exclude(status__in=OPEN_STATUSES)
    return qs


def create_submission(
    ic: InstanceConfig,
    *,
    period_start: int,
    period_end: int | None = None,
    kind: SubmissionKind = SubmissionKind.INVENTORY,
    user: User | None,
) -> Submission:
    """
    Open a submission for a period.

    If the period already has a final submission, the new one is a correction of
    it and supersedes it once finalised.
    """
    if ic.config_source != 'database':
        raise SubmissionError('Only database-backed instances can be submitted')
    period_end = period_start if period_end is None else period_end
    if kind == SubmissionKind.INVENTORY and period_end != period_start:
        raise SubmissionError('An inventory covers a single year')
    if period_end < period_start:
        raise SubmissionError('The period ends before it starts')
    years = ic.ensure_spec().years
    if years.min_historical is not None and period_start < years.min_historical:
        raise SubmissionError(f'{period_start} is before the first historical year {years.min_historical}')
    if kind == SubmissionKind.INVENTORY and years.max_historical is not None and period_end > years.max_historical:
        raise SubmissionError(f'{period_end} is after the last historical year {years.max_historical}')

    same_period = Q(instance_config=ic, kind=kind, period_start=period_start, period_end=period_end)
    with transaction.atomic():
        if Submission.objects.filter(same_period, status__in=OPEN_STATUSES).exists():
            raise SubmissionError('This period already has an open submission')
        final = Submission.objects.select_for_update().filter(same_period, status=SubmissionStatus.FINAL).first()
        try:
            with transaction.atomic():
                return Submission.objects.create(
                    instance_config=ic,
                    kind=kind,
                    period_start=period_start,
                    period_end=period_end,
                    supersedes=final,
                    created_by=user,
                    last_modified_by=user,
                )
        except IntegrityError as exc:
            raise SubmissionError('This period already has an open submission') from exc


def _transition(submission: Submission, *, allowed_from: tuple[str, ...], to: str, user: User | None) -> Submission:
    with transaction.atomic():
        locked = Submission.objects.select_for_update().get(pk=submission.pk)
        if locked.status not in allowed_from:
            raise SubmissionError(f'A {locked.get_status_display().lower()} submission cannot become {to}')
        locked.status = to
        locked.last_modified_by = user
        locked.save(update_fields=['status', 'last_modified_by', 'last_modified_at'])
    return locked


def request_review(submission: Submission, *, user: User | None) -> Submission:
    return _transition(submission, allowed_from=(SubmissionStatus.DRAFT,), to=SubmissionStatus.IN_REVIEW, user=user)


def return_to_draft(submission: Submission, *, user: User | None) -> Submission:
    return _transition(submission, allowed_from=(SubmissionStatus.IN_REVIEW,), to=SubmissionStatus.DRAFT, user=user)


def discard(submission: Submission) -> None:
    with transaction.atomic():
        locked = Submission.objects.select_for_update().get(pk=submission.pk)
        if locked.status not in OPEN_STATUSES:
            raise SubmissionError('Only an open submission can be discarded')
        locked.delete()


def finalise(submission: Submission, *, user: User | None) -> Submission:
    """
    Freeze a submission in review by publishing the instance and pinning that revision.

    Publication failures (structural conflicts, dataset validation violations)
    propagate unchanged and leave the submission in review. A superseded final
    submission gives way in the same transaction.
    """
    with transaction.atomic():
        locked = Submission.objects.select_for_update().select_related('instance_config').get(pk=submission.pk)
        if locked.status != SubmissionStatus.IN_REVIEW:
            raise SubmissionError('Only a submission in review can be finalised')
        ic = locked.instance_config
        try:
            ic.publish_instance(user=user)
        except PublishPermissionError as exc:
            raise PermissionDenied('Finalising requires permission to publish the instance') from exc
        ic.refresh_from_db(fields=['live_revision', 'template_revision'])
        if ic.live_revision_id is None:
            raise SubmissionError('Publishing the instance produced no revision')

        if locked.supersedes_id is not None:
            Submission.objects.filter(pk=locked.supersedes_id, status=SubmissionStatus.FINAL).update(
                status=SubmissionStatus.SUPERSEDED, last_modified_at=timezone.now()
            )
        locked.status = SubmissionStatus.FINAL
        locked.instance_revision_id = ic.live_revision_id
        locked.template_revision_id = ic.template_revision_id
        locked.finalised_at = timezone.now()
        locked.finalised_by = user
        locked.last_modified_by = user
        locked.save()
    return locked
