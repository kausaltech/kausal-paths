"""The reporting unit: one balance of one instance for one period, as delivered."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from django.db import models
from django.utils.translation import gettext_lazy as _

from kausal_common.models.modification_tracking import UserModifiableModel
from kausal_common.models.uuid import UUIDIdentifiedModel

if TYPE_CHECKING:
    from wagtail.models import Revision

    from kausal_common.models.types import FK

    from nodes.models import InstanceConfig
    from users.models import User


class SubmissionKind(models.TextChoices):
    INVENTORY = 'inventory', _('Inventory')


class SubmissionStatus(models.TextChoices):
    DRAFT = 'draft', _('Draft')
    IN_REVIEW = 'in_review', _('In review')
    FINAL = 'final', _('Final')
    SUPERSEDED = 'superseded', _('Superseded')


OPEN_STATUSES = (SubmissionStatus.DRAFT, SubmissionStatus.IN_REVIEW)


class Submission(UserModifiableModel, UUIDIdentifiedModel):
    """
    A balance of one instance over a period, moving from draft to final.

    Distinct from instance publication and from assessment: the instance draft
    keeps changing, assessments are recomputed continuously, and a submission is
    frozen once, at finalisation, by pinning the published instance revision and
    the template revision it was built on. A final submission is never edited; a
    correction is a new submission that supersedes it, and the old one stays final
    until the correction is itself finalised.

    Per instance, kind and period there is at most one open (draft or in review)
    submission and at most one final one.
    """

    instance_config: FK[InstanceConfig] = models.ForeignKey(
        'nodes.InstanceConfig', on_delete=models.CASCADE, related_name='submissions'
    )
    kind = models.CharField(max_length=30, choices=SubmissionKind.choices, default=SubmissionKind.INVENTORY)
    period_start = models.PositiveIntegerField(help_text=_('First year covered.'))
    period_end = models.PositiveIntegerField(help_text=_('Last year covered; equal to the first for an inventory.'))
    status = models.CharField(max_length=20, choices=SubmissionStatus.choices, default=SubmissionStatus.DRAFT)

    instance_revision: FK[Revision | None] = models.ForeignKey(
        'wagtailcore.Revision', on_delete=models.PROTECT, null=True, blank=True, related_name='+'
    )
    template_revision: FK[Revision | None] = models.ForeignKey(
        'wagtailcore.Revision', on_delete=models.PROTECT, null=True, blank=True, related_name='+'
    )
    finalised_at = models.DateTimeField(null=True, blank=True)
    finalised_by: FK[User | None] = models.ForeignKey(
        'users.User', on_delete=models.SET_NULL, null=True, blank=True, related_name='+'
    )
    supersedes: FK[Submission | None] = models.ForeignKey(
        'self', on_delete=models.PROTECT, null=True, blank=True, related_name='superseded_by'
    )

    instance_config_id: int
    instance_revision_id: int | None
    template_revision_id: int | None
    supersedes_id: int | None

    objects: ClassVar[models.Manager[Submission]]

    class Meta:
        ordering = ['instance_config', '-period_start', 'kind', 'created_at']
        constraints = [
            models.CheckConstraint(condition=models.Q(period_end__gte=models.F('period_start')), name='submission_period_order'),
            models.CheckConstraint(
                condition=~models.Q(kind=SubmissionKind.INVENTORY) | models.Q(period_end=models.F('period_start')),
                name='submission_inventory_single_year',
            ),
            models.CheckConstraint(
                condition=models.Q(status__in=OPEN_STATUSES, instance_revision__isnull=True, finalised_at__isnull=True)
                | models.Q(
                    status__in=(SubmissionStatus.FINAL, SubmissionStatus.SUPERSEDED),
                    instance_revision__isnull=False,
                    finalised_at__isnull=False,
                ),
                name='submission_pinned_iff_finalised',
            ),
            models.UniqueConstraint(
                fields=['instance_config', 'kind', 'period_start', 'period_end'],
                condition=models.Q(status__in=OPEN_STATUSES),
                name='unique_open_submission_per_period',
            ),
            models.UniqueConstraint(
                fields=['instance_config', 'kind', 'period_start', 'period_end'],
                condition=models.Q(status=SubmissionStatus.FINAL),
                name='unique_final_submission_per_period',
            ),
        ]

    def __str__(self) -> str:
        period = str(self.period_start) if self.period_start == self.period_end else f'{self.period_start}-{self.period_end}'
        return f'{self.instance_config_id} {self.kind} {period} ({self.status})'
