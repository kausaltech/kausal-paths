"""Evidence assertions about individual data points: how a value was obtained and how good it is."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from django.db import models
from django.utils.translation import gettext_lazy as _

from kausal_common.models.modification_tracking import UserModifiableModel

if TYPE_CHECKING:
    from kausal_common.datasets.models import DataPoint
    from kausal_common.models.types import FK

    from .quality import DataQualityLevel


class DataEvidenceKind(models.TextChoices):
    OBSERVED = 'observed', _('Observed')
    ESTIMATED = 'estimated', _('Estimated')
    EXPLICIT_ZERO = 'explicit_zero', _('Confirmed zero')
    PROVIDER_DEFAULT = 'provider_default', _('Provider default')


class DataPointEvidence(UserModifiableModel):
    """
    What is asserted about one data point's value.

    Lives on the Paths side because the quality vocabulary is framework-owned;
    `DataPoint` itself is shared with Watch. A missing row means nothing has been
    asserted: the value is ungraded and its kind unknown, which is not the same
    statement as the lowest grade. A row asserts at least one of the two.

    Access follows the data point: whoever may change the point may change its
    evidence.
    """

    data_point: FK[DataPoint] = models.OneToOneField('datasets.DataPoint', on_delete=models.CASCADE, related_name='evidence')
    kind = models.CharField(max_length=30, choices=DataEvidenceKind.choices, null=True, blank=True)
    quality_level: FK[DataQualityLevel | None] = models.ForeignKey(
        'frameworks.DataQualityLevel', on_delete=models.PROTECT, null=True, blank=True, related_name='+'
    )

    data_point_id: int
    quality_level_id: int | None

    objects: ClassVar[models.Manager[DataPointEvidence]]

    class Meta:
        constraints = [
            models.CheckConstraint(
                condition=models.Q(kind__isnull=False) | models.Q(quality_level__isnull=False),
                name='data_point_evidence_asserts_something',
            ),
        ]

    def __str__(self) -> str:
        return f'Evidence for data point {self.data_point_id}: {self.kind or "-"}/{self.quality_level_id or "-"}'
