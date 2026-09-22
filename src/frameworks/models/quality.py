"""Framework-owned, versioned quality vocabularies for evidence and assessment criteria."""

from decimal import Decimal
from typing import TYPE_CHECKING, Self

from django.core.exceptions import ValidationError
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models

from kausal_common.models.permission_policy import ModelReadOnlyPolicy
from kausal_common.models.types import copy_signature
from kausal_common.models.uuid import UUIDIdentifiedModel

from paths.types import PathsModel
from paths.utils import IdentifierField

if TYPE_CHECKING:
    from kausal_common.models.types import FK, RevMany

    from .framework import Framework


class DataQualityScheme(PathsModel, UUIDIdentifiedModel):
    """A named version of a framework's quality scale, independent of certification profiles."""

    framework: FK[Framework] = models.ForeignKey('frameworks.Framework', on_delete=models.PROTECT, related_name='quality_schemes')
    identifier = IdentifierField()
    version = models.CharField(max_length=50)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)

    framework_id: int
    levels: RevMany[DataQualityLevel]

    class Meta:
        ordering = ['framework', 'identifier', 'version']
        constraints = [
            models.UniqueConstraint(
                fields=['framework', 'identifier', 'version'], name='unique_framework_quality_scheme_version'
            ),
        ]

    def __str__(self) -> str:
        return f'{self.name} ({self.version})'

    @classmethod
    def permission_policy(cls) -> ModelReadOnlyPolicy[Self]:
        return ModelReadOnlyPolicy(cls)

    def clean(self) -> None:
        super().clean()
        if self.pk is not None:
            previous = type(self).objects.get(pk=self.pk)
            if (self.framework_id, self.identifier, self.version) != (
                previous.framework_id,
                previous.identifier,
                previous.version,
            ):
                raise ValidationError('Create a new quality scheme version instead of changing its identity.')

    @copy_signature(models.Model.save)
    def save(self, *args, **kwargs):
        self.full_clean()
        return super().save(*args, **kwargs)


class DataQualityLevel(PathsModel, UUIDIdentifiedModel):
    """One categorical grade and its numeric calculation weight. Missing evidence is not a grade."""

    scheme: FK[DataQualityScheme] = models.ForeignKey(DataQualityScheme, on_delete=models.PROTECT, related_name='levels')
    identifier = models.CharField(max_length=50)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    order = models.PositiveIntegerField(default=0)
    score = models.DecimalField(
        max_digits=7,
        decimal_places=6,
        validators=[MinValueValidator(Decimal(0)), MaxValueValidator(Decimal(1))],
    )

    scheme_id: int

    class Meta:
        ordering = ['scheme', 'order', 'identifier']
        constraints = [
            models.UniqueConstraint(fields=['scheme', 'identifier'], name='unique_quality_level_identifier'),
            models.CheckConstraint(condition=models.Q(score__gte=0, score__lte=1), name='quality_level_score_range'),
        ]

    def __str__(self) -> str:
        return f'{self.identifier}: {self.name}'

    @classmethod
    def permission_policy(cls) -> ModelReadOnlyPolicy[Self]:
        return ModelReadOnlyPolicy(cls)

    def clean(self) -> None:
        super().clean()
        if self.pk is not None:
            previous = type(self).objects.get(pk=self.pk)
            if (self.scheme_id, self.identifier, self.score) != (previous.scheme_id, previous.identifier, previous.score):
                raise ValidationError('Create a new quality scheme version instead of reinterpreting an existing grade.')

    @copy_signature(models.Model.save)
    def save(self, *args, **kwargs):
        self.full_clean()
        return super().save(*args, **kwargs)
