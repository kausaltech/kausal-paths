"""Paths-specific dataset caches and advisory reference ranges."""

import math
from uuid import UUID

from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import Q

from django_choices_field import TextChoicesField

from kausal_common.datasets.models import DatasetSchemaDimension, DimensionCategory
from kausal_common.models.uuid import UUIDIdentifiedModel

from nodes.units import Unit, unit_registry


class DVCSourceManifest(models.Model):
    repository_url = models.CharField(max_length=500)
    revision = models.CharField(max_length=40)
    remote_name = models.CharField(max_length=100, blank=True)
    dataset_identifier = models.CharField(max_length=500)
    format_version = models.PositiveSmallIntegerField()
    content = models.JSONField()
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'paths_datasets_dvc_source_manifest__rebuildable'
        constraints = [
            models.UniqueConstraint(
                fields=['repository_url', 'revision', 'remote_name', 'dataset_identifier', 'format_version'],
                name='unique_dataset_source_manifest',
            ),
        ]

    def __str__(self) -> str:
        return f'{self.dataset_identifier}@{self.revision}'


class PreparedDataset(models.Model):
    key = models.CharField(max_length=64, primary_key=True)
    recipe = models.JSONField()
    frame_metadata = models.JSONField()
    payload = models.BinaryField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'paths_datasets_prepared_dataset__rebuildable'

    def __str__(self) -> str:
        return self.key


class PlausibilitySource(UUIDIdentifiedModel):
    """Where a set of reference ranges comes from and how they were derived."""

    identifier = models.SlugField(max_length=100, unique=True)
    name = models.CharField(max_length=200)
    url = models.URLField(max_length=500, blank=True)
    revision = models.CharField(max_length=100)
    """The source data version the ranges were derived from, e.g. a snapshot date."""
    method = models.TextField()
    is_example = models.BooleanField(default=False)
    """Local demonstration data, never to be presented as an empirical benchmark."""

    def __str__(self) -> str:
        return f'{self.identifier}@{self.revision}'

    def clean(self) -> None:
        super().clean()
        if not self.is_example and not self.url:
            raise ValidationError('A published plausibility source requires a URL.')


class PlausibilityAggregation(models.TextChoices):
    CELL = 'cell', 'Each selected cell'
    SUM = 'sum', 'Sum of the selected cells'


class PlausibilityDenominator(models.TextChoices):
    NONE = 'none', 'None'
    POPULATION = 'population', 'Observed population'


class PlausibilityReference(models.TextChoices):
    ABSOLUTE = 'absolute', 'The value itself'
    PREVIOUS_YEAR = 'previous_year', 'Ratio to an earlier year'


class DatasetMetricPlausibilityRange(UUIDIdentifiedModel):
    """
    A versioned, advisory reference band for cells of one dataset metric.

    ``selection`` picks the cells: each listed dimension is restricted to the
    listed categories, and unlisted dimensions are unrestricted. ``aggregation``
    says whether each selected cell is checked on its own, or their sum per year.
    ``reference`` says whether the bounds apply to the value itself or to its
    ratio to an earlier year.

    Bounds carry no unit of their own. An absolute bound is in the metric's
    unit, divided by the denominator's unit when there is one; a previous-year
    bound is a dimensionless ratio. See ``bound_unit``.
    """

    Aggregation = PlausibilityAggregation
    Denominator = PlausibilityDenominator
    Reference = PlausibilityReference

    source = models.ForeignKey(PlausibilitySource, on_delete=models.PROTECT, related_name='ranges')
    metric = models.ForeignKey('datasets.DatasetMetric', on_delete=models.CASCADE, related_name='plausibility_ranges')
    framework = models.ForeignKey(
        'frameworks.Framework', on_delete=models.CASCADE, null=True, blank=True, related_name='plausibility_ranges'
    )
    instance_config = models.ForeignKey(
        'nodes.InstanceConfig', on_delete=models.CASCADE, null=True, blank=True, related_name='plausibility_ranges'
    )
    identifier = models.SlugField(max_length=100)
    selection = models.JSONField(default=dict, blank=True)
    """Stable dimension UUID strings mapped to lists of category UUID strings."""
    aggregation = TextChoicesField(choices_enum=PlausibilityAggregation, default=PlausibilityAggregation.CELL)  # pyright: ignore[reportCallIssue]
    denominator = TextChoicesField(choices_enum=PlausibilityDenominator, default=PlausibilityDenominator.NONE)  # pyright: ignore[reportCallIssue]
    reference = TextChoicesField(choices_enum=PlausibilityReference, default=PlausibilityReference.ABSOLUTE)  # pyright: ignore[reportCallIssue]
    max_gap_years = models.PositiveSmallIntegerField(null=True, blank=True)
    """For a previous-year reference: how many years back the earlier value may be."""
    lower = models.FloatField()
    upper = models.FloatField()
    first_year = models.PositiveSmallIntegerField(null=True, blank=True)
    last_year = models.PositiveSmallIntegerField(null=True, blank=True)
    sample_size = models.PositiveIntegerField(null=True, blank=True)
    revision = models.PositiveIntegerField(default=1)
    enabled = models.BooleanField(default=True)

    metric_id: int
    source_id: int
    framework_id: int | None
    instance_config_id: int | None

    class Meta:
        constraints = [
            models.CheckConstraint(
                condition=(
                    Q(framework__isnull=False, instance_config__isnull=True)
                    | Q(framework__isnull=True, instance_config__isnull=False)
                ),
                name='plausibility_range_has_one_scope',
            ),
            models.CheckConstraint(condition=Q(lower__lt=models.F('upper')), name='plausibility_range_bounds_ordered'),
            models.CheckConstraint(
                condition=(
                    Q(reference='absolute', max_gap_years__isnull=True)
                    | Q(reference='previous_year', max_gap_years__gte=1, denominator='none', lower__gt=0)
                ),
                name='plausibility_range_reference_shape',
            ),
            models.UniqueConstraint(
                fields=['framework', 'metric', 'identifier'],
                condition=Q(framework__isnull=False),
                name='unique_framework_metric_plausibility_range',
            ),
            models.UniqueConstraint(
                fields=['instance_config', 'metric', 'identifier'],
                condition=Q(instance_config__isnull=False),
                name='unique_instance_metric_plausibility_range',
            ),
        ]

    def __str__(self) -> str:
        return f'{self.identifier}: {self.lower}-{self.upper} {self.bound_unit}'

    @property
    def bound_unit(self) -> Unit:
        """The unit the bounds are expressed in, derived from the metric and denominator."""
        if self.reference == self.Reference.PREVIOUS_YEAR:
            return unit_registry.parse_units('dimensionless')
        unit = unit_registry.parse_units(self.metric.unit)
        if self.denominator == self.Denominator.POPULATION:
            unit /= unit_registry.parse_units('cap')
        return unit

    def selected_categories(self) -> dict[str, list[str]]:
        return {dimension: list(categories) for dimension, categories in self.selection.items()}

    def clean(self) -> None:
        super().clean()
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValidationError('Plausibility bounds must be finite.')
        if self.lower >= self.upper:
            raise ValidationError('The lower plausibility bound must be below the upper bound.')
        if self.first_year is not None and self.last_year is not None and self.first_year > self.last_year:
            raise ValidationError('The first year must not be after the last year.')
        self._validate_reference()
        self._validate_selection()
        if self.metric_id is not None:
            try:
                self.bound_unit  # noqa: B018
            except Exception as exc:
                raise ValidationError('The linked metric has no parseable unit.') from exc

    def _validate_reference(self) -> None:
        if self.reference == self.Reference.ABSOLUTE:
            if self.max_gap_years is not None:
                raise ValidationError('Only a previous-year reference has a maximum gap.')
            return
        if self.max_gap_years is None or self.max_gap_years < 1:
            raise ValidationError('A previous-year reference needs a maximum gap of at least one year.')
        if self.denominator != self.Denominator.NONE:
            raise ValidationError('A previous-year ratio is not normalized by a denominator.')
        if self.lower <= 0:
            raise ValidationError('A previous-year ratio needs a positive lower bound.')

    def _validate_selection(self) -> None:
        message = 'Plausibility selection must map dimension UUIDs to non-empty lists of category UUIDs.'
        if not isinstance(self.selection, dict):
            raise ValidationError(message)
        try:
            for dimension, categories in self.selection.items():
                UUID(dimension)
                if not isinstance(categories, list) or not categories or len(set(categories)) != len(categories):
                    raise ValidationError(message)
                for category in categories:
                    UUID(category)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValidationError(message) from exc
        if self.metric_id is None:
            return
        schema_dimensions = {
            str(uuid)
            for uuid in DatasetSchemaDimension.objects.filter(schema_id=self.metric.schema_id).values_list(
                'dimension__uuid', flat=True
            )
        }
        unknown = set(self.selection) - schema_dimensions
        if unknown:
            raise ValidationError(f'Selected dimensions are not in the metric schema: {sorted(unknown)}')
        if self.aggregation == self.Aggregation.SUM and set(self.selection) != schema_dimensions:
            # An unlisted dimension would let a new category, or a total category such
            # as "Gesamt", silently join the sum.
            raise ValidationError('A sum must list its categories for every dimension of the metric schema.')
        known = {
            (str(dimension), str(category))
            for dimension, category in DimensionCategory.objects.filter(dimension__uuid__in=list(self.selection)).values_list(
                'dimension__uuid', 'uuid'
            )
        }
        missing = [
            (dimension, category)
            for dimension, categories in self.selection.items()
            for category in categories
            if (dimension, category) not in known
        ]
        if missing:
            raise ValidationError(f'Selected categories do not belong to their dimension: {missing}')
