from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

from django.contrib import admin
from django.contrib.postgres.fields import ArrayField
from django.db import models
from django.db.models import Case, OuterRef
from django.db.models.expressions import Subquery, When
from django.db.models.functions import Length, Substr
from django.utils.translation import gettext_lazy as _
from django_stubs_ext.db.models import TypedModelMeta

from treebeard.mp_tree import MP_Node, MP_NodeManager, MP_NodeQuerySet

from kausal_common.models.ordered import OrderedModel
from kausal_common.models.permission_policy import ModelReadOnlyPolicy, ParentInheritedPolicy
from kausal_common.models.tree import get_indented_name
from kausal_common.models.types import ModelManager
from kausal_common.models.uuid import UUIDIdentifiedModel

from paths.types import CacheablePathsModel, PathsQuerySet
from paths.utils import IdentifierField, UnitField

if TYPE_CHECKING:
    from rich.repr import RichReprResult

    from kausal_common.models.permission_policy import ModelPermissionPolicy
    from kausal_common.models.types import FK, M2M, QS, RevMany

    from frameworks.object_cache import (
        FrameworkConfigCacheData,  # noqa: F401
        FrameworkSpecificCache,  # noqa: F401
        MeasureTemplateDefaultDataPointCache,  # noqa: F401
        SectionCacheData,  # noqa: F401
    )
    from frameworks.permissions import (
        MeasureTemplatePermissionPolicy,
        SectionPermissionPolicy,
    )
    from nodes.node import Node
    from users.models import User

    from .config import FrameworkConfig, NodeDimensionSelection


from .framework import Framework, FrameworkDimension, FrameworkDimensionCategory


class SectionQuerySet(MP_NodeQuerySet['Section'], PathsQuerySet['Section']):  # type: ignore[override]
    def _parents(self) -> SectionQuerySet:
        model = self.model
        qs = cast('SectionQuerySet', model._default_manager.get_queryset())
        parents = qs.filter(
            path=Substr(OuterRef('path'), 1, Length(OuterRef('path')) - model.steplen),
        )
        return parents

    def annotate_parent_field(self, annotation_name: str, parent_field: str, min_depth: int = 1) -> Self:
        parents = self._parents()
        sq = Case(
            When(depth__gt=min_depth, then=Subquery(parents.values(parent_field)[:1])),
            default=None,
        )
        return self.annotate(**{annotation_name: sq})


class SectionManager(MP_NodeManager['Section'], ModelManager['Section', SectionQuerySet]):
    def get_queryset(self) -> SectionQuerySet:
        return SectionQuerySet(Section).order_by('path')


class Section(CacheablePathsModel['SectionCacheData'], MP_Node[SectionQuerySet], UUIDIdentifiedModel):
    """
    Represents a section within a framework.

    This model defines a hierarchical structure for organizing framework measures.
    Each section can contain subsections and measure templates.
    """

    framework: FK[Framework] = models.ForeignKey(Framework, on_delete=models.CASCADE, related_name='sections')
    identifier = IdentifierField[str | None, str | None](null=True, blank=True)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    # validation_rules?
    available_years = ArrayField(models.IntegerField(), null=True, blank=True)
    min_total = models.FloatField(null=True)
    max_total = models.FloatField(null=True)
    help_text = models.TextField(blank=True, default='')

    measure_templates: RevMany[MeasureTemplate]
    influencing_measure_templates: M2M[MeasureTemplate, Any] = models.ManyToManyField(
        'frameworks.MeasureTemplate',
        related_name='influenced_sections',
    )

    public_fields: ClassVar = [
        'identifier',
        'uuid',
        'path',
        'name',
        'description',
        'available_years',
        'min_total',
        'max_total',
        'help_text',
        'influencing_measure_templates',
    ]

    objects: ClassVar[SectionManager] = SectionManager()
    _default_manager: ClassVar[SectionManager]

    class Meta:
        ordering = ['path']
        constraints = [
            models.UniqueConstraint(name='section_identifier', fields=['framework', 'identifier'], nulls_distinct=True),
        ]

    def __str__(self):
        return self.name

    def __rich_repr__(self):
        yield self.name
        yield 'framework', self.framework.identifier
        yield 'uuid', self.uuid

    @classmethod
    def permission_policy(cls) -> SectionPermissionPolicy:
        from frameworks.permissions import SectionPermissionPolicy

        return SectionPermissionPolicy()

    @admin.display(description=_('Name'), ordering='name')
    def indented_name(self) -> str:
        return get_indented_name(self, html=True)

    def print_tree(self, indent: int = 0):
        """Print the subsections and measures in each section as an indented hierarchical tree."""

        # Print the current section
        print('  ' * indent + f'Section: {self.name}')

        # Print measures in the current section
        for measure_template in self.measure_templates.all():
            print('  ' * (indent + 1) + f'Measure: {measure_template.name}')

        # Recursively print subsections
        for child in self.get_children():
            child.print_tree(indent + 1)

    def to_dict(self):
        parent = self.get_parent()
        return {
            'uuid': str(self.uuid),
            'identifier': self.identifier,
            'name': self.name,
            'description': self.description,
            'available_years': self.available_years,
            'parent': str(parent.uuid) if parent else None,
        }


class MeasurePriority(models.TextChoices):
    HIGH = 'high', _('High')
    MEDIUM = 'medium', _('Medium')
    LOW = 'low', _('Low')


class DefaultValueScaling(models.TextChoices):
    POPULATION = 'population', _('Population')


class MeasureTemplateQuerySet(PathsQuerySet['MeasureTemplate']):
    pass


if TYPE_CHECKING:

    class MeasureTemplateManager(ModelManager['MeasureTemplate', MeasureTemplateQuerySet]):
        """Model manager for MeasureTemplate."""

else:
    MeasureTemplateManager = ModelManager.from_queryset(MeasureTemplateQuerySet)


class MeasureTemplate(CacheablePathsModel['FrameworkSpecificCache'], OrderedModel, UUIDIdentifiedModel):
    """
    Represents a template for measures within a framework.

    This model defines the structure and attributes of a measure template,
    which is used to hold the metadata for the organization-specific
    measure instances.

    Attributes
    ----------
        section (ForeignKey): A reference to the Section this measure template belongs to.

    """

    section: FK[Section] = models.ForeignKey(Section, on_delete=models.CASCADE, related_name='measure_templates')
    name = models.CharField(max_length=200)
    unit = UnitField()
    priority = models.CharField(max_length=10, choices=MeasurePriority.choices, default=MeasurePriority.MEDIUM)
    min_value = models.FloatField(null=True, blank=True)
    max_value = models.FloatField(null=True, blank=True)
    time_series_max = models.FloatField(null=True, blank=True)
    year_bound = models.BooleanField(default=False)
    hidden = models.BooleanField(default=False)
    help_text = models.TextField(blank=True, default='')
    include_in_progress_tracker = models.BooleanField(default=False)
    default_value_scaling = models.CharField(
        max_length=50,
        choices=DefaultValueScaling.choices,
        null=True,
        blank=True,
    )

    default_value_source = models.TextField(blank=True)

    dimensions: models.ManyToManyField[FrameworkDimension, MeasureTemplateDimension] = models.ManyToManyField(
        FrameworkDimension,
        through='MeasureTemplateDimension',
        blank=True,
        related_name='measure_templates',
    )

    default_data_points: RevMany[MeasureTemplateDefaultDataPoint]
    measures: RevMany[Measure]

    public_fields: ClassVar = [
        'uuid',
        'name',
        'unit',
        'priority',
        'min_value',
        'max_value',
        'time_series_max',
        'default_value_source',
        'year_bound',
        'hidden',
        'help_text',
        'include_in_progress_tracker',
        'default_value_scaling',
    ]

    objects: ClassVar[MeasureTemplateManager] = MeasureTemplateManager()

    section_id: int

    class Meta:
        ordering = ['section', 'order']

    @property
    def framework(self) -> Framework:
        return self.section.framework

    @classmethod
    def permission_policy(cls) -> MeasureTemplatePermissionPolicy:
        from frameworks.permissions import MeasureTemplatePermissionPolicy

        return MeasureTemplatePermissionPolicy()

    def __str__(self):
        return f'{self.section.name} - {self.name}'

    def __rich_repr__(self):
        yield self.name
        yield 'unit', self.unit
        yield 'framework', self.framework.identifier
        yield 'section', self.section.name

    def filter_siblings(self, qs: models.QuerySet[Self]) -> models.QuerySet[Self]:
        return qs.filter(section=self.section)

    def to_dict(self, include_section: bool = True):
        out = {
            'uuid': str(self.uuid),
            'name': self.name,
            'unit': self.unit,
            'priority': self.priority,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'time_series_max': self.time_series_max,
            'default_value_source': self.default_value_source,
            'default_value_scaling': self.default_value_scaling,
            'default_data_points': [dict(year=dp.year, value=dp.value) for dp in self.default_data_points.all()],
        }
        if include_section:
            out['section'] = str(self.section.uuid)
        return out


class MeasureTemplateDimension(OrderedModel):
    template = models.ForeignKey(MeasureTemplate, on_delete=models.CASCADE, related_name='dimensions_through')
    dimension = models.ForeignKey(FrameworkDimension, on_delete=models.CASCADE, related_name='measure_templates_through')

    class Meta:
        ordering = ['template', 'order']

    def filter_siblings(self, qs: models.QuerySet[Self]) -> models.QuerySet[Self]:
        return qs.filter(template=self.template)


class MeasureTemplateDefaultDataPointQuerySet(PathsQuerySet['MeasureTemplateDefaultDataPoint']):
    pass


if TYPE_CHECKING:

    class MeasureTemplateDefaultDataPointManager(
        ModelManager['MeasureTemplateDefaultDataPoint', MeasureTemplateDefaultDataPointQuerySet],
    ):
        """Model manager for MeasureTemplateDefaultDataPoint."""

else:
    MeasureTemplateDefaultDataPointManager = ModelManager.from_queryset(MeasureTemplateDefaultDataPointQuerySet)


class MeasureTemplateDefaultDataPoint(CacheablePathsModel['MeasureTemplateDefaultDataPointCache']):
    """
    Represents a default (fallback) value for a measure template.

    This model stores default values for specific years and category combinations
    for a template. These fallback values can be used when actual data
    is not available for a specific instance.
    """

    template: FK[MeasureTemplate] = models.ForeignKey(
        MeasureTemplate,
        on_delete=models.CASCADE,
        related_name='default_data_points',
    )
    categories: M2M[FrameworkDimensionCategory, Any] = models.ManyToManyField(FrameworkDimensionCategory)
    year = models.IntegerField()
    value = models.FloatField()
    probable_lower_bound = models.FloatField(null=True, blank=True)
    probable_upper_bound = models.FloatField(null=True, blank=True)

    public_fields: ClassVar = ['year', 'value', 'probable_lower_bound', 'probable_upper_bound']

    objects: ClassVar[MeasureTemplateDefaultDataPointManager] = MeasureTemplateDefaultDataPointManager()

    template_id: int

    class Meta:
        ordering = ['template', 'year']

    def __str__(self):
        return f'{self.template.name} - {self.year}'

    def __rich_repr__(self):
        yield 'template', self.template.name
        yield 'year', self.year
        yield 'value', self.value
        yield 'unit', self.template.unit

    @classmethod
    def permission_policy(cls) -> ModelPermissionPolicy[Self, QS[Self]]:
        return ModelReadOnlyPolicy(cls)


class MeasureQuerySet(PathsQuerySet['Measure']):
    pass


if TYPE_CHECKING:

    class MeasureManager(ModelManager['Measure', MeasureQuerySet]):
        """Model manager for Measure."""

else:
    MeasureManager = ModelManager.from_queryset(MeasureQuerySet)


class Measure(CacheablePathsModel['FrameworkConfigCacheData'], models.Model):
    """
    Represents the concrete measure for an organization-specific Instance.

    This model links a MeasureTemplate to a FrameworkConfig, allowing for
    organization-specific instances of measures. It can override the unit
    from the template and store internal notes.
    """

    framework_config: FK[FrameworkConfig] = models.ForeignKey(
        'frameworks.FrameworkConfig', on_delete=models.CASCADE, related_name='measures'
    )
    measure_template: FK[MeasureTemplate] = models.ForeignKey(MeasureTemplate, on_delete=models.CASCADE, related_name='measures')
    unit = UnitField[str | None](null=True, blank=True)
    internal_notes = models.TextField(blank=True)

    data_points: RevMany[MeasureDataPoint]
    measure_template_id: int

    public_fields: ClassVar = [
        'framework_config',
        'measure_template',
        'unit',
        'data_points',
        'internal_notes',
    ]

    objects: ClassVar[MeasureManager] = MeasureManager()

    framework_config_id: int

    _node: tuple[Node | None, NodeDimensionSelection | None]

    class Meta:
        ordering = ['framework_config', 'measure_template']
        constraints = [
            models.UniqueConstraint(fields=['framework_config', 'measure_template'], name='unique_instance_measure'),
        ]

    def __str__(self):
        return f'{self.framework_config.framework.name} - {self.measure_template.name}'

    def __rich_repr__(self) -> RichReprResult:
        yield 'framework', self.framework_config.framework.name
        yield 'instance', self.framework_config.instance_config.name
        yield 'template', self.measure_template.name
        yield 'nr_data_points', len(self.data_points.all())

    @classmethod
    def permission_policy(cls) -> ParentInheritedPolicy[Self, FrameworkConfig, MeasureQuerySet]:
        from .config import FrameworkConfig

        return ParentInheritedPolicy(cls, FrameworkConfig, 'framework_config')

    @classmethod
    def user_can_create(cls, user: User, fwc: FrameworkConfig) -> bool:
        return fwc.permission_policy().user_can_create(user, fwc.framework)


class MeasureDataPointQuerySet(PathsQuerySet['MeasureDataPoint']):
    pass


if TYPE_CHECKING:

    class MeasureDataPointManager(ModelManager['MeasureDataPoint', MeasureDataPointQuerySet]):
        """Model manager for MeasureDataPoint."""

else:
    MeasureDataPointManager = ModelManager.from_queryset(MeasureDataPointQuerySet)


class MeasureDataPoint(CacheablePathsModel[None], models.Model):
    """
    Represents a specific data point for a Measure.

    This model stores the actual value for a specific year for a given Measure.
    It provides a way to record and track the data points over time for each
    organization-specific measure instance.
    """

    measure: FK[Measure] = models.ForeignKey(Measure, on_delete=models.CASCADE, related_name='data_points')
    year = models.IntegerField()
    value = models.FloatField(null=True)
    default_value = models.FloatField(null=True)
    probable_lower_bound = models.FloatField(null=True, blank=True)
    probable_upper_bound = models.FloatField(null=True, blank=True)

    public_fields: ClassVar = [
        'id',
        'year',
        'value',
        'default_value',
        'probable_lower_bound',
        'probable_upper_bound',
    ]

    objects: ClassVar[MeasureDataPointManager] = MeasureDataPointManager()
    _default_manager: ClassVar[MeasureDataPointManager]

    measure_id: int

    class Meta(TypedModelMeta):
        ordering = ['measure', 'year']
        constraints = [
            models.UniqueConstraint(fields=['measure', 'year'], name='unique_measure_year_datapoints'),
        ]

    def __str__(self):
        return f'{self.measure.measure_template.name} - {self.year}'

    def __rich_repr__(self):
        yield 'year', self.year
        yield 'value', self.value
        yield 'measure', self.measure

    @classmethod
    def permission_policy(cls) -> ParentInheritedPolicy[Self, Measure, MeasureDataPointQuerySet]:
        return ParentInheritedPolicy(cls, Measure, 'measure')
