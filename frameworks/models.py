from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self
import uuid

from django.contrib.postgres.fields import ArrayField
from django.db.models import QuerySet
from django.db import models
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from treebeard.mp_tree import MP_Node

from kausal_common.models.ordered import OrderedModel
from kausal_common.models.uuid import UUIDIdentifiedModel
from nodes.models import InstanceConfig
from paths.utils import IdentifierField, UnitField

if TYPE_CHECKING:
    from django.db.models.fields.related_descriptors import RelatedManager  # noqa  # pyright: ignore
    from django.db.models.manager import RelatedManager  # type: ignore  # noqa


class Framework(UUIDIdentifiedModel):
    """
    Represents a framework for Paths models

    A framework is a combination of a common computation model,
    a set of measures (with their default, fallback values),
    the data that is collected per model instance, and classifications
    for the default values.

    This model defines the common metadata for a model, including its name
    and description. It serves as the top-level container for related components
    such as dimensions, sections, and measure templates.

    Attributes:
        name (CharField): The name of the framework, limited to 200 characters.
        description (TextField): An optional description of the framework.
    """

    name = models.CharField(max_length=200, verbose_name=_("Name"))
    identifier = IdentifierField()
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    root_section = models.OneToOneField("Section", on_delete=models.CASCADE, related_name="root_for_framework", null=True)

    public_fields: ClassVar = ["name", "identifier", "description"]

    objects: models.Manager[Framework]

    dimensions: RelatedManager[FrameworkDimension]
    sections: RelatedManager[Section]
    configs: RelatedManager[FrameworkConfig]

    def __str__(self):
        return self.name


class FrameworkDimension(UUIDIdentifiedModel, OrderedModel):
    """
    Represents a classification dimension within a framework.

    A FrameworkDimension is a Framework-specific model for categorizing or organizing various
    aspects of the framework. It can be used e.g. to have different default values for measures
    depending on the region, sector, etc. of the instance.
    """

    framework = models.ForeignKey(Framework, on_delete=models.CASCADE, related_name="dimensions")
    name = models.CharField(max_length=200)
    identifier = IdentifierField()

    categories: RelatedManager[FrameworkDimensionCategory]

    class Meta:
        ordering = ["framework", "order"]

    def __str__(self):
        return f"{self.framework.name} - {self.name}"

    def filter_siblings(self, qs: models.QuerySet[Self]) -> models.QuerySet[Self]:
        return qs.filter(framework=self.framework)


class FrameworkDimensionCategory(UUIDIdentifiedModel, OrderedModel):
    """
    Represents a category within a FrameworkDimension.

    This model defines categories that belong to a specific FrameworkDimension.
    Categories are used to further classify or organize aspects within a dimension
    of the framework. For example, a 'Region' dimension might have categories such as
    'Northern Europe', 'Southern Europe', etc.

    Attributes:
        dimension (ForeignKey): A reference to the FrameworkDimension this category belongs to.
    """

    dimension = models.ForeignKey(FrameworkDimension, on_delete=models.CASCADE, related_name="categories")
    name = models.CharField(max_length=200)

    objects: models.Manager[FrameworkDimensionCategory]

    class Meta:
        ordering = ["dimension", "order"]

    def __str__(self):
        return f"{self.dimension.name} - {self.name}"

    def filter_siblings(self, qs: models.QuerySet[Self]) -> models.QuerySet[Self]:
        return qs.filter(dimension=self.dimension)

# Monkeypatching MP_Node to make it work with type hints
MP_Node.__class_getitem__ = classmethod(lambda cls, *args, **kwargs: cls)  # type: ignore

class Section(MP_Node['Section', QuerySet['Section']], UUIDIdentifiedModel):
    """
    Represents a section within a framework.

    This model defines a hierarchical structure for organizing framework measures.
    Each section can contain subsections and measure templates.
    """

    framework = models.ForeignKey(Framework, on_delete=models.CASCADE, related_name="sections")
    identifier = IdentifierField(null=True, blank=True)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    # validation_rules?
    available_years = ArrayField(models.IntegerField(), null=True, blank=True)

    measure_templates: RelatedManager[MeasureTemplate]

    public_fields: ClassVar = ["identifier", "path", "name", "description", "available_years"]

    class Meta:
        constraints = [
            models.UniqueConstraint(name='section_identifier', fields=['framework', 'identifier'], nulls_distinct=True)
        ]

    def __str__(self):
        return self.name

    def print_tree(self, indent: int = 0):
        """
        Prints the subsections and measures in each section as an indented hierarchical tree.
        """
        # Print the current section
        print("  " * indent + f"Section: {self.name}")

        # Print measures in the current section
        for measure_template in self.measure_templates.all():
            print("  " * (indent + 1) + f"Measure: {measure_template.name}")

        # Recursively print subsections
        for child in self.get_children():
            child.print_tree(indent + 1)


class MeasurePriority(models.TextChoices):
    HIGH = "high", _("High")
    MEDIUM = "medium", _("Medium")
    LOW = "low", _("Low")


class MeasureTemplate(OrderedModel, UUIDIdentifiedModel):
    """
    Represents a template for measures within a framework.

    This model defines the structure and attributes of a measure template,
    which is used to hold the metadata for the organization-specific
    measure instances.

    Attributes:
        section (ForeignKey): A reference to the Section this measure template belongs to.
    """

    section = models.ForeignKey(Section, on_delete=models.CASCADE, related_name="measure_templates")
    name = models.CharField(max_length=200)
    unit = UnitField()
    priority = models.CharField(max_length=10, choices=MeasurePriority.choices, default=MeasurePriority.MEDIUM)
    min_value = models.FloatField(null=True, blank=True)
    max_value = models.FloatField(null=True, blank=True)
    time_series_max = models.FloatField(null=True, blank=True)

    default_value_source = models.TextField(blank=True)

    dimensions: models.ManyToManyField[FrameworkDimension, MeasureTemplateDimension] = models.ManyToManyField(
        FrameworkDimension, through="MeasureTemplateDimension", blank=True, related_name="measure_templates"
    )

    default_data_points: RelatedManager[MeasureTemplateDefaultDataPoint]

    objects: models.Manager[MeasureTemplate]
    public_fields: ClassVar = [
        "name", "unit", "priority", "min_value", "max_value", "time_series_max", "default_value_source",
    ]

    class Meta:
        ordering = ["section", "order"]

    @property
    def framework(self):
        return self.section.framework

    def __str__(self):
        return f"{self.section.name} - {self.name}"

    def filter_siblings(self, qs: models.QuerySet[Self]) -> models.QuerySet[Self]:
        return qs.filter(section=self.section)


class MeasureTemplateDimension(OrderedModel):
    template = models.ForeignKey(MeasureTemplate, on_delete=models.CASCADE, related_name="dimensions_through")
    dimension = models.ForeignKey(FrameworkDimension, on_delete=models.CASCADE, related_name="measure_templates_through")

    class Meta:
        ordering = ["template", "order"]

    def filter_siblings(self, qs: models.QuerySet[Self]) -> models.QuerySet[Self]:
        return qs.filter(template=self.template)


class MeasureTemplateDefaultDataPoint(models.Model):
    """
    Represents a default (fallback) value for a measure template.

    This model stores default values for specific years and category combinations
    for a template. These fallback values can be used when actual data
    is not available for a specific instance.
    """

    template = models.ForeignKey(MeasureTemplate, on_delete=models.CASCADE, related_name="default_data_points")
    categories = models.ManyToManyField(FrameworkDimensionCategory)
    year = models.IntegerField()
    value = models.FloatField()

    public_fields: ClassVar = ['year', 'value']

    class Meta:
        ordering = ["template", "year"]

    def clean(self):
        super().clean()
        framework = self.template.framework
        valid_categories = set(
            FrameworkDimensionCategory.objects.filter(dimension__framework=framework).values_list("id", flat=True)
        )
        current_categories = set([cat.pk for cat in self.categories.all()])
        invalid_categories = valid_categories - current_categories
        if invalid_categories:
            raise ValidationError(
                {
                    "categories": _("Invalid categories for this framework: %(categories)s")
                    % {"categories": ", ".join(str(cat) for cat in invalid_categories)}
                }
            )

    def __str__(self):
        return f"{self.template.name} - {self.year}"


class FrameworkConfig(models.Model):
    """
    Represents a configuration of a Framework for a specific instance.

    This model links a Framework to an InstanceConfig, allowing for customization
    of framework settings for each organization or instance. It includes fields
    for specifying the organization name, baseline year, and associated categories.
    """
    framework = models.ForeignKey(Framework, on_delete=models.CASCADE, related_name="configs")
    instance_config = models.ForeignKey(InstanceConfig, on_delete=models.CASCADE, related_name="framework_configs")
    organization_name = models.CharField(max_length=200, blank=True)
    baseline_year = models.IntegerField()
    categories = models.ManyToManyField(FrameworkDimensionCategory)

    public_fields: ClassVar = ['framework', 'organization_name', 'baseline_year']

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['framework', 'instance_config'], name='unique_framework_instance')
        ]

    @classmethod
    def create_instance(cls, framework: Framework, org_name: str, baseline_year: int):
        new_uuid = uuid.uuid4()
        ic = InstanceConfig.objects.create(
            name='%s: %s' % (framework.name, org_name), identifier=str(new_uuid),
            primary_language="en", other_languages=[],
        )
        fc = cls.objects.create(framework=framework, instance_config=ic, organization_name=org_name, baseline_year=baseline_year)
        return fc

    def __str__(self):
        return f"{self.framework.identifier}: {self.instance_config.name}"


class Measure(models.Model):
    """
    Represents the concrete measure for an organization-specific Instance.

    This model links a MeasureTemplate to a FrameworkConfig, allowing for
    organization-specific instances of measures. It can override the unit
    from the template and store internal notes.
    """
    framework_config = models.ForeignKey(FrameworkConfig, on_delete=models.CASCADE, related_name="measures")
    measure_template = models.ForeignKey(MeasureTemplate, on_delete=models.CASCADE, related_name="measures")
    unit = UnitField(null=True, blank=True)
    internal_notes = models.TextField(blank=True)

    public_fields = [
        'framework_config', 'measure_template', 'unit',
    ]

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['framework_config', 'measure_template'], name='unique_instance_measure')
        ]

    def __str__(self):
        return f"{self.framework_config.framework.name} - {self.measure_template.name}"


class MeasureDataPoint(models.Model):
    """
    Represents a specific data point for a Measure.

    This model stores the actual value for a specific year for a given Measure.
    It provides a way to record and track the data points over time for each
    organization-specific measure instance.
    """
    measure = models.ForeignKey(Measure, on_delete=models.CASCADE, related_name="data_points")
    year = models.IntegerField()
    value = models.FloatField()

    class Meta:
        ordering = ["measure", "year"]

    def __str__(self):
        return f"{self.measure.measure_template.name} - {self.year}"
