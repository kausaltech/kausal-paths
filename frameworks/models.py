from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

from django.conf import settings
from django.contrib.auth.models import Group
from django.contrib.postgres.fields import ArrayField
from django.db import models, transaction
from django.db.models import Case, OuterRef
from django.db.models.expressions import Subquery, When
from django.db.models.functions import Length, Substr
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from django_stubs_ext.db.models import TypedModelMeta

from treebeard.mp_tree import MP_Node, MP_NodeManager, MP_NodeQuerySet

from kausal_common.models.modification_tracking import UserModifiableModel
from kausal_common.models.ordered import OrderedModel
from kausal_common.models.types import FK, M2M, OneToOne, copy_signature
from kausal_common.models.uuid import UUIDIdentifiedModel
from kausal_common.users import user_or_none

from paths.utils import IdentifierField, UnitField

from nodes.instance import Instance, InstanceLoader
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from kausal_common.models.types import RevMany

    from paths.types import UserOrAnon


class Framework(UUIDIdentifiedModel):
    """
    Represents a framework for Paths models.

    A framework is a combination of a common computation model,
    a set of measures (with their default, fallback values),
    the data that is collected per model instance, and classifications
    for the default values.

    This model defines the common metadata for a model, including its name
    and description. It serves as the top-level container for related components
    such as dimensions, sections, and measure templates.

    Attributes
    ----------
        name (CharField): The name of the framework, limited to 200 characters.
        description (TextField): An optional description of the framework.

    """

    name = models.CharField(max_length=200, verbose_name=_("Name"))
    identifier = IdentifierField()
    description = models.TextField(blank=True)
    public_base_fqdn = models.CharField(max_length=100, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    root_section: OneToOne[Section | None] = models.OneToOneField(
        "frameworks.Section", on_delete=models.CASCADE, related_name="root_for_framework", null=True,
    )
    result_excel_url = models.URLField(max_length=250, null=True, blank=True)
    result_excel_node_ids = ArrayField(base_field=models.CharField(max_length=200), null=True, blank=True)
    admin_group: OneToOne[Group | None] = models.OneToOneField(
        Group, on_delete=models.PROTECT, editable=False, related_name='admin_for_framework',
        null=True,
    )

    public_fields: ClassVar = ["name", "identifier", "description"]

    objects: ClassVar[models.Manager[Framework]]

    root_section_id: int | None
    admin_group_id: int | None
    dimensions: RevMany[FrameworkDimension]
    sections: RevMany[Section]
    configs: RevMany[FrameworkConfig]

    def __str__(self):
        return self.name

    def __rich_repr__(self):
        yield self.name
        yield 'identifier', self.identifier
        yield 'uuid', self.uuid

    def to_dict(self):
        return {
            'identifier': self.identifier,
            'name': self.name,
            'description': self.description,
            'public_base_fqdn': self.public_base_fqdn,
            'result_excel_url': self.result_excel_url,
            'result_excel_node_ids': self.result_excel_node_ids,
        }

    def export_sections(self):
        root_section: Section | None = getattr(self, 'root_section', None)
        if not root_section:
            return []
        sections = root_section.get_descendants()
        out: list[dict[str, Any]] = []
        for section in sections:
            sd = section.to_dict()
            if section.get_parent() == root_section:
                # Do not include the root section in the export
                sd['parent'] = None
            sd['measure_templates'] = [mt.to_dict(include_section=False) for mt in section.measure_templates.order_by('order')]
            out.append(sd)
        return out

    @transaction.atomic
    @copy_signature(models.Model.delete)
    def delete(self, **kwargs):
        if self.admin_group_id is not None:
            g_id = self.admin_group_id
            has_others = type(self).objects.filter(admin_group_id=g_id).exclude(pk=self.pk).exists()
            if not has_others:
                self.admin_group = None
                super().save(update_fields=['admin_group'])
                Group.objects.get(id=g_id).delete()
        return super().delete(**kwargs)

    @copy_signature(models.Model.save)
    def save(self, *args, **kwargs):
        #from .roles import framework_admin_role
        super().save(*args, **kwargs)
        #framework_admin_role.create_or_update_instance_group(self)

    def create_root_section(self) -> Section:
        if self.root_section:
            return self.root_section
        root_section = Section.add_root(instance=Section(framework=self, name=f"{self.name} Root"))
        self.root_section = root_section
        self.save(update_fields=['root_section'])
        return root_section


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

    categories: RevMany[FrameworkDimensionCategory]

    class Meta:  # pyright: ignore
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

    Attributes
    ----------
        dimension (ForeignKey): A reference to the FrameworkDimension this category belongs to.

    """

    dimension = models.ForeignKey(FrameworkDimension, on_delete=models.CASCADE, related_name="categories")
    name = models.CharField(max_length=200)

    objects: models.Manager[FrameworkDimensionCategory]  # pyright: ignore

    class Meta:  # pyright: ignore
        ordering = ["dimension", "order"]

    def __str__(self):
        return f"{self.dimension.name} - {self.name}"

    def filter_siblings(self, qs: models.QuerySet[Self]) -> models.QuerySet[Self]:
        return qs.filter(dimension=self.dimension)


class SectionQuerySet(MP_NodeQuerySet['Section']):
    def _parents(self) -> SectionQuerySet:
        model = cast(type[Section], self.model)
        qs = cast(SectionQuerySet, model._default_manager.get_queryset())
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


class SectionManager(MP_NodeManager['Section']):
    def get_queryset(self) -> SectionQuerySet:
        return SectionQuerySet(Section).order_by('path')


class Section(MP_Node[SectionQuerySet], UUIDIdentifiedModel):
    """
    Represents a section within a framework.

    This model defines a hierarchical structure for organizing framework measures.
    Each section can contain subsections and measure templates.
    """

    framework: FK[Framework] = models.ForeignKey(Framework, on_delete=models.CASCADE, related_name="sections")
    identifier = IdentifierField(null=True, blank=True)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    # validation_rules?
    available_years = ArrayField(models.IntegerField(), null=True, blank=True)

    measure_templates: RevMany[MeasureTemplate]

    public_fields: ClassVar = ["identifier", "uuid", "path", "name", "description", "available_years"]

    objects: ClassVar[SectionManager] = SectionManager()  # pyright: ignore
    _default_manager: ClassVar[SectionManager]

    class Meta:  # pyright: ignore
        constraints = [
            models.UniqueConstraint(name='section_identifier', fields=['framework', 'identifier'], nulls_distinct=True),
        ]

    def __str__(self):
        return self.name

    def __rich_repr__(self):
        yield self.name
        yield "framework", self.framework.identifier
        yield "uuid", self.uuid

    def print_tree(self, indent: int = 0):
        """Print the subsections and measures in each section as an indented hierarchical tree."""

        # Print the current section
        print("  " * indent + f"Section: {self.name}")

        # Print measures in the current section
        for measure_template in self.measure_templates.all():
            print("  " * (indent + 1) + f"Measure: {measure_template.name}")

        # Recursively print subsections
        for child in self.get_children():
            child.print_tree(indent + 1)

    def to_dict(self):
        parent = self.get_parent()
        return {
            "uuid": str(self.uuid),
            "identifier": self.identifier,
            "name": self.name,
            "description": self.description,
            "available_years": self.available_years,
            "parent": str(parent.uuid) if parent else None,
        }


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

    Attributes
    ----------
        section (ForeignKey): A reference to the Section this measure template belongs to.

    """

    section: FK[Section] = models.ForeignKey(Section, on_delete=models.CASCADE, related_name="measure_templates")
    name = models.CharField(max_length=200)
    unit = UnitField()
    priority = models.CharField(max_length=10, choices=MeasurePriority.choices, default=MeasurePriority.MEDIUM)
    min_value = models.FloatField(null=True, blank=True)
    max_value = models.FloatField(null=True, blank=True)
    time_series_max = models.FloatField(null=True, blank=True)

    default_value_source = models.TextField(blank=True)

    dimensions: models.ManyToManyField[FrameworkDimension, MeasureTemplateDimension] = models.ManyToManyField(
        FrameworkDimension, through="MeasureTemplateDimension", blank=True, related_name="measure_templates",
    )

    default_data_points: RevMany[MeasureTemplateDefaultDataPoint]
    measures: RevMany[Measure]

    objects: ClassVar[models.Manager[MeasureTemplate]]
    public_fields: ClassVar = [
        "uuid", "name", "unit", "priority", "min_value", "max_value", "time_series_max", "default_value_source",
    ]

    class Meta:  # pyright: ignore
        ordering = ["section", "order"]

    @property
    def framework(self) -> Framework:
        return self.section.framework

    def __str__(self):
        return f"{self.section.name} - {self.name}"

    def __rich_repr__(self):
        yield self.name
        yield "unit", self.unit
        yield "framework", self.framework.identifier
        yield "section", self.section.name

    def filter_siblings(self, qs: models.QuerySet[Self]) -> models.QuerySet[Self]:
        return qs.filter(section=self.section)

    def to_dict(self, include_section: bool = True):
        out = {
            "uuid": str(self.uuid),
            "name": self.name,
            "unit": self.unit,
            "priority": self.priority,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "time_series_max": self.time_series_max,
            "default_value_source": self.default_value_source,
            "default_data_points": [
                dict(year=dp.year, value=dp.value) for dp in self.default_data_points.all()
            ],
        }
        if include_section:
            out['section'] = str(self.section.uuid)
        return out


class MeasureTemplateDimension(OrderedModel):
    template = models.ForeignKey(MeasureTemplate, on_delete=models.CASCADE, related_name="dimensions_through")
    dimension = models.ForeignKey(FrameworkDimension, on_delete=models.CASCADE, related_name="measure_templates_through")

    class Meta:  # pyright: ignore
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

    template: FK[MeasureTemplate] = models.ForeignKey(
        MeasureTemplate, on_delete=models.CASCADE, related_name='default_data_points',
    )
    categories: M2M[FrameworkDimensionCategory, Any] = models.ManyToManyField(FrameworkDimensionCategory)
    year = models.IntegerField()
    value = models.FloatField()

    public_fields: ClassVar = ['year', 'value']

    class Meta:
        ordering = ["template", "year"]

    def __str__(self):
        return f"{self.template.name} - {self.year}"

    def __rich_repr__(self):
        yield "template", self.template.name
        yield "year", self.year
        yield "value", self.value
        yield "unit", self.template.unit


def create_random_token():
    return uuid.uuid4().hex


class FrameworkConfig(UserModifiableModel, UUIDIdentifiedModel):
    """
    Represents a configuration of a Framework for a specific instance.

    This model links a Framework to an InstanceConfig, allowing for customization
    of framework settings for each organization or instance. It includes fields
    for specifying the organization name, baseline year, and associated categories.
    """

    framework: FK[Framework] = models.ForeignKey(Framework, on_delete=models.CASCADE, related_name="configs")
    instance_config: FK[InstanceConfig] = models.ForeignKey(
        InstanceConfig, on_delete=models.CASCADE, related_name='framework_configs',
    )
    organization_name = models.CharField(max_length=200, blank=True)
    baseline_year = models.IntegerField()
    categories: M2M[FrameworkDimensionCategory, Any] = models.ManyToManyField(FrameworkDimensionCategory)
    token = models.CharField(max_length=50, default=create_random_token)

    measures: RevMany[Measure]

    public_fields: ClassVar = ['framework', 'organization_name', 'baseline_year', 'uuid', 'instance_config']

    class Meta:  # pyright: ignore
        constraints = [
            models.UniqueConstraint(fields=['framework', 'instance_config'], name='unique_framework_instance'),
        ]

    @classmethod
    @transaction.atomic
    def create_instance(  # noqa: PLR0913
        cls, framework: Framework, instance_identifier: str, org_name: str, baseline_year: int, uuid: str | None = None,
        user: UserOrAnon | None = None,
    ) -> Self:
        ic = InstanceConfig.objects.create(
            name='%s: %s' % (framework.name, org_name), identifier=instance_identifier,
            primary_language="en", other_languages=[],
        )
        fc = cls.objects.create(
            framework=framework,
            instance_config=ic,
            organization_name=org_name,
            baseline_year=baseline_year,
            uuid=uuid,
            created_by=user_or_none(user),  # type: ignore[misc]
        )
        ic.site_url = fc.get_view_url()
        if ic.site_url is not None:
            ic.sync_nodes()
            ic.create_default_content()
        return fc

    def create_model_instance(self, ic: InstanceConfig) -> Instance:
        fw = self.framework
        config_fn = Path(settings.BASE_DIR, 'configs', '%s.yaml' % fw.identifier)
        loader = InstanceLoader.from_yaml(config_fn, fw_config=self)
        return loader.instance

    def get_view_url(self):
        fw = self.framework
        if not fw.public_base_fqdn:
            return None
        return 'https://%s.%s' % (self.instance_config.identifier, fw.public_base_fqdn)

    def __str__(self):
        return f"{self.framework.identifier}: {self.instance_config.name}"

    def notify_change(self, user: UserOrAnon | None = None, save: bool = False):
        self.last_modified_by = user_or_none(user)
        self.last_modified_at = timezone.now()
        if save:
            self.save(update_fields=['last_modified_by', 'last_modified_at'])
        self.instance_config.invalidate_cache()


class Measure(models.Model):
    """
    Represents the concrete measure for an organization-specific Instance.

    This model links a MeasureTemplate to a FrameworkConfig, allowing for
    organization-specific instances of measures. It can override the unit
    from the template and store internal notes.
    """

    framework_config: FK[FrameworkConfig] = models.ForeignKey(FrameworkConfig, on_delete=models.CASCADE, related_name="measures")
    measure_template: FK[MeasureTemplate] = models.ForeignKey(MeasureTemplate, on_delete=models.CASCADE, related_name="measures")
    unit = UnitField(null=True, blank=True)
    internal_notes = models.TextField(blank=True)

    data_points: RevMany[MeasureDataPoint]
    measure_template_id: int

    public_fields: ClassVar = [
        'framework_config', 'measure_template', 'unit', 'data_points', 'internal_notes',
    ]

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['framework_config', 'measure_template'], name='unique_instance_measure'),
        ]

    def __str__(self):
        return f"{self.framework_config.framework.name} - {self.measure_template.name}"

    def __rich_repr__(self):
        yield "framework", self.framework_config.framework.name
        yield "instance", self.framework_config.organization_name
        yield "template", self.measure_template.name

class MeasureDataPoint(models.Model):
    """
    Represents a specific data point for a Measure.

    This model stores the actual value for a specific year for a given Measure.
    It provides a way to record and track the data points over time for each
    organization-specific measure instance.
    """

    measure: FK[Measure] = models.ForeignKey(Measure, on_delete=models.CASCADE, related_name="data_points")
    year = models.IntegerField()
    value = models.FloatField(null=True)
    default_value = models.FloatField(null=True)

    public_fields: ClassVar = ['id', 'year', 'value']

    class Meta(TypedModelMeta):
        ordering = ["measure", "year"]
        constraints = [
            models.UniqueConstraint(fields=['measure', 'year'], name='unique_measure_year_datapoints'),
        ]

    def __str__(self):
        return f"{self.measure.measure_template.name} - {self.year}"

    def __rich_repr__(self):
        yield "year", self.year
        yield "value", self.value
        yield "measure", self.measure
