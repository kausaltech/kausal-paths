from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

from django.conf import settings
from django.contrib import admin
from django.contrib.auth.models import Group
from django.contrib.postgres.fields import ArrayField
from django.db import models, transaction
from django.db.models import Case, OuterRef, QuerySet
from django.db.models.expressions import F, Subquery, When
from django.db.models.functions import Length, Substr
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from django_stubs_ext.db.models import TypedModelMeta
from pydantic import BaseModel

import sentry_sdk
from django_pydantic_field import SchemaField
from loguru import logger
from treebeard.mp_tree import MP_Node, MP_NodeManager, MP_NodeQuerySet

from kausal_common.models.modification_tracking import UserModifiableModel
from kausal_common.models.ordered import OrderedModel
from kausal_common.models.permission_policy import ModelPermissionPolicy, ModelReadOnlyPolicy, ParentInheritedPolicy
from kausal_common.models.tree import get_indented_name
from kausal_common.models.types import FK, M2M, QS, ModelManager, OneToOne, RevManyQS, copy_signature
from kausal_common.models.uuid import UUIDIdentifiedModel
from kausal_common.users import UserOrAnon, user_or_none

from paths.types import CacheablePathsModel, PathsModel, PathsQuerySet
from paths.utils import IdentifierField, UnitField

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rich.repr import RichReprResult

    from kausal_common.models.types import RevMany

    from frameworks.permissions import MeasureTemplatePermissionPolicy
    from nodes.gpc import DatasetNode
    from nodes.instance import Instance
    from nodes.models import InstanceConfig
    from nodes.node import Node
    from users.models import User

    from .object_cache import (
        FrameworkConfigCacheData,  # noqa: F401
        FrameworkSpecificCache,  # noqa: F401
        MeasureCache,  # noqa: F401
        MeasureDataPointCache,  # noqa: F401
        MeasureTemplateDefaultDataPointCache,  # noqa: F401
        SectionCacheData,  # noqa: F401
    )
    from .permissions import FrameworkConfigPermissionPolicy, FrameworkPermissionPolicy, SectionPermissionPolicy


@dataclass
class NodeDimensionSelection:
    node_id: str
    dimensions: dict[str, str] | None


class FrameworkQuerySet(PathsQuerySet['Framework']):
    pass


_FrameworkManager = models.Manager.from_queryset(FrameworkQuerySet)
class FrameworkManager(ModelManager['Framework', FrameworkQuerySet], _FrameworkManager):  # pyright: ignore
    """Model manager for Framework."""
del _FrameworkManager


class MinMaxDefaultInt(BaseModel):
    min: int | None = None
    """Minimum accepted value."""

    max: int | None = None
    """Maximum accepted value."""

    default: int | None = None
    """Default value."""

    def validate_value(self, value: int) -> int:
        if self.min is not None and value < self.min:
            raise ValueError(f"Value must be at least {self.min}")
        if self.max is not None and value > self.max:
            raise ValueError(f"Value must be at most {self.max}")
        return value


class FrameworkDefaults(BaseModel):
    target_year: MinMaxDefaultInt = MinMaxDefaultInt(min=2030, default=2030, max=2050)
    baseline_year: MinMaxDefaultInt = MinMaxDefaultInt(min=2018, default=None, max=2023)


class Framework(CacheablePathsModel['FrameworkSpecificCache'], UUIDIdentifiedModel):
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

    defaults = SchemaField(schema=FrameworkDefaults, default=FrameworkDefaults)

    admin_group: OneToOne[Group | None] = models.OneToOneField(
        Group, on_delete=models.PROTECT, editable=False, related_name='admin_for_framework',
        null=True,
    )
    viewer_group: OneToOne[Group | None] = models.OneToOneField(
        Group, on_delete=models.PROTECT, editable=False, related_name='viewer_for_framework',
        null=True,
    )

    public_fields: ClassVar = ["name", "identifier", "description"]

    objects: ClassVar[FrameworkManager] = FrameworkManager()  # pyright: ignore

    id: int
    root_section_id: int | None
    admin_group_id: int | None
    dimensions: RevMany[FrameworkDimension]
    sections: RevManyQS[Section, SectionQuerySet]
    configs: RevManyQS[FrameworkConfig, FrameworkConfigQuerySet]

    def __str__(self):
        return self.name

    def __rich_repr__(self):
        yield self.name
        yield 'identifier', self.identifier
        yield 'uuid', self.uuid

    @classmethod
    def permission_policy(cls) -> FrameworkPermissionPolicy:
        from .permissions import FrameworkPermissionPolicy
        return FrameworkPermissionPolicy()

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

    def measure_templates(self) -> MeasureTemplateQuerySet:
        return MeasureTemplate.objects.get_queryset().filter(section__framework=self)

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


class SectionQuerySet(MP_NodeQuerySet['Section'], PathsQuerySet['Section']):  # type: ignore[override]
    def _parents(self) -> SectionQuerySet:
        model = cast('type[Section]', self.model)
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

    framework: FK[Framework] = models.ForeignKey(Framework, on_delete=models.CASCADE, related_name="sections")
    identifier = IdentifierField[str | None, str | None](null=True, blank=True)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    # validation_rules?
    available_years = ArrayField(models.IntegerField(), null=True, blank=True)
    max_total = models.FloatField(null=True)
    help_text = models.TextField(blank=True, default='')

    measure_templates: RevMany[MeasureTemplate]

    public_fields: ClassVar = ["identifier", "uuid", "path", "name", "description", "available_years", "max_total", "help_text"]

    objects: ClassVar[SectionManager] = SectionManager()
    _default_manager: ClassVar[SectionManager]

    class Meta:
        constraints = [
            models.UniqueConstraint(name='section_identifier', fields=['framework', 'identifier'], nulls_distinct=True),
        ]

    def __str__(self):
        return self.name

    def __rich_repr__(self):
        yield self.name
        yield "framework", self.framework.identifier
        yield "uuid", self.uuid

    @classmethod
    def permission_policy(cls) -> SectionPermissionPolicy:
        from .permissions import SectionPermissionPolicy
        return SectionPermissionPolicy()

    @admin.display(description=_("Name"), ordering='name')
    def indented_name(self) -> str:
        return get_indented_name(self, html=True)

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


class MeasureTemplateQuerySet(PathsQuerySet['MeasureTemplate']):
    pass


_MeasureTemplateManager = models.Manager.from_queryset(MeasureTemplateQuerySet)
class MeasureTemplateManager(ModelManager['MeasureTemplate', MeasureTemplateQuerySet], _MeasureTemplateManager):  # pyright: ignore
    """Model manager for MeasureTemplate."""
del _MeasureTemplateManager


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

    section: FK[Section] = models.ForeignKey(Section, on_delete=models.CASCADE, related_name="measure_templates")
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

    default_value_source = models.TextField(blank=True)

    dimensions: models.ManyToManyField[FrameworkDimension, MeasureTemplateDimension] = models.ManyToManyField(
        FrameworkDimension, through="MeasureTemplateDimension", blank=True, related_name="measure_templates",
    )

    default_data_points: RevMany[MeasureTemplateDefaultDataPoint]
    measures: RevMany[Measure]

    public_fields: ClassVar = [
        "uuid", "name", "unit", "priority", "min_value", "max_value", "time_series_max", "default_value_source",
        "year_bound", "hidden", "help_text", "include_in_progress_tracker",
    ]

    objects: ClassVar[MeasureTemplateManager] = MeasureTemplateManager()  # pyright: ignore

    section_id: int

    class Meta:  # pyright: ignore
        ordering = ["section", "order"]

    @property
    def framework(self) -> Framework:
        return self.section.framework

    @classmethod
    def permission_policy(cls) -> MeasureTemplatePermissionPolicy:
        from .permissions import MeasureTemplatePermissionPolicy
        return MeasureTemplatePermissionPolicy()

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


class MeasureTemplateDefaultDataPointQuerySet(PathsQuerySet['MeasureTemplateDefaultDataPoint']):
    pass

_MeasureTemplateDefaultDataPointManager = models.Manager.from_queryset(MeasureTemplateDefaultDataPointQuerySet)
class MeasureTemplateDefaultDataPointManager(  # pyright: ignore
    ModelManager['MeasureTemplateDefaultDataPoint', MeasureTemplateDefaultDataPointQuerySet],
    _MeasureTemplateDefaultDataPointManager,
):
    """Model manager for MeasureTemplateDefaultDataPoint."""
del _MeasureTemplateDefaultDataPointManager


class MeasureTemplateDefaultDataPoint(CacheablePathsModel['MeasureTemplateDefaultDataPointCache'], models.Model):
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

    objects: ClassVar[MeasureTemplateDefaultDataPointManager] = MeasureTemplateDefaultDataPointManager()  # pyright: ignore

    template_id: int

    class Meta:
        ordering = ["template", "year"]

    def __str__(self):
        return f"{self.template.name} - {self.year}"

    def __rich_repr__(self):
        yield "template", self.template.name
        yield "year", self.year
        yield "value", self.value
        yield "unit", self.template.unit

    @classmethod
    def permission_policy(cls) -> ModelPermissionPolicy[Self, QS[Self]]:
        return ModelReadOnlyPolicy(cls)


def create_random_token():
    return uuid.uuid4().hex


def filter_viewable_by[QS: QuerySet[PathsModel]](qs: QS, user: UserOrAnon) -> QS:
    model = qs.model
    pp = model.permission_policy()
    return qs.filter(id__in=pp.instances_user_has_permission_for(user, 'view'))


class FrameworkConfigQuerySet(PathsQuerySet['FrameworkConfig']):
    pass


_FrameworkConfigManager = models.Manager.from_queryset(FrameworkConfigQuerySet)
class FrameworkConfigManager(ModelManager['FrameworkConfig', FrameworkConfigQuerySet], _FrameworkConfigManager):  # pyright: ignore
    """Model manager for FrameworkConfig."""
del _FrameworkConfigManager


class FrameworkConfig(CacheablePathsModel['FrameworkConfigCacheData'], UserModifiableModel, UUIDIdentifiedModel, models.Model):
    """
    Represents a configuration of a Framework for a specific instance.

    This model links a Framework to an InstanceConfig, allowing for customization
    of framework settings for each organization or instance. It includes fields
    for specifying the organization name, baseline year, and associated categories.
    """

    framework: FK[Framework] = models.ForeignKey(Framework, on_delete=models.CASCADE, related_name="configs")
    instance_config: OneToOne[InstanceConfig] = models.OneToOneField(
        'nodes.InstanceConfig', on_delete=models.CASCADE, related_name='framework_config',
    )
    organization_name = models.CharField(max_length=200, blank=True, null=True)
    organization_identifier = models.CharField(max_length=200, blank=True, null=True)
    organization_slug = models.CharField(max_length=200, blank=True, null=True)
    baseline_year = models.IntegerField()
    target_year = models.IntegerField(null=True)
    categories: M2M[FrameworkDimensionCategory, Any] = models.ManyToManyField(FrameworkDimensionCategory)
    token = models.CharField(max_length=50, default=create_random_token)

    objects: ClassVar[FrameworkConfigManager] = FrameworkConfigManager()  # pyright: ignore

    instance_config_id: int
    framework_id: int
    measures: RevMany[Measure]

    public_fields: ClassVar = ['framework', 'organization_name', 'baseline_year', 'target_year', 'uuid', 'instance_config']

    class Meta:  # pyright: ignore
        constraints = [
            models.UniqueConstraint(fields=['framework', 'instance_config'], name='unique_framework_instance'),
        ]

    def __str__(self):
        return f"{self.framework.name}: {self.instance_config.name}"

    def __rich_repr__(self) -> RichReprResult:
        yield "id", self.pk
        yield "framework", self.framework.identifier
        yield "instance", self.instance_config.identifier
        yield "nr_measures", len(self.measures.all())

    @classmethod
    def permission_policy(cls) -> FrameworkConfigPermissionPolicy:
        from .permissions import FrameworkConfigPermissionPolicy
        return FrameworkConfigPermissionPolicy()

    @classmethod
    @transaction.atomic
    def create_instance(
        cls, framework: Framework, instance_identifier: str, org_name: str, baseline_year: int, uuid: str | None = None,
        target_year: int | None = None, user: UserOrAnon | None = None,
    ) -> FrameworkConfig:
        from nodes.models import InstanceConfig
        from orgs.models import Organization

        instance_name = '%s: %s' % (framework.name, org_name)

        # Create new organization for instance
        org = Organization.add_root(
            name=instance_name,
            primary_language="en",
        )

        ic = InstanceConfig.objects.create(
            name=instance_name,
            identifier=instance_identifier,
            primary_language="en",
            other_languages=[],
            organization=org,
        )

        pp = cls.permission_policy()
        if pp.user_is_authenticated(user):
            extra = cls.permission_policy().get_create_defaults(user, framework)
        else:
            extra = {}
        fc = cls.objects.create(
            framework=framework,
            instance_config=ic,
            organization_name=org_name,
            baseline_year=baseline_year,
            target_year=target_year,
            uuid=uuid,
            created_by=user_or_none(user),  # type: ignore[misc]
            **extra,
        )
        if pp.user_is_authenticated(user):
            pp.realm_admin_role.assign_user(ic, user)
        ic.site_url = fc.get_view_url()
        if ic.site_url is not None:
            from pages.models import ActionListPage

            ic.sync_nodes()
            ic.create_default_content()
            site = ic.site
            assert site is not None
            for alp in site.root_page.get_descendants().type(ActionListPage).specific():
                assert isinstance(alp, ActionListPage)
                alp.show_in_footer = False
                alp.show_in_menus = False
                alp.save()

        return fc

    def create_measure_defaults(self, defaults: dict[str, float] | None = None):
        if not defaults:
            defaults = {}
        fw = self.framework
        mt_qs = fw.measure_templates()
        m_qs = self.measures.filter(measure_template__in=mt_qs)
        m_by_uuid: dict[uuid.UUID, Measure] = {
            m.mt_uuid: m for m in m_qs.annotate(mt_uuid=F('measure_template__uuid'))  # type: ignore[attr-defined]
        }
        year = self.baseline_year
        mdp_qs = (
            MeasureDataPoint.objects.get_queryset().filter(year=year, measure__in=m_qs)
            .annotate(mt_uuid=F('measure__measure_template__uuid'))
        )
        mdp_by_uuid: dict[uuid.UUID, MeasureDataPoint] = {
            mdp.mt_uuid: mdp for mdp in mdp_qs  # type: ignore[attr-defined]
        }
        new_measures: list[Measure] = []
        for mt in mt_qs:
            m = m_by_uuid.get(mt.uuid)
            if m is None:
                m = Measure(framework_config=self, measure_template=mt)
                new_measures.append(m)

        if new_measures:
            Measure.objects.bulk_create(new_measures)

        new_mdps: list[MeasureDataPoint] = []
        update_mdps: list[MeasureDataPoint] = []

        for mt in mt_qs:
            mdp = mdp_by_uuid.get(mt.uuid)
            m = m_by_uuid[mt.uuid]
            default_value = defaults.get(str(mt.uuid))
            if mdp is None:
                mdp = MeasureDataPoint(
                    measure=m,
                    year=year,
                    default_value=default_value,
                    value=None,
                )
                new_mdps.append(mdp)
            else:
                mdp.default_value = default_value
                update_mdps.append(mdp)
        if new_mdps:
            MeasureDataPoint.objects.bulk_create(new_mdps)
        if update_mdps:
            MeasureDataPoint.objects.bulk_update(update_mdps, fields=['default_value'])

    def create_model_instance(self, ic: InstanceConfig) -> Instance:
        from nodes.instance_loader import InstanceLoader

        fw = self.framework
        config_fn = Path(settings.BASE_DIR, 'configs', '%s.yaml' % fw.identifier)
        loader = InstanceLoader.from_yaml(config_fn, fw_config=self)
        return loader.instance

    def get_view_url(self):
        fw = self.framework
        if not fw.public_base_fqdn:
            return None
        return 'https://%s.%s' % (self.instance_config.identifier, fw.public_base_fqdn)

    @property
    def data_points(self) -> MeasureDataPointQuerySet:
        qs = MeasureDataPoint.objects.get_queryset()
        return qs.filter(measure__framework_config=self)

    def notify_change(self, user: UserOrAnon | None = None, save: bool = False):
        self.last_modified_by = user_or_none(user)
        self.last_modified_at = timezone.now()
        if save:
            self.save(update_fields=['last_modified_by', 'last_modified_at'])
        self.instance_config.notify_change()

    def _dimension_name_to_dataset_column_label(self, name: str) -> str:
        return name.replace('_', ' ').capitalize()

    def _get_measure_template_uuids(self, node: DatasetNode) -> list[tuple[str, dict[str, str] | None]]:
        df = node.get_filtered_dataset_df(tag=None)
        if df is None:
            return []
        uuids = {x for x in df.get_column('UUID').to_list() if x is not None}
        dimensions = node.output_dimensions.values()
        column_names = ['UUID'] + [
            self._dimension_name_to_dataset_column_label(dim.id) for dim in dimensions
        ]
        if len(uuids) < 2 or len(node.output_dimensions) == 0:
            return [(u, None) for u in uuids]

        combinations = set()

        df = df.select(column_names)
        df = node.convert_names_to_ids(df)
        for row in df.iter_rows():
            if row[0] is None:
                continue
            combinations.add(row)
        dim_combinations = [c[1:] for c in combinations]
        if len(dim_combinations) != len(set(dim_combinations)):
            logger.error(f'For node {node.id} unique MeasureTemplate uuids could not be found.')
            return []
        result: list[tuple[str, dict[str, str] | None]] = []
        for _uuid, *categories in combinations:
            dims = {}
            for i, dimension in enumerate(dimensions):
                dims[dimension.id] = categories[i]
            result.append((_uuid, dims))
        return result

    @cached_property
    def measure_template_uuid_to_node_dimension_selection(self) -> Mapping[str, NodeDimensionSelection]:
        from nodes.gpc import DatasetNode

        measure_template_uuid_to_multiple_node_dimensions_selections: dict[str, list[NodeDimensionSelection]] = dict()
        instance = self.instance_config.get_instance()
        for node_id, node in instance.context.nodes.items():
            # Intentionally test for concrete type, filter out subclasses
            if type(node) is not DatasetNode:
                continue
            # Workaround to filter viz helper nodes
            # FIXME: Implement this better later
            uuid_param = node.get_parameter_value_str('uuid', required=False)
            if uuid_param:
                continue
            measure_template_uuids = self._get_measure_template_uuids(node)
            for _uuid, dimensions in measure_template_uuids:
                measure_template_uuid_to_multiple_node_dimensions_selections.setdefault(_uuid, []).append(
                    NodeDimensionSelection(node_id=node_id, dimensions=dimensions)
                )

        re_historical = re.compile(r'.*_historical$')
        re_observed = re.compile(r'.*_observed$')

        measure_template_uuid_to_single_node_dimension_selection: dict[str, NodeDimensionSelection] = dict()
        for _uuid, values in measure_template_uuid_to_multiple_node_dimensions_selections.items():
            if len(values) == 1:
                measure_template_uuid_to_single_node_dimension_selection[_uuid] = values[0]
                continue
            accepted_values = [v for v in values if re_observed.match(v.node_id)]
            if len(accepted_values) != 1:
                accepted_values = [v for v in values if not re_historical.match(v.node_id)]
            if len(accepted_values) == 1:
                measure_template_uuid_to_single_node_dimension_selection[_uuid] = accepted_values[0]
                continue
            msg = f'Cannot find single Node to match MeasureTemplate {_uuid}: {", ".join([n.node_id for n in values])}'
            logger.warning(msg)
            sentry_sdk.capture_message(msg)
        return measure_template_uuid_to_single_node_dimension_selection

class MeasureQuerySet(PathsQuerySet['Measure']):
    pass

_MeasureManager = models.Manager.from_queryset(MeasureQuerySet)
class MeasureManager(ModelManager['Measure', MeasureQuerySet], _MeasureManager):  # pyright: ignore
    """Model manager for Measure."""
del _MeasureManager


class Measure(CacheablePathsModel['FrameworkConfigCacheData'], models.Model):
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

    objects: ClassVar[MeasureManager] = MeasureManager()  # pyright: ignore

    framework_config_id: int

    _node: tuple[Node | None, NodeDimensionSelection | None]

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['framework_config', 'measure_template'], name='unique_instance_measure'),
        ]

    def __str__(self):
        return f"{self.framework_config.framework.name} - {self.measure_template.name}"

    def __rich_repr__(self) -> RichReprResult:
        yield "framework", self.framework_config.framework.name
        yield "instance", self.framework_config.instance_config.name
        yield "template", self.measure_template.name
        yield "nr_data_points", len(self.data_points.all())

    @classmethod
    def permission_policy(cls) -> ParentInheritedPolicy[Self, FrameworkConfig, MeasureQuerySet]:
        return ParentInheritedPolicy(cls, FrameworkConfig, 'framework_config')

    @classmethod
    def user_can_create(cls, user: User, fwc: FrameworkConfig) -> bool:
        return fwc.permission_policy().user_can_create(user, fwc.framework)


class MeasureDataPointQuerySet(PathsQuerySet['MeasureDataPoint']):
    pass


_MeasureDataPointManager = models.Manager.from_queryset(MeasureDataPointQuerySet)
class MeasureDataPointManager(ModelManager['MeasureDataPoint', MeasureDataPointQuerySet], _MeasureDataPointManager):  # pyright: ignore
    """Model manager for MeasureDataPoint."""
del _MeasureDataPointManager


class MeasureDataPoint(CacheablePathsModel[None], models.Model):
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

    public_fields: ClassVar = ['id', 'year', 'value', 'default_value']

    objects: ClassVar[MeasureDataPointManager] = MeasureDataPointManager()  # pyright: ignore
    _default_manager: ClassVar[MeasureDataPointManager]

    measure_id: int

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

    @classmethod
    def permission_policy(cls) -> ParentInheritedPolicy[Self, Measure, MeasureDataPointQuerySet]:
        return ParentInheritedPolicy(cls, Measure, 'measure')
