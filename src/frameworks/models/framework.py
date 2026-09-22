from typing import TYPE_CHECKING, Any, ClassVar, Self

from django.contrib.auth.models import Group
from django.contrib.postgres.fields import ArrayField
from django.db import models, transaction
from django.utils.translation import gettext_lazy as _
from pydantic import BaseModel

from django_pydantic_field import SchemaField

from kausal_common.models.ordered import OrderedModel
from kausal_common.models.types import ModelManager, copy_signature
from kausal_common.models.uuid import UUIDIdentifiedModel

from paths.types import CacheablePathsModel, PathsQuerySet
from paths.utils import IdentifierField

if TYPE_CHECKING:
    from kausal_common.models.types import FK, OneToOne, RevMany, RevManyQS

    from frameworks.object_cache import FrameworkSpecificCache  # noqa: F401
    from frameworks.permissions import (
        FrameworkPermissionPolicy,
    )
    from nodes.models import InstanceConfig

    from .config import FrameworkConfig, FrameworkConfigQuerySet
    from .measures import MeasureTemplateQuerySet, Section, SectionQuerySet
    from .quality import DataQualityScheme


class FrameworkQuerySet(PathsQuerySet['Framework']):
    pass


if TYPE_CHECKING:

    class FrameworkManager(ModelManager['Framework', FrameworkQuerySet]):
        """Model manager for Framework."""

else:
    FrameworkManager = ModelManager.from_queryset(FrameworkQuerySet)


class MinMaxDefaultInt(BaseModel):
    min: int | None = None
    """Minimum accepted value."""

    max: int | None = None
    """Maximum accepted value."""

    default: int | None = None
    """Default value."""

    def validate_value(self, value: int) -> int:
        if self.min is not None and value < self.min:
            raise ValueError(f'Value must be at least {self.min}')
        if self.max is not None and value > self.max:
            raise ValueError(f'Value must be at most {self.max}')
        return value


class FrameworkDefaults(BaseModel):
    target_year: MinMaxDefaultInt = MinMaxDefaultInt(min=2030, default=2030, max=2050)
    baseline_year: MinMaxDefaultInt = MinMaxDefaultInt(min=2018, default=None, max=2025)


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

    name = models.CharField(max_length=200, verbose_name=_('Name'))
    identifier = IdentifierField()
    description = models.TextField(blank=True)
    public_base_fqdn = models.CharField(max_length=100, blank=True, null=True)
    use_instance_subdomains = models.BooleanField(
        default=True,
        verbose_name=_('Use instance subdomains'),
        help_text=_('Whether public instance URLs should use instance identifiers as subdomains instead of UUID paths.'),
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    root_section: OneToOne[Section | None] = models.OneToOneField(
        'frameworks.Section',
        on_delete=models.CASCADE,
        related_name='root_for_framework',
        null=True,
    )
    result_excel_url = models.URLField(max_length=250, null=True, blank=True)
    result_excel_node_ids = ArrayField(base_field=models.CharField(max_length=200), null=True, blank=True)
    accept_invitation_url = models.URLField(
        max_length=500,
        null=True,
        blank=True,
        verbose_name=_('Accept invitation URL'),
        help_text=_('URL template for the invitation acceptance page. Use {code} as a placeholder for the invitation code.'),
    )
    template_instance: FK[InstanceConfig | None] = models.ForeignKey(
        'nodes.InstanceConfig',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='+',
        verbose_name=_('Template instance'),
        help_text=_('Instance to clone when creating new instances under this framework.'),
    )
    root_instance: FK[InstanceConfig | None] = models.ForeignKey(
        'nodes.InstanceConfig',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='+',
        verbose_name=_('Root instance'),
        help_text=_('Instance that serves framework-level content and anchors path-based instance URLs.'),
    )
    root_instance_id: int | None = None
    allow_user_registration = models.BooleanField(
        default=False,
        verbose_name=_('Allow user registration'),
        help_text=_('Whether new users can self-register under this framework.'),
    )
    allow_instance_creation = models.BooleanField(
        default=False,
        verbose_name=_('Allow instance creation'),
        help_text=_('Whether authenticated users can create new model instances under this framework.'),
    )
    enable_user_management = models.BooleanField(
        default=False,
        verbose_name=_('Enable user management'),
        help_text=_('Whether instance admins under this framework can manage users (add, invite, remove).'),
    )

    defaults = SchemaField(schema=FrameworkDefaults, default=FrameworkDefaults)

    admin_group: OneToOne[Group | None] = models.OneToOneField(
        Group,
        on_delete=models.PROTECT,
        editable=False,
        related_name='admin_for_framework',
        null=True,
    )
    viewer_group: OneToOne[Group | None] = models.OneToOneField(
        Group,
        on_delete=models.PROTECT,
        editable=False,
        related_name='viewer_for_framework',
        null=True,
    )

    public_fields: ClassVar = [
        'name',
        'identifier',
        'description',
        'allow_user_registration',
        'allow_instance_creation',
    ]

    objects: ClassVar[FrameworkManager] = FrameworkManager()

    class Meta:
        ordering = ['name']

    id: int
    root_section_id: int | None
    admin_group_id: int | None
    dimensions: RevMany[FrameworkDimension]
    sections: RevManyQS[Section, SectionQuerySet]
    configs: RevManyQS[FrameworkConfig, FrameworkConfigQuerySet]
    quality_schemes: RevMany[DataQualityScheme]

    def __str__(self):
        return self.name

    def __rich_repr__(self):
        yield self.name
        yield 'identifier', self.identifier
        yield 'uuid', self.uuid

    @classmethod
    def permission_policy(cls) -> FrameworkPermissionPolicy:
        from frameworks.permissions import FrameworkPermissionPolicy

        return FrameworkPermissionPolicy()

    def to_dict(self):
        return {
            'identifier': self.identifier,
            'name': self.name,
            'description': self.description,
            'public_base_fqdn': self.public_base_fqdn,
            'use_instance_subdomains': self.use_instance_subdomains,
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
        # from .roles import framework_admin_role
        super().save(*args, **kwargs)
        # framework_admin_role.create_or_update_instance_group(self)

    def create_root_section(self) -> Section:
        from .measures import Section

        if self.root_section:
            return self.root_section
        root_section = Section.add_root(instance=Section(framework=self, name=f'{self.name} Root'))
        self.root_section = root_section
        self.save(update_fields=['root_section'])
        return root_section

    def measure_templates(self) -> MeasureTemplateQuerySet:
        from .measures import MeasureTemplate

        return MeasureTemplate.objects.get_queryset().filter(section__framework=self)


class FrameworkDimension(UUIDIdentifiedModel, OrderedModel):
    """
    Represents a classification dimension within a framework.

    A FrameworkDimension is a Framework-specific model for categorizing or organizing various
    aspects of the framework. It can be used e.g. to have different default values for measures
    depending on the region, sector, etc. of the instance.
    """

    framework = models.ForeignKey(Framework, on_delete=models.CASCADE, related_name='dimensions')
    name = models.CharField(max_length=200)
    identifier = IdentifierField()

    categories: RevMany[FrameworkDimensionCategory]

    class Meta:
        ordering = ['framework', 'order']

    def __str__(self):
        return f'{self.framework.name} - {self.name}'

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

    dimension = models.ForeignKey(FrameworkDimension, on_delete=models.CASCADE, related_name='categories')
    name = models.CharField(max_length=200)

    objects: models.Manager[FrameworkDimensionCategory]

    class Meta:
        ordering = ['dimension', 'order']

    def __str__(self):
        return f'{self.dimension.name} - {self.name}'

    def filter_siblings(self, qs: models.QuerySet[Self]) -> models.QuerySet[Self]:
        return qs.filter(dimension=self.dimension)
