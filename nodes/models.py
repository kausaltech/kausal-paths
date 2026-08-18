from __future__ import annotations

import re
import secrets
import threading
import uuid
from collections.abc import Mapping
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, TypedDict, cast
from urllib.parse import urlparse
from uuid import UUID

from django.conf import settings
from django.contrib.auth.models import Group
from django.contrib.contenttypes.fields import GenericRelation
from django.contrib.contenttypes.models import ContentType
from django.contrib.postgres.expressions import ArraySubquery
from django.contrib.postgres.fields import ArrayField
from django.core.exceptions import ObjectDoesNotExist
from django.db import models, transaction
from django.db.models import F, OuterRef, Q
from django.db.models.expressions import DatabaseDefault
from django.db.models.functions import JSONObject
from django.db.models.manager import Manager
from django.http import HttpRequest
from django.utils import timezone
from django.utils.translation import get_language, gettext, gettext_lazy as _, override
from modelcluster.models import ClusterableModel
from modeltrans.fields import TranslationField
from modeltrans.manager import MultilingualQuerySet
from wagtail import blocks
from wagtail.fields import RichTextField, StreamField
from wagtail.models import DraftStateMixin, Locale, Page, RevisionMixin
from wagtail.search import index

import sentry_sdk
from asgiref.sync import async_to_sync, sync_to_async
from channels.layers import get_channel_layer
from django_choices_field import TextChoicesField
from django_pydantic_field import SchemaField
from loguru import logger
from wagtail_color_panel.fields import ColorField

from kausal_common.datasets.models import (
    Dataset as DatasetModel,
    DatasetMetric,
    DatasetSchema,
    DatasetSchemaScope,
    Dimension as DatasetDimensionModel,
    DimensionCategory,
    DimensionScope,
)
from kausal_common.deployment.http import get_request_wildcard_domains
from kausal_common.i18n.helpers import convert_language_code
from kausal_common.i18n.pydantic import get_modeltrans_attrs_from_str
from kausal_common.models.modification_tracking import UserModifiableModel
from kausal_common.models.permission_policy import (
    ModelPermissionPolicy,
    ParentInheritedPolicy,
    PermissionBlock,
)
from kausal_common.models.permissions import PermissionedManager, PermissionedModel, PermissionedQuerySet
from kausal_common.models.types import (
    MLModelManager,
    copy_signature,
)
from kausal_common.models.uuid import UUIDIdentifiedModel, query_pk_or_uuid_or_identifier

from paths.const import INSTANCE_CHANGE_GROUP, INSTANCE_CHANGE_TYPE
from paths.context import InstanceSpecificCache
from paths.types import CacheablePathsModel, PathsModel, PathsQuerySet
from paths.utils import (
    ChoiceArrayField,
    IdentifierField,
    InstanceIdentifierValidator,
    get_default_language,
    get_supported_languages,
)

from nodes.defs import DatasetBindingDef, DatasetPortSpec, EdgeBindingDef, InstanceModelSpec, NodeSpec, YearsSpec
from nodes.defs.instance_defs import ActionGroup, InstanceFeatures, InstanceMetadata
from nodes.defs.transform_def import EdgeTransformOp, StoredPortTransformOp
from nodes.instance_serialization import (
    DatasetPortSnapshot,
    EdgeSnapshot,
    InputBindingSnapshot,
    NodeSnapshot,
)
from orgs.models import Organization
from pages.blocks import CardListBlock

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

    from django.db.models import CharField

    from loguru import Logger

    from kausal_common.i18n.pydantic import I18nString, TranslatedString
    from kausal_common.models.permission_policy import (
        BaseObjectAction,
        ObjectSpecificAction,
    )
    from kausal_common.models.types import (
        FK,
        M2M,
        OneToOne,
        RevMany,
        RevManyQS,
        RevOne,
    )
    from kausal_common.users import UserOrAnon

    from frameworks.models import FrameworkConfig
    from nodes.dimensions import Dimension as NodeDimension
    from nodes.instance_serialization import InstanceSnapshot, ModelSnapshot
    from nodes.node import Node
    from pages.config import OutcomePage as OutcomePageConfig
    from pages.models import ActionListPage, InstanceSiteContent
    from users.models import User

    from .instance import Instance


instance_cache_lock = threading.Lock()


def get_instance_identifier_from_wildcard_domain(
    hostname: str,
    request: HttpRequest | None = None,
    wildcard_domains: list[str] | None = None,
) -> tuple[str, str] | tuple[None, None]:
    # Get instance identifier from hostname for development and testing
    parts = hostname.lower().split('.', maxsplit=1)
    settings_wildcards = get_request_wildcard_domains(request=None, include_django_settings=True)
    if wildcard_domains is None:
        if request is not None:
            req_wildcards = (
                get_request_wildcard_domains(request, include_django_settings=False) if isinstance(request, HttpRequest) else []
            )
        else:
            req_wildcards = set()
    else:
        req_wildcards = set(wildcard_domains)
    wd_domains = list(settings_wildcards.union(req_wildcards))
    if len(parts) == 2 and parts[1].lower() in wd_domains:
        return (parts[0], parts[1])
    return (None, None)


class InstanceConfigQuerySet(MultilingualQuerySet['InstanceConfig'], PermissionedQuerySet['InstanceConfig']):  # type: ignore[override, misc]
    def for_hostname(self, hostname: str, request: HttpRequest | None = None, wildcard_domains: list[str] | None = None):
        hostname = hostname.lower()
        hostnames = InstanceHostname.objects.filter(hostname=hostname)
        lookup = models.Q(id__in=hostnames.values_list('instance'))

        # Get instance identifier from hostname for development and testing
        identifier, _ = get_instance_identifier_from_wildcard_domain(hostname, request=request, wildcard_domains=wildcard_domains)
        if identifier:
            lookup |= models.Q(identifier=identifier)
        return self.filter(lookup)

    def adminable_for(self, user: User):
        return InstanceConfig.permission_policy().adminable_instances(user)

    def by_all_identifiers(self, id_or_identifier: str) -> InstanceConfigQuerySet:
        if id_or_identifier.isdigit():
            return self.filter(id=id_or_identifier)
        return self.filter(query_pk_or_uuid_or_identifier(id_or_identifier))

    def active(self) -> InstanceConfigQuerySet:
        return self.filter(is_active=True)


_InstanceConfigManager = cast('Manager[InstanceConfig]', Manager).from_queryset(InstanceConfigQuerySet)


class InstanceConfigManager(MLModelManager['InstanceConfig', InstanceConfigQuerySet], _InstanceConfigManager):  # type: ignore[valid-type,misc]
    def get_queryset(self) -> InstanceConfigQuerySet:
        return super().get_queryset().active()

    def get_queryset_all(self):
        return super().get_queryset()

    def get_by_natural_key(self, identifier: str) -> InstanceConfig:
        return self.get(identifier=identifier)


del _InstanceConfigManager


class InstanceConfigPermissionPolicy(ModelPermissionPolicy['InstanceConfig', None, InstanceConfigQuerySet]):
    def __init__(self):
        from frameworks.roles import framework_admin_role, framework_viewer_role

        from .roles import (
            instance_admin_role,
            instance_reviewer_role,
            instance_super_admin_role,
            instance_viewer_role,
        )

        self.super_admin_role = instance_super_admin_role
        self.admin_role = instance_admin_role
        self.viewer_role = instance_viewer_role
        self.reviewer_role = instance_reviewer_role
        self.fw_admin_role = framework_admin_role
        self.fw_viewer_role = framework_viewer_role
        super().__init__(InstanceConfig)

    def is_admin(self, user: User, obj: InstanceConfig) -> bool:
        return user.has_instance_role(self.admin_role, obj) or user.has_instance_role(self.super_admin_role, obj)

    def is_viewer(self, user: User, obj: InstanceConfig) -> bool:
        return user.has_instance_role(self.viewer_role, obj)

    def is_reviewer(self, user: User, obj: InstanceConfig) -> bool:
        return user.has_instance_role(self.reviewer_role, obj)

    def is_framework_admin(self, user: User, obj: InstanceConfig) -> bool:
        if not obj.has_framework_config():
            return False
        return user.has_instance_role(self.fw_admin_role, obj.framework_config.framework)

    def is_framework_viewer(self, user: User, obj: InstanceConfig) -> bool:
        if not obj.has_framework_config():
            return False
        return user.has_instance_role(self.fw_viewer_role, obj.framework_config.framework)

    def user_can_preview_draft(self, user: User, obj: InstanceConfig) -> bool:
        if user.is_superuser:
            return True
        return self.is_admin(user, obj) or self.is_framework_admin(user, obj)

    def user_can_set_lock(self, user: User, obj: InstanceConfig) -> bool:
        if user.is_superuser:
            return True
        return user.has_instance_role(self.super_admin_role, obj) or self.is_framework_admin(user, obj)

    def construct_perm_q(self, user: User, action: ObjectSpecificAction, include_implicit_public: bool = True) -> models.Q | None:
        is_super_admin = self.super_admin_role.role_q(user)
        is_admin = self.admin_role.role_q(user)
        is_viewer = self.viewer_role.role_q(user)
        is_reviewer = self.reviewer_role.role_q(user)
        is_fw_admin = self.fw_admin_role.role_q(user, prefix='framework_config__framework')
        is_fw_viewer = self.fw_viewer_role.role_q(user, prefix='framework_config__framework')

        q = is_super_admin | is_admin | is_fw_admin
        if action in ('change', 'delete'):
            q &= Q(is_locked=False)
        if action == 'view':
            q = is_viewer | is_reviewer | is_super_admin | is_admin | is_fw_admin | is_fw_viewer
            if include_implicit_public:
                q |= Q(framework_config__isnull=True)
        else:
            return q

        # PersonGroupPermissions and PersonPermissions can assign permissions directly to datasetschemas and their associated
        # datasets. We want to count the instanceconfigs those schemas are scoped for as accessible for those users.
        schema_q = DatasetSchema.accessible_by_user_q(user)
        if schema_q is None or bool(schema_q) is False:
            # schema_q can be false for superusers, but we do not want
            # to handle that case here since superusers are already
            # taken into account elsewhere.
            #
            # schema_q is None if there are no dataset schemas the user can access
            return q

        schemas = DatasetSchema.objects.filter(schema_q).values_list('pk', flat=True)
        ic_content_type_id = ContentType.objects.get_for_model(InstanceConfig).pk
        instance_configs_accessible_through_datasets = (
            DatasetSchemaScope.objects.qs
            .filter(scope_content_type_id=ic_content_type_id)
            .filter(schema_id__in=schemas)
            .values_list('scope_id', flat=True)
        )
        return q | Q(pk__in=instance_configs_accessible_through_datasets)

    def construct_perm_q_anon(self, action: BaseObjectAction) -> Q | None:
        if action == 'view':
            # If it's a framework-based config, require authentication for viewing
            return Q(framework_config__isnull=True)
        return None

    def construct_state_perm_q(self, action: ObjectSpecificAction) -> Q:
        if action in ('change', 'delete'):
            return Q(is_locked=False)
        return Q()

    def adminable_instances(self, user: User) -> InstanceConfigQuerySet:
        """
        Return instances that the user has been explicitly granted access to.

        We can't simply use the 'view' permission on InstanceConfig, because
        most instances will be public by default.
        """
        qs = self.get_queryset()
        if user.is_superuser:
            return qs
        return qs.filter(self.construct_perm_q(user, 'view', include_implicit_public=False))

    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: InstanceConfig) -> bool:
        if self.get_permission_block(action, obj=obj) is not None:
            return False
        if user.is_superuser:
            return True
        if action == 'delete':
            return self.is_framework_admin(user, obj)
        if action == 'view':
            if self.anon_has_perm('view', obj):
                return True
            if self.is_viewer(user, obj):
                return True
            if self.is_reviewer(user, obj):
                return True
            if self.is_framework_viewer(user, obj):
                return True
        return self.is_admin(user, obj) or self.is_framework_admin(user, obj)

    def get_permission_block(
        self,
        action: BaseObjectAction,
        *,
        obj: InstanceConfig | None = None,
        context: None = None,
    ) -> PermissionBlock | None:
        if obj is not None and action in ('change', 'delete') and obj.is_locked:
            return PermissionBlock('Instance is locked', code='instance_locked')
        return None

    def anon_has_perm(self, action: ObjectSpecificAction, obj: InstanceConfig) -> bool:
        if action != 'view':
            return False
        if not obj.has_framework_config():
            # FIXME: Add checking for a "published" status here
            return True
        return False

    def user_can_create(self, user: User, context: None) -> bool:
        return False


class NodeCache(TypedDict):
    pass


class DatasetCache(TypedDict):
    dvc_hash: str
    dvc_metadata: dict[str, Any]


class InstanceModelCache(TypedDict):
    nodes: dict[str, NodeCache]
    datasets: dict[str, DatasetCache]


_pytest_instances: dict[str, Instance] = {}
"""Used only in unittests to work around having to parse YAML configs."""

instance_context: ContextVar[Instance | None] = ContextVar('instance_context', default=None)
"""Global instance context for e.g. GraphQL queries."""


class PreferredInstanceSource(StrEnum):
    """
    Which slice of a DB-sourced ``InstanceConfig`` to hydrate.

    ``DRAFT`` reads the current editor tables (``NodeConfig`` / ``NodeEdge``
    / ``DatasetPort`` + ``InstanceConfig.spec``). ``PUBLISHED`` reads the
    latest live ``Revision``'s ``InstanceSnapshot`` payload, falling back
    to ``DRAFT`` if no revision has been published yet.

    The enum values are the exact strings accepted by
    ``_create_from_config(source=...)`` so callers can pass either a
    member or the underlying literal without conversion.
    """

    DRAFT = 'draft'
    PUBLISHED = 'published'


def make_empty_instance_spec() -> InstanceModelSpec:
    return InstanceModelSpec()


YAML_SPEC_VERSION = 3
"""Version of the lightweight YAML-to-InstanceModelSpec materialization."""


def make_minimal_instance_spec(instance: Instance | Mapping[str, Any]) -> InstanceModelSpec:
    """
    Build the computation-only ``InstanceModelSpec``.

    Identity metadata (identifier, name, owner, languages) lives on the
    ``InstanceConfig`` columns and is *not* part of the spec — callers must
    set those columns separately.
    """
    if isinstance(instance, Mapping):
        features = InstanceFeatures.model_validate(instance.get('features') or {})
        return InstanceModelSpec(
            years=YearsSpec(
                reference=instance.get('reference_year'),
                min_historical=instance.get('minimum_historical_year'),
                max_historical=instance.get('maximum_historical_year'),
                target=instance.get('target_year'),
                model_end=instance.get('model_end_year'),
            ),
            features=features,
            action_groups=[ActionGroup.model_validate(dict(group)) for group in instance.get('action_groups', [])],
            theme_identifier=instance.get('theme_identifier'),
        )

    context = instance.context
    features = instance.features
    if not isinstance(features, InstanceFeatures):
        features = InstanceFeatures.model_validate(features or {})
    return InstanceModelSpec(
        years=YearsSpec(
            reference=instance.reference_year,
            min_historical=instance.minimum_historical_year,
            max_historical=instance.maximum_historical_year,
            target=context.target_year,
            model_end=context.model_end_year,
        ),
        features=features,
        action_groups=[group.model_copy() for group in instance.action_groups],
        theme_identifier=instance.theme_identifier,
    )


@dataclass
class InstanceGraphQLContext:
    requested_hostname: str
    matched_hostname: InstanceHostname | None = None


class InstanceConfig(
    DraftStateMixin, RevisionMixin, CacheablePathsModel[InstanceSpecificCache], UUIDIdentifiedModel, UserModifiableModel
):
    """Metadata for one Paths computational model instance."""

    identifier = IdentifierField(max_length=100, unique=True, validators=[InstanceIdentifierValidator()])
    is_active = models.BooleanField(default=True, help_text=_('Whether this instance is active or soft-deleted.'))
    name = models.CharField[str, str](max_length=150, verbose_name=_('name'), unique=True)
    owner = models.CharField[str, str](
        blank=True,
        default='',
        max_length=200,
        verbose_name=_('Owner name'),
        help_text=_('Display name of the organization that owns this instance.'),
    )
    owner_i18n: str | None
    lead_title = models.CharField[str, str](blank=True, max_length=100, verbose_name=_('Lead title'))
    lead_title_i18n: str
    lead_paragraph = RichTextField[str | None, str | None](null=True, blank=True, verbose_name=_('Lead paragraph'))
    lead_paragraph_i18n: str | None
    root_page = models.OneToOneField(
        Page,
        null=True,
        on_delete=models.PROTECT,
        editable=False,
        related_name='instance_config_root',
    )
    organization: FK[Organization] = models.ForeignKey(
        Organization,
        related_name='instances',
        on_delete=models.PROTECT,
        verbose_name=_('organization'),
        help_text=_('The main organization for the instance'),
    )

    copy_of: FK[InstanceConfig | None] = models.ForeignKey(
        'self',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='copies',
        editable=False,
        help_text=_('The instance this one was copied from, if any.'),
    )
    copy_of_id: int | None

    is_protected = models.BooleanField(default=False)
    protection_password = models.CharField(max_length=50, null=True, blank=True)
    is_locked = models.BooleanField(
        default=False,
        help_text=_('Whether end-user mutation surfaces should treat this instance as read-only.'),
    )
    is_hidden = models.BooleanField(
        default=False,
        help_text=_('Hide this instance from the admin instance chooser. It remains reachable directly and via permissions.'),
    )

    created_at = models.DateTimeField(default=timezone.now)
    modified_at = models.DateTimeField(auto_now=True)
    cache_invalidated_at = models.DateTimeField(default=timezone.now)
    yaml_mtime_hash = models.CharField(max_length=32, null=True, blank=True, editable=False)
    yaml_spec_version = models.PositiveSmallIntegerField(default=0, editable=False)

    primary_language = models.CharField[str, str](
        max_length=8,
        choices=get_supported_languages,
        default=get_default_language,
    )
    other_languages = ChoiceArrayField(
        models.CharField(
            max_length=8,
            choices=get_supported_languages,
            default=get_default_language,
        ),
        default=list,
        blank=True,
    )

    config_source = models.CharField(
        max_length=20,
        choices=[('yaml', 'YAML'), ('database', 'Database')],
        default='yaml',
    )
    spec = SchemaField(schema=InstanceModelSpec, null=True, blank=True)

    viewer_group: FK[Group | None] = models.ForeignKey(
        Group,
        on_delete=models.PROTECT,
        editable=False,
        related_name='viewer_instances',
        null=True,
    )
    viewer_group_id: int | None
    reviewer_group: FK[Group | None] = models.ForeignKey(
        Group, on_delete=models.PROTECT, editable=False, related_name='reviewer_instances', null=True
    )
    reviewer_group_id: int | None
    admin_group: FK[Group | None] = models.ForeignKey(
        Group,
        on_delete=models.PROTECT,
        editable=False,
        related_name='admin_instances',
        null=True,
    )
    admin_group_id: int | None
    super_admin_group: FK[Group | None] = models.ForeignKey(
        Group,
        on_delete=models.PROTECT,
        editable=False,
        related_name='super_admin_instances',
        null=True,
    )
    super_admin_group_id: int | None

    owned_by: FK[User | None] = models.ForeignKey(
        'users.User',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='owned_instances',
        verbose_name=_('Owner'),
        help_text=_('The user who owns this instance and has full control over it (e.g. user management).'),
    )
    owned_by_id: int | None

    """
    model_cache = JSONField[InstanceModelCache | None, InstanceModelCache | None](
        verbose_name='cached model data', null=True, blank=True,
    )
    """
    """Used to store data to speed up model runs"""

    i18n = TranslationField(fields=('name', 'owner', 'lead_title', 'lead_paragraph'))

    objects: ClassVar[InstanceConfigManager] = InstanceConfigManager()

    dataset_schema_scopes = GenericRelation(
        'datasets.DatasetSchemaScope',
        related_query_name='instance_config',
        content_type_field='scope_content_type',
        object_id_field='scope_id',
    )

    # Type annotations for reverse FK managers
    nodes: RevManyQS[NodeConfig, NodeConfigQuerySet]
    hostnames: RevMany[InstanceHostname]
    dimensions: RevMany[DatasetDimensionModel]
    datasets: RevMany[DatasetModel]
    edges: RevMany[NodeEdge]
    dataset_ports: RevMany[DatasetPort]
    input_bindings: RevMany[NodeInputPortBinding]
    dataset_revision_pins: RevMany[InstanceRevisionDatasetPin]
    change_operations: RevMany[InstanceChangeOperation]
    framework_config: RevOne[InstanceConfig, FrameworkConfig]
    framework_config_id: int | None
    live_revision_id: int | None  # from DraftStateMixin; id-side of ``live_revision`` FK
    organization_id: int
    site_content: RevOne[InstanceConfig, InstanceSiteContent]

    # Backing storage for the ``nodes_for_serialization`` property. ``None``
    # means "not yet computed or explicitly invalidated, recompute on next
    # read". ``_create_from_config`` clears it so hydrate calls that follow
    # a post-publish edit see the current DB state.
    _nodes_for_serialization: list[NodeConfig] | None
    _annotated_dataset_ports: list[NodeInputPortBinding]
    _publication_dataset_revision_pins: dict[int, Any] | None = None
    # Avoid revalidating the same YAML materialization for every field resolved
    # from one model object. A newly loaded row validates against disk again.
    _verified_yaml_spec_hash: str | None = None
    graphql_context: InstanceGraphQLContext | None = None

    search_fields = [
        index.SearchField('identifier'),
        index.SearchField('name_i18n'),
    ]

    class Meta:
        verbose_name = _('Instance')
        verbose_name_plural = _('Instances')
        ordering = ['id']
        constraints = (
            models.CheckConstraint(
                condition=Q(config_source='yaml') | Q(spec__isnull=False),
                name='instance_database_source_has_spec',
            ),
        )

    def __str__(self) -> str:
        return self.get_name()

    def __rich_repr__(self):
        yield self.identifier
        yield 'id', self.pk
        yield 'name', self.name

    def save(self, *args, update_fields: Iterable[str] | None = None, **kwargs):
        if update_fields is None:
            if not self.uuid or isinstance(self.uuid, DatabaseDefault):
                self.uuid = uuid.uuid4()
            if self.spec is None and (self.config_source == 'database' or self.get_yaml_config_entrypoint()):
                self.spec = self.ensure_spec(update_self=True, save=False)

        super().save(*args, update_fields=update_fields, **kwargs)

    @transaction.atomic
    @copy_signature(models.Model.delete)
    def delete(self, **kwargs):
        from kausal_common.datasets.models import Dataset, DatasetSchema, DatasetSchemaScope

        root_page = self.root_page
        if root_page is not None:
            self.root_page = None
            self.save(update_fields=['root_page'])
            root_page.get_descendants(inclusive=True).delete()

        pp = self.permission_policy()
        pp.admin_role.delete_instance_group(self)
        pp.viewer_role.delete_instance_group(self)
        pp.reviewer_role.delete_instance_group(self)
        pp.super_admin_role.delete_instance_group(obj=self)
        from pages.models import OutcomePage

        OutcomePage.objects.filter(outcome_node__instance=self).delete()
        self.nodes.all().delete()

        # Delete this instance's own dataset graph, but preserve anything shared with another scope.
        # A DatasetSchema (and its schema-scoped placeholder datasets) can be made available to
        # several instances via DatasetSchemaScope, so deleting one of them must not remove a schema
        # or placeholder that another instance still relies on. This matters especially during a
        # partial `destructively_trim_db` run, where a deleted instance can share a schema with a
        # retained one.
        own_scope = models.Q(scope_content_type=ContentType.objects.get_for_model(type(self)), scope_id=self.pk)
        own_schema_ids = set(DatasetSchemaScope.objects.qs.filter(own_scope).values_list('schema_id', flat=True))
        shared_schema_ids = set(
            DatasetSchemaScope.objects.qs
            .filter(schema_id__in=own_schema_ids)
            .exclude(own_scope)
            .values_list('schema_id', flat=True)
        )
        exclusive_schema_ids = own_schema_ids - shared_schema_ids
        # Schemas to check for orphanhood afterwards: those scoped exclusively to this instance, plus
        # the (possibly unscoped) schemas backing its directly-scoped datasets.
        affected_schema_ids = exclusive_schema_ids | {
            sid for sid in Dataset.objects.qs.filter(own_scope).values_list('schema_id', flat=True) if sid is not None
        }
        # Dataset revisions are protected while an instance revision pins them.
        # This explicit instance FK lets full instance deletion release all pins
        # before its owned datasets and their generic Wagtail revisions cascade.
        self.dataset_revision_pins.all().delete()
        # Delete the instance's own datasets: everything directly scoped to it (its own data, even
        # when the schema is shared), plus placeholder datasets whose schema is scoped only to this
        # instance. Placeholders of a schema shared with another scope are left for that scope.
        Dataset.objects.qs.filter(own_scope | models.Q(schema_id__in=exclusive_schema_ids)).delete()
        # Drop this instance's schema-scope links, then delete the schemas left with no scopes and no
        # datasets (so a shared schema, which keeps another scope or its datasets, survives).
        DatasetSchemaScope.objects.qs.filter(own_scope).delete()
        DatasetSchema.objects.qs.filter(pk__in=affected_schema_ids, scopes__isnull=True, datasets__isnull=True).delete()
        super().delete(**kwargs)

    def natural_key(self):
        return (self.identifier,)

    @classmethod
    def permission_policy(cls) -> InstanceConfigPermissionPolicy:
        return InstanceConfigPermissionPolicy()

    def model_cache_from_global(self) -> InstanceSpecificCache | None:
        if self._global_cache is None:
            return None
        return self._global_cache.for_instance(self)

    @classmethod
    def create_for_instance(cls, instance: Instance, **kwargs) -> InstanceConfig:
        assert not cls.objects.filter(identifier=instance.id).exists()

        org = Organization.objects.get(name='Kausal')  # TODO: Define the organization better when we have a better idea?
        # Identity metadata now lives on the columns (not in the spec), so
        # populate name/owner here, splitting TranslatedStrings into their
        # modeltrans parts.
        name_val, i18n = get_modeltrans_attrs_from_str(instance.name, 'name', instance.default_language)
        owner_val = ''
        if instance.owner:
            owner_val, owner_i18n = get_modeltrans_attrs_from_str(instance.owner, 'owner', instance.default_language)
            i18n.update(owner_i18n)
        fields = {
            'identifier': instance.id,
            'name': name_val,
            'owner': owner_val,
            'i18n': i18n,
            'organization': org,
            'primary_language': instance.default_language,
            'other_languages': [lang for lang in instance.supported_languages if lang != instance.default_language],
            'spec': make_minimal_instance_spec(instance),
            'yaml_mtime_hash': instance.config_mtime_hash,
            'yaml_spec_version': YAML_SPEC_VERSION,
            **kwargs,
        }
        ic = cls.objects.create(**fields)
        instance.config = ic
        return ic

    def has_framework_config(self) -> bool:
        try:
            _ = self.framework_config
        except ObjectDoesNotExist:
            return False
        else:
            return True

    def get_yaml_config_entrypoint(self) -> Path | None:
        config_base_dir = Path(settings.BASE_DIR, 'configs')
        if self.has_framework_config():
            fw = self.framework_config.framework
            fw_yaml_path = config_base_dir / f'{fw.identifier}.yaml'
            if fw_yaml_path.exists():
                return fw_yaml_path
            return None
        instance_yaml_path = config_base_dir / f'{self.identifier}.yaml'
        if instance_yaml_path.exists():
            return instance_yaml_path
        return None

    def update_instance_from_configs(self, instance: Instance, node_refs: bool = False):
        bind_reconciled_snapshots = instance.source_snapshot is None
        if bind_reconciled_snapshots:
            instance.source_nodes_by_uuid = {}
        for node_config in self.nodes_for_serialization:
            node = instance.context.nodes.get(node_config.identifier)
            if node is None:
                continue
            if bind_reconciled_snapshots:
                from .instance_serialization import NodeSnapshot, reconcile_node_snapshot_metadata

                source_snapshot = NodeSnapshot.from_runtime_node(
                    node,
                    uuid=node_config.uuid,
                    primary_language=self.primary_language,
                )
                node.source_snapshot = reconcile_node_snapshot_metadata(
                    source_snapshot,
                    node_config,
                    primary_language=self.primary_language,
                )
                instance.source_nodes_by_uuid[node_config.uuid] = node
                # Metrics and other runtime consumers read these fields directly,
                # so they must see the same reconciled metadata as GraphQL fields.
                node.name = node.source_snapshot.name or node.name
                node.short_name = node.source_snapshot.short_name
                node.description = node.source_snapshot.short_description
                node.color = node.source_snapshot.color or None
                node.order = node.source_snapshot.order
                node.is_visible = node.source_snapshot.is_visible
            node_config.update_node_from_config(node, keep_ref=node_refs)

    def update_identity_metadata(
        self,
        *,
        name: I18nString,
        owner: I18nString | None,
        primary_language: str,
        other_languages: list[str],
    ) -> None:
        """
        Seed YAML identity fields without overwriting DB-authored content.

        The configured language set still follows YAML because it determines
        how modeltrans interprets and exposes the stored translations.
        """
        for field_name, field_val in (('name', name), ('owner', owner)):
            if field_val is None:
                continue
            current_i18n = self.i18n or {}
            has_value = bool(getattr(self, field_name)) or any(
                key.startswith(f'{field_name}_') and value for key, value in current_i18n.items()
            )
            if has_value:
                continue

            val, field_i18n = get_modeltrans_attrs_from_str(
                cast('str | TranslatedString', field_val), field_name, primary_language
            )
            setattr(self, field_name, val)
            self.i18n = {**current_i18n, **field_i18n}

        self.primary_language = primary_language
        self.other_languages = [lang for lang in other_languages if lang != primary_language]

    def update_from_instance(self, instance: Instance, overwrite=False):
        """Update identity/content metadata columns from the instance but do not call save()."""

        for field_name in ('lead_title', 'lead_paragraph', 'name', 'owner'):
            field_val = getattr(instance, field_name)
            if field_val is None:
                continue
            val, i18n = get_modeltrans_attrs_from_str(field_val, field_name, instance.default_language)
            if not getattr(self, field_name, None) or overwrite:
                setattr(self, field_name, val)
                if self.i18n is None:
                    self.i18n = {}
                self.i18n.update(i18n)

        if self.primary_language != instance.default_language:
            self.log.info('Updating instance.primary_language to %s' % instance.default_language)
            self.primary_language = instance.default_language
        other_langs = set(instance.supported_languages) - {self.primary_language}
        if set(self.other_languages or []) != other_langs:
            self.log.info('Updating instance.other_languages to [%s]' % ', '.join(other_langs))
            self.other_languages = list(other_langs)
        if self.config_source == 'yaml':
            self.spec = make_minimal_instance_spec(instance)
            self.yaml_mtime_hash = instance.config_mtime_hash
            self.yaml_spec_version = YAML_SPEC_VERSION

    def serializable_data(self) -> dict[str, Any]:
        """
        Revision payload for a DB-sourced InstanceConfig.

        Deliberately *not* a ``super()`` call: Wagtail's default dumps every
        concrete field, which for this model includes both ``name`` and
        the modeltrans-synthesized ``name_i18n``. Modeltrans rejects
        round-tripping that duplication (``Attempted override of 'name'
        with 'name_i18n'``). ``name_i18n`` is a view-time projection of
        ``i18n`` under the active language anyway, so it doesn't belong
        in persisted revision content.

        Trailhead's restore path is ``_create_from_published_revision``,
        which reads ``model_snapshot.hydrate_dict`` directly — so the only
        load-bearing key below is ``model_snapshot``. The other fields
        exist for admin-side revision diffs and for
        ``from_serializable_data`` to look up the live row by pk.
        """
        from .instance_serialization import SNAPSHOT_SCHEMA_VERSION, build_instance_snapshot

        data: dict[str, Any] = {
            'pk': self.pk,
            'identifier': self.identifier,
            'config_source': self.config_source,
        }
        if self.config_source == 'database':
            snapshot = build_instance_snapshot(self, self._publication_dataset_revision_pins)
            data['model_snapshot'] = {
                'schema_version': SNAPSHOT_SCHEMA_VERSION,
                'structured': snapshot.model_dump(mode='json'),
            }
        return data

    @classmethod
    def from_serializable_data(
        cls,
        data: dict[str, Any],
        check_fks: bool = True,  # noqa: ARG003
        strict_fks: bool = False,  # noqa: ARG003
    ) -> InstanceConfig:
        """
        Return the live DB row for ``pk`` / ``identifier`` in ``data``.

        Wagtail's contract is "reconstruct the model's historical state
        from the revision blob" (Option B in the design discussion). We
        take a pragmatic shortcut: Trailhead's authoritative restore path
        is ``_create_from_published_revision``, which rebuilds the
        in-memory ``Instance`` from ``model_snapshot.hydrate_dict``. The
        ``InstanceConfig`` row itself is never reverted — its metadata
        (name, site, permissions) is the responsibility of the live row.
        Nothing downstream currently reads the object returned here, so
        returning the live row keeps ``with_content_json`` non-crashy
        without inventing reconstruction logic we don't use.

        If a future admin surface wants to preview historical revision
        state, revisit this to rebuild from ``data['model_snapshot']``.
        """
        pk = data.get('pk')
        if pk is not None:
            row = cls.objects.filter(pk=pk).first()
            if row is not None:
                return row
        identifier = data.get('identifier')
        if identifier:
            row = cls.objects.filter(identifier=identifier).first()
            if row is not None:
                return row
        # No live row to return — construct a blank instance so callers
        # don't crash. This branch is unexpected in practice.
        return cls(pk=pk, identifier=identifier or '')

    def clear_model_editor_data(self) -> None:
        """Delete all model editor related objects (input bindings) and reset spec."""
        self.input_bindings.all().delete()
        self.nodes.update(spec='{}')
        self.spec = InstanceModelSpec()

    @property
    def draft_head_token(self) -> UUID | None:
        """
        UUID of the most recent ``InstanceChangeOperation`` for this instance.

        This is the optimistic-locking token: every editing mutation passes
        the token it observed, and the server rejects the write if the
        current head has advanced. ``None`` means no edits have ever been
        recorded (fresh instance, or all operations deleted).
        """
        latest = self.change_operations.only('uuid').order_by('-created_at').first()
        return latest.uuid if latest is not None else None

    def validate_draft_constraints(self) -> None:
        """
        Run strict whole-graph constraint validation on the current draft.

        Raises ``InstanceConstraintError`` carrying the complete conflict set;
        used as the publication gate and before strict computation contexts.
        """
        from nodes.constraints.validation import require_valid_instance_constraints
        from nodes.instance_graph_cache import get_instance_graph, resolve_instance_source

        source = resolve_instance_source(self, PreferredInstanceSource.DRAFT)
        graph = get_instance_graph(self, PreferredInstanceSource.DRAFT, resolved_source=source)
        require_valid_instance_constraints(self, graph, source)

    def publish_instance(self, user: User | None = None) -> None:
        """Atomically publish the model and immutable revisions of its DB datasets."""
        from wagtail.models import Revision

        from nodes.instance_serialization import DatasetRevisionPinSnapshot

        with transaction.atomic():
            locked = InstanceConfig.objects.select_for_update().get(pk=self.pk)
            dataset_ids = list(
                NodeInputPortBinding.objects
                .filter(instance=locked, dataset__isnull=False)
                .order_by()
                .values_list('dataset_id', flat=True)
                .distinct()
            )
            datasets = list(
                DatasetModel.objects
                .select_for_update()
                .filter(pk__in=dataset_ids, is_external_placeholder=False)
                .only('pk', 'uuid', 'identifier', 'last_modified_at', 'latest_revision_id')
                .order_by('pk')
            )
            materializations = {
                materialization.dataset_id: materialization
                for materialization in DatasetMaterialization.objects.select_for_update().filter(
                    dataset_id__in=[dataset.pk for dataset in datasets],
                )
            }
            from nodes.dataset_materialization import (
                materialization_is_fresh,
                refresh_dataset_materialization,
                require_valid_dataset_rules,
            )

            for dataset in datasets:
                materialization = materializations.get(dataset.pk)
                if materialization is None or not materialization_is_fresh(dataset, materialization):
                    materialization = refresh_dataset_materialization(dataset, touch=False)
                    materializations[dataset.pk] = materialization

            # Publication is the strictness boundary: a draft may carry
            # structural constraint conflicts and stay inspectable, but a
            # conflicted graph never becomes a published revision. Runs after
            # the materialization refresh so shape profiles read the same
            # observed facts the revision will pin.
            locked.validate_draft_constraints()
            # Dataset validation rules gate publication the same way: the
            # violations were just re-evaluated by the refresh above.
            require_valid_dataset_rules(materializations.values())

            dataset_ct = ContentType.objects.get_for_model(DatasetModel, for_concrete_model=False)
            now = timezone.now()
            dataset_revisions: list[Revision] = [
                Revision(
                    content_type=dataset_ct,
                    base_content_type=dataset_ct,
                    object_id=str(dataset.pk),
                    created_at=now,
                    user=user,
                    object_str=dataset.identifier or str(dataset.uuid),
                    content=materializations[dataset.pk].content,
                )
                for dataset in datasets
            ]
            Revision.objects.bulk_create(dataset_revisions)

            pins_by_dataset: dict[int, DatasetRevisionPinSnapshot] = {}
            revisions_by_dataset: dict[int, Revision] = {}
            for dataset, dataset_revision in zip(datasets, dataset_revisions, strict=True):
                materialization = materializations[dataset.pk]
                dataset.latest_revision = dataset_revision
                revisions_by_dataset[dataset.pk] = dataset_revision
                pins_by_dataset[dataset.pk] = DatasetRevisionPinSnapshot(
                    dataset_uuid=dataset.uuid,
                    identifier=dataset.identifier,
                    revision_id=dataset_revision.pk,
                    content_hash=materialization.content_hash,
                    generation=materialization.generation,
                    forecast_from=materialization.forecast_from,
                )
            if datasets:
                DatasetModel.objects.bulk_update(datasets, ['latest_revision'])

            locked._publication_dataset_revision_pins = pins_by_dataset
            try:
                revision = locked.save_revision(user=user)
            finally:
                locked._publication_dataset_revision_pins = None

            InstanceRevisionDatasetPin.objects.bulk_create([
                InstanceRevisionDatasetPin(
                    instance_config=locked,
                    instance_revision=revision,
                    dataset=dataset,
                    dataset_revision=revisions_by_dataset[dataset.pk],
                    dataset_uuid=dataset.uuid,
                    identifier=dataset.identifier,
                    forecast_from=materializations[dataset.pk].forecast_from,
                    shape_profiles=materializations[dataset.pk].shape_profiles,
                )
                for dataset in datasets
            ])
            locked.publish(revision, user=user)
            locked.invalidate_cache()

            self.latest_revision_id = locked.latest_revision_id
            self.live_revision_id = locked.live_revision_id
            self.cache_invalidated_at = locked.cache_invalidated_at

    def revert_to_published(self) -> None:
        """Restore draft state from the published revision snapshot."""
        # TODO: Rewrite for spec-based storage
        raise NotImplementedError('revert_to_published needs rewriting for spec-based storage')

    def _complete_legacy_snapshot_content(self, snapshot: InstanceSnapshot) -> None:
        """
        Fill fields that pre-v6 revisions never persisted.

        Historical values do not exist for these fields, so the only
        backwards-compatible value is the current row value. Keeping this
        one-time compatibility read at the revision boundary prevents public
        GraphQL resolvers from acquiring live-row fallbacks of their own.
        """
        current_metadata = InstanceMetadata.from_model(self)
        snapshot.metadata.lead_title = current_metadata.lead_title
        snapshot.metadata.lead_paragraph = current_metadata.lead_paragraph

        bodies_by_uuid = {
            node.uuid: list(node.body.raw_data) if node.body else None for node in self.nodes.get_queryset().only('uuid', 'body')
        }
        for node_snapshot in snapshot.nodes:
            node_snapshot.body = bodies_by_uuid.get(node_snapshot.uuid)

    def _create_from_published_revision(self, node_refs: bool = False) -> Instance | None:
        """
        Hydrate an Instance from the latest published revision, if any.

        Returns ``None`` if the instance has never been published, so the
        caller can fall back to the draft (tables) path.
        """
        from .instance_loader import InstanceLoader

        rev = self.live_revision
        if rev is None:
            return None
        content = rev.content or {}
        snapshot_data = content.get('model_snapshot') or {}
        structured = snapshot_data.get('structured')
        if structured is not None:
            from kausal_common.i18n.pydantic import set_i18n_context

            from .instance_serialization import InstanceSnapshot

            source_schema_version = structured.get('schema_version', 1)
            raw_metadata = structured.get('metadata') or {}
            primary_language = raw_metadata.get('primary_language', self.primary_language)
            other_languages = raw_metadata.get('other_languages', self.other_languages or [])
            with set_i18n_context(primary_language, other_languages):
                snapshot = InstanceSnapshot.from_serialized_data(structured)
                if source_schema_version < 6:
                    self._complete_legacy_snapshot_content(snapshot)
            if source_schema_version >= 7:
                expected_pins = {(pin.dataset_uuid, pin.revision_id) for pin in snapshot.dataset_revisions}
                persisted_pins = set(
                    self.dataset_revision_pins.filter(instance_revision=rev).values_list(
                        'dataset_uuid',
                        'dataset_revision_id',
                    )
                )
                if persisted_pins != expected_pins:
                    raise RuntimeError(
                        f'Instance revision {rev.pk} dataset manifest mismatch: '
                        f'snapshot={sorted(map(str, expected_pins))}, persisted={sorted(map(str, persisted_pins))}',
                    )
            instance = InstanceLoader.from_snapshot(
                snapshot,
                published=source_schema_version >= 7,
                instance_config=self,
            ).instance
            instance.bind_source_snapshot(snapshot)
            return instance
        # Legacy revisions carry only the serialized config dict; the
        # config-dict loader is gone, so they fall back to the draft.
        # Republishing is the migration path (no such revisions are known
        # to exist -- publication opened after the snapshot restructure).
        self.log.warning('Published revision predates the structured snapshot; serving the draft instead')
        return None

    def _create_from_config(
        self,
        node_refs: bool = False,
        source: PreferredInstanceSource | Literal['draft', 'published'] = PreferredInstanceSource.DRAFT,
        tolerate_node_failures: bool = False,
    ) -> Instance:
        from .instance_loader import InstanceLoader

        # Defensively invalidate the cached node list: a prior read might
        # have warmed it before a post-publish edit, and we want to see
        # current DB state here.
        self._nodes_for_serialization = None

        if self.config_source == 'database':
            if source == PreferredInstanceSource.PUBLISHED:
                instance = self._create_from_published_revision(node_refs=node_refs)
                if instance is not None:
                    return instance
                # Fall through to the draft path if no published revision exists.

            from .instance_from_db import _check_dimension_orm_coverage
            from .instance_serialization import build_instance_snapshot

            _check_dimension_orm_coverage(self)
            snapshot = build_instance_snapshot(self)
            loader = InstanceLoader.from_snapshot(
                snapshot,
                tolerate_node_failures=tolerate_node_failures,
                instance_config=self,
            )
            instance = loader.instance
            instance.bind_source_snapshot(snapshot)
            self.update_instance_from_configs(instance, node_refs=True)
            return instance

        if self.has_framework_config():
            fwc = self.framework_config
            instance = fwc.create_model_instance(self)
            with sentry_sdk.start_span(name='update-instance-from-configs: %s' % self.identifier, op='function'):
                self.update_instance_from_configs(instance, node_refs=node_refs)
        else:
            config_fn = self.get_yaml_config_entrypoint()
            if config_fn is None:
                raise ValueError(f'No YAML config entrypoint found for instance {self.identifier}')
            self.log.debug('Creating instance from YAML file: %s' % config_fn)
            loader = InstanceLoader.from_yaml(
                config_fn,
                tolerate_node_failures=tolerate_node_failures,
                instance_config=self,
            )
            instance = loader.instance
            with sentry_sdk.start_span(name='update-instance-from-configs: %s' % self.identifier, op='function'):
                # We only need to do this on the plain old YAML path
                self.update_instance_from_configs(instance, node_refs=node_refs)

        return instance

    def _initialize_instance(
        self,
        node_refs: bool = False,
        source: PreferredInstanceSource = PreferredInstanceSource.DRAFT,
        tolerate_node_failures: bool = False,
    ) -> Instance:
        self.log.info(
            'Creating new instance from %s (source=%s)'
            % (
                'database' if self.config_source == 'database' else 'YAML config',
                source.value,
            )
        )

        with sentry_sdk.start_span(name='create-instance-from-config: %s' % self.identifier, op='function'):
            instance = self._create_from_config(
                node_refs=node_refs,
                source=source,
                tolerate_node_failures=tolerate_node_failures,
            )

        instance.config = self
        instance.modified_at = timezone.now()
        if settings.ENABLE_PERF_TRACING:
            instance.context.perf_context.enabled = True
        return instance

    def set_instance_scope(self, scope: sentry_sdk.Scope | None = None) -> None:
        if scope is None:
            scope = sentry_sdk.get_current_scope()
        scope.set_tag('instance_id', self.identifier)
        scope.set_tag('instance_uuid', str(self.uuid))

    @contextmanager
    def enter_instance_context(
        self,
        source: PreferredInstanceSource = PreferredInstanceSource.DRAFT,
        tolerate_node_failures: bool = False,
        force_reinitialize: bool = False,
    ):
        if not force_reinitialize and self.identifier in _pytest_instances:
            instance = _pytest_instances[self.identifier]
        else:
            instance = self._initialize_instance(node_refs=True, source=source, tolerate_node_failures=tolerate_node_failures)

        # Set explicitly every time (default False) so the flag never leaks across requests
        # that reuse a cached instance/context. See docs/architecture/fault-tolerance.md.
        instance.context.tolerate_node_failures = tolerate_node_failures

        token = instance_context.set(instance)
        try:
            with sentry_sdk.new_scope() as scope, logger.contextualize(instance=self.identifier):
                self.set_instance_scope(scope)
                yield instance
        finally:
            instance_context.reset(token)

    @asynccontextmanager
    async def enter_instance_context_async(
        self,
        source: PreferredInstanceSource = PreferredInstanceSource.DRAFT,
        tolerate_node_failures: bool = False,
        force_reinitialize: bool = False,
    ):
        if not force_reinitialize and self.identifier in _pytest_instances:
            instance = _pytest_instances[self.identifier]
        else:
            instance = await sync_to_async(self._initialize_instance)(
                node_refs=True,
                source=source,
                tolerate_node_failures=tolerate_node_failures,
            )

        # Set explicitly every time (default False) so the flag never leaks across requests.
        instance.context.tolerate_node_failures = tolerate_node_failures

        token = instance_context.set(instance)
        try:
            with sentry_sdk.new_scope() as scope:
                self.set_instance_scope(scope)
                yield instance
        finally:
            instance_context.reset(token)

    def _get_instance(
        self,
        node_refs: bool = False,
        source: PreferredInstanceSource = PreferredInstanceSource.DRAFT,
    ) -> Instance:
        if self.identifier in _pytest_instances:
            return _pytest_instances[self.identifier]

        current_instance = instance_context.get()
        if current_instance is not None and current_instance.id == self.identifier:
            # Trust the ContextVar: whoever entered the request-scoped
            # context already chose a source, and all in-request resolvers
            # should see the same Instance.
            return current_instance

        with instance_cache_lock:
            instance = self._initialize_instance(node_refs=node_refs, source=source)
        return instance

    def get_instance(
        self,
        node_refs: bool = False,
        source: PreferredInstanceSource = PreferredInstanceSource.DRAFT,
    ) -> Instance:
        # Unit tests will set the Instance to `_instance` so that we don't need
        # to read the YAML configs
        instance = self._get_instance(node_refs=node_refs, source=source)
        return instance

    def get_name(self) -> str:
        if self.name:
            return self.name
        instance = self.get_instance()
        return str(instance.name)

    def _get_spec_from_yaml(self) -> tuple[InstanceModelSpec, str, list[str], str] | None:
        """Return ``(spec, primary_language, other_languages, mtime_hash)`` from the YAML entrypoint."""
        from .instance_loader import InstanceYAMLConfig

        config_fn = self.get_yaml_config_entrypoint()
        if config_fn is None:
            return None
        yaml_conf = InstanceYAMLConfig.load_for_entrypoint(config_fn)
        data = yaml_conf.data
        assert data is not None
        primary_language = data['default_language']
        other_languages = list(data.get('supported_languages') or [])
        from kausal_common.i18n.pydantic import set_i18n_context

        with set_i18n_context(primary_language, other_languages):
            spec = make_minimal_instance_spec(data)
        mtime_hash = yaml_conf.meta.mtime_hash or yaml_conf.meta.calculate_mtime_hash()
        return spec, primary_language, other_languages, mtime_hash

    def ensure_spec(self, update_self: bool = True, save: bool = True) -> InstanceModelSpec:
        if self.config_source != 'yaml':
            return self._ensure_database_spec(save=save)

        if (
            self.spec is not None
            and self.yaml_spec_version == YAML_SPEC_VERSION
            and self._verified_yaml_spec_hash == self.yaml_mtime_hash
        ):
            return self.spec

        yaml_ret = self._get_spec_from_yaml()
        if yaml_ret is None:
            if self.spec is not None:
                return self.spec
            raise ValueError(f'No YAML config entrypoint found for instance {self.identifier}')

        spec, primary_language, other_languages, yaml_mtime_hash = yaml_ret
        if self.spec is not None and self.yaml_spec_version == YAML_SPEC_VERSION and self.yaml_mtime_hash == yaml_mtime_hash:
            self._verified_yaml_spec_hash = yaml_mtime_hash
            return self.spec

        if not save and not update_self:
            return spec

        self.spec = spec
        self.yaml_mtime_hash = yaml_mtime_hash
        self.yaml_spec_version = YAML_SPEC_VERSION
        self._verified_yaml_spec_hash = yaml_mtime_hash
        self.primary_language = primary_language
        self.other_languages = other_languages
        if save:
            self.save(update_fields=['primary_language', 'other_languages', 'spec', 'yaml_mtime_hash', 'yaml_spec_version'])
        return self.spec

    def _ensure_database_spec(self, *, save: bool) -> InstanceModelSpec:
        spec = self.spec
        if spec is None:
            assert self.pk is None, f'Persisted database-sourced instance {self.identifier} has no spec'
            # Database-sourced identity metadata lives on the columns, so a
            # newly created blank model starts with an empty computation spec.
            spec = InstanceModelSpec()
            if save:
                self.spec = spec
                self.save(update_fields=['spec'])
        return spec

    @property
    def default_language(self) -> str:
        return self.primary_language

    @property
    def supported_languages(self) -> list[str]:
        return [self.primary_language, *self.other_languages]

    @property
    def theme_identifier(self) -> str:
        spec = self.ensure_spec()
        if spec.theme_identifier is not None:
            return spec.theme_identifier
        return 'default'

    @cached_property
    def action_list_page(self) -> ActionListPage | None:
        from pages.models import ActionListPage

        if self.root_page is None:
            return None
        qs = self.root_page.get_descendants().type(ActionListPage).specific()
        return cast('ActionListPage | None', qs.first())

    def get_translated_root_page(self) -> Page:
        """Return root page in activated language, fall back to default language."""
        root = self.root_page
        assert root is not None
        language = get_language()
        language = convert_language_code(language, 'wagtail')
        try:
            locale = Locale.objects.get(language_code=language)
            root = root.get_translation(locale)
        except Locale.DoesNotExist, Page.DoesNotExist:
            pass
        return root

    @staticmethod
    def _format_url_from_hostname(
        hostname: str,
        base_path: str = '',
        *,
        scheme: str | None = None,
        port: int | None = None,
    ) -> str:
        if scheme is None:
            scheme = 'http' if hostname == 'localhost' or hostname.endswith('.localhost') else 'https'
        port_str = f':{port}' if port else ''
        return f'{scheme}://{hostname}{port_str}{base_path.rstrip("/")}'

    @staticmethod
    def _get_client_url_parts(
        request: HttpRequest | None = None,
        client_url: str | None = None,
    ) -> tuple[str, str, int | None] | None:
        from paths.schema_context import PathsGraphQLContext

        if client_url is None and request is None:
            return None

        if not client_url and request is not None:
            if isinstance(request, PathsGraphQLContext):
                headers = request.get_request_headers()
            else:
                headers = request.headers
            client_url = headers.get('origin') or headers.get('referer')
        if not client_url:
            return None

        parts = urlparse(client_url)
        if parts.scheme not in ('http', 'https') or not parts.hostname:
            return None
        try:
            port = parts.port
        except ValueError:
            port = None
        if (parts.scheme == 'https' and port == 443) or (parts.scheme == 'http' and port == 80):
            port = None
        return parts.scheme, parts.hostname.lower(), port

    def get_view_url(self, request: HttpRequest | None = None, client_url: str | None = None) -> str | None:
        client_parts = self._get_client_url_parts(request=request, client_url=client_url)
        if client_parts is not None:
            scheme, hostname, port = client_parts
            hn = self.hostnames.filter(hostname__iexact=hostname).first()
            if hn is not None:
                return self._format_url_from_hostname(hn.hostname, hn.base_path, scheme=scheme, port=port)

        wildcard_domain = settings.INSTANCE_WILDCARD_DOMAIN
        if not wildcard_domain:
            return None
        hn = self.hostnames.order_by('pk').first()
        if hn is not None:
            return self._format_url_from_hostname(hn.hostname, hn.base_path)

        if client_parts is not None:
            scheme, hostname, port = client_parts
            if hostname == wildcard_domain or hostname.endswith(f'.{wildcard_domain}'):
                return self._format_url_from_hostname(f'{self.identifier}.{wildcard_domain}', scheme=scheme, port=port)

        hostname = f'{self.identifier}.{wildcard_domain}'
        return self._format_url_from_hostname(hostname)

    def add_hostname_from_url(self, url: str) -> InstanceHostname | None:
        parts = urlparse(url)
        if not parts.hostname:
            return None
        base_path = parts.path.rstrip('/')
        hostname = parts.hostname.lower()
        hostname_obj = self.hostnames.filter(hostname=hostname).first()
        if hostname_obj is not None:
            if (
                hostname_obj.base_path != base_path
                and not InstanceHostname.objects.filter(
                    hostname=hostname,
                    base_path=base_path,
                ).exists()
            ):
                hostname_obj.base_path = base_path
                hostname_obj.save(update_fields=['base_path'])
            return hostname_obj
        if InstanceHostname.objects.filter(hostname=hostname, base_path=base_path).exists():
            return None
        return InstanceHostname.objects.create(instance=self, hostname=hostname, base_path=base_path)

    def sync_nodes(self, update_existing=False, delete_stale=False, overwrite=False, skip_descriptions=False):
        from nodes.datasets import DBDataset

        instance = self.get_instance()
        node_configs = {n.identifier: n for n in self.nodes.all()}
        found_nodes = set()
        for node in instance.context.nodes.values():
            node_config = node_configs.get(node.id)
            if node_config is None:
                node_config = NodeConfig(instance=self, **node.as_node_config_attributes())
                self.log.info('Creating node config for node %s' % node.id)
                node_config.save()
                has_db_datasets = any(isinstance(ds, DBDataset) for ds in node.input_dataset_instances)
                if has_db_datasets:
                    node_config.update_relations_from_node(node)
                node.database_id = node_config.pk
            else:
                found_nodes.add(node.id)
                if update_existing:
                    node_config.update_from_node(node, overwrite=overwrite, skip_descriptions=skip_descriptions)
                    node_config.save()

        for node in list(node_configs.values()):
            if node.identifier in found_nodes:
                continue

            self.log.info("Node %s exists in database, but it's not found in node graph" % node.identifier)
            if delete_stale:
                node.delete()

    def sync_categories(
        self,
        dataset_dim: DatasetDimensionModel,
        scope: DimensionScope,
        dim: NodeDimension,
        update_existing=False,
        delete_stale=False,
    ):
        found_cats = set()
        default_lang = self.primary_language
        assert scope.identifier is not None

        from datasets.defs import DimensionCategorySpec

        cats = {cat.identifier: cat for cat in dataset_dim.categories.all()}
        for order, cat in enumerate(dim.categories):
            cat_obj = cats.get(cat.id)
            label, i18n = get_modeltrans_attrs_from_str(cat.label, 'label', default_lang)
            cat_spec = DimensionCategorySpec.from_runtime(cat).to_json()
            if cat_obj is None:
                cat_obj = DimensionCategory.objects.create(
                    dimension=dataset_dim,
                    identifier=cat.id,
                    label=label,
                    i18n=i18n,
                    spec=cat_spec,
                    order=order,
                )
                print('Creating category %s' % cat.id)
            else:
                found_cats.add(cat_obj.pk)
                # OrderableModel ordering starts from 1
                changed = (
                    i18n != cat_obj.i18n or cat_obj.label != label or cat_obj.spec != cat_spec or cat_obj.order != (order + 1)
                )
                if changed:
                    cat_obj.label, cat_obj.i18n, cat_obj.spec, cat_obj.order = label, i18n, cat_spec, order + 1
                    print('Updating category %s' % cat.id)
                    cat_obj.save()

        if delete_stale:
            for cat_obj in cats.values():
                if cat_obj.pk in found_cats:
                    continue
                print('Deleting stale category %s' % cat_obj)
                cat_obj.delete()

    def sync_dimension(
        self,
        dim: NodeDimension,
        update_existing=False,
        delete_stale=False,
    ) -> DatasetDimensionModel:
        from datasets.defs import DimensionSpec

        scope = DimensionScope.objects.filter(
            scope_content_type=ContentType.objects.get_for_model(self),
            scope_id=self.pk,
            identifier=dim.id,
        ).first()
        label, i18n = get_modeltrans_attrs_from_str(dim.label, 'name', self.primary_language)
        dim_spec = DimensionSpec.from_runtime(dim).to_json()
        if scope is None:
            dim_obj = DatasetDimensionModel.objects.create(name=label, i18n=i18n, spec=dim_spec)
            scope = DimensionScope.objects.create(
                scope_content_type=ContentType.objects.get_for_model(self), scope_id=self.pk, identifier=dim.id, dimension=dim_obj
            )
            print('Creating dimension %s' % dim.id)
        else:
            dim_obj = scope.dimension

        if update_existing and (dim_obj.name != label or dim_obj.i18n != i18n or dim_obj.spec != dim_spec):
            if dim_obj.pk:
                print('Updating dimension %s' % dim.id)
            dim_obj.name = label
            dim_obj.i18n = i18n
            dim_obj.spec = dim_spec
            dim_obj.save()

        self.sync_categories(
            dataset_dim=dim_obj,
            scope=scope,
            dim=dim,
            update_existing=update_existing,
            delete_stale=delete_stale,
        )
        return dim_obj

    def sync_dimensions(self, update_existing=False, delete_stale=False, instance: Instance | None = None) -> None:
        if instance is None:
            instance = self.get_instance()
        found_dims = set()
        for dim in instance.context.dimensions.values():
            obj = self.sync_dimension(dim, update_existing=update_existing, delete_stale=delete_stale)
            found_dims.add(obj)

        if delete_stale:
            dimensions = DatasetDimensionModel.objects.filter(
                scopes__scope_content_type=ContentType.objects.get_for_model(self),
                scopes__scope_id=self.pk,
            )
            for dim_obj in dimensions:
                if dim_obj not in found_dims:
                    dim_obj.delete()

    def update_modified_at(self, save=True):
        self.modified_at = timezone.now()
        if save:
            self.save(update_fields=['modified_at'])

    def get_outcome_nodes(self) -> list[NodeConfig]:
        instance = self.get_instance()
        ctx = instance.context
        root_nodes = ctx.get_outcome_nodes()
        pks = [node.database_id for node in root_nodes if node.database_id is not None]
        return list(self.nodes.filter(pk__in=pks))

    def _create_instance_root_page(self) -> Page:
        from pages.models import InstanceRootPage

        root_node: Page = cast('Page', Page.get_first_root_node())
        with override(self.primary_language):
            locale, _ = Locale.objects.get_or_create(language_code=self.primary_language)
            page = root_node.add_child(
                instance=InstanceRootPage(
                    locale=locale,
                    title=self.get_name(),
                    slug=self.identifier,
                    url_path='',
                )
            )
        return page

    def _create_default_pages(self) -> Page:  # noqa: C901
        from pages.models import ActionListPage, OutcomePage

        root = cast('Page', Page.get_first_root_node())
        home_pages = root.get_children()

        instance = self.get_instance()
        outcome_nodes = {node.identifier: node for node in self.get_outcome_nodes()}
        # Create default pages only in default language for now
        # TODO: Also create translations to other supported languages

        home_page_conf: OutcomePageConfig | None = None
        for page in instance.pages:
            if page.id == 'home':
                home_page_conf = page
                break
        if home_page_conf is None:
            if not outcome_nodes:
                return self._create_instance_root_page()

            onode = outcome_nodes.get('net_emissions') or next(iter(outcome_nodes.values()))
        else:
            assert home_page_conf.outcome_node is not None
            onode = outcome_nodes.get(home_page_conf.outcome_node)
            if onode is None:
                raise ValueError(f"Your node '{home_page_conf.outcome_node}' is not an outcome node.")

        if onode is None:
            raise ValueError('No outcome node found for the instance.')

        root_node: Page = cast('Page', Page.get_first_root_node())
        with override(self.primary_language):
            locale, _ = Locale.objects.get_or_create(language_code=self.primary_language)
            try:
                home_page = home_pages.get(slug=self.identifier)
            except Page.DoesNotExist:
                home_page = root_node.add_child(
                    instance=OutcomePage(
                        locale=locale,
                        title=self.get_name(),
                        slug=self.identifier,
                        url_path='',
                        outcome_node=onode,
                    )
                )

            action_list_pages: models.QuerySet[ActionListPage] = home_page.get_children().type(ActionListPage)  # type: ignore
            if not action_list_pages.exists():
                home_page.add_child(
                    instance=ActionListPage(
                        title=gettext('Actions'),
                        slug='actions',
                        show_in_menus=True,
                        show_in_footer=True,
                    )
                )

            for page_config in instance.pages:
                slug = page_config.id
                if slug == 'home':
                    continue

                page = cast('OutcomePage', home_page.get_children().filter(slug=slug).first())
                if page is not None:
                    continue

                assert page_config.outcome_node is not None
                home_page.add_child(
                    instance=OutcomePage(
                        locale=locale,
                        title=str(page_config.name),
                        slug=slug,
                        url_path=page_config.path,
                        outcome_node=outcome_nodes[page_config.outcome_node],
                        show_in_menus=page_config.show_in_menus,
                        show_in_footer=page_config.show_in_footer,
                    )
                )

        return home_page

    def create_default_content(self):
        self.create_or_update_instance_groups()
        root_page = self._create_default_pages()
        if self.root_page is None:
            self.root_page = root_page
            self.save(update_fields=['root_page'])

    def create_or_update_instance_groups(self):
        pp = self.permission_policy()
        pp.admin_role.create_or_update_instance_group(self)
        pp.viewer_role.create_or_update_instance_group(self)
        # For now, try not to proliferate reviewer groups for NZC instances.
        if not self.has_framework_config() or self.framework_config.framework.identifier != 'nzc':
            pp.reviewer_role.create_or_update_instance_group(self)
        pp.super_admin_role.create_or_update_instance_group(self)

    def invalidate_cache(self, save: bool = True):
        self.cache_invalidated_at = timezone.now()
        self.log.info('Invalidating cache')
        if save:
            self.save(update_fields=['cache_invalidated_at'])

    def notify_change(self):
        self.update_modified_at(save=False)
        self.invalidate_cache(save=False)
        self.log.info('Instance modified')
        self.save(update_fields=['modified_at', 'cache_invalidated_at'])
        cl = get_channel_layer()
        if cl is None:
            return
        async_to_sync(cl.group_send)(
            INSTANCE_CHANGE_GROUP,
            {
                'type': INSTANCE_CHANGE_TYPE,
                'pk': self.pk,
            },
        )

    @property
    def nodes_for_serialization(self) -> list[NodeConfig]:
        """
        Node rows laid out for serialization, with a manual lazy cache.

        Not a ``cached_property``: hydrate paths (``_create_from_config``)
        clear ``_nodes_for_serialization`` at entry so the next read sees
        any post-publish edits. In production each request gets a fresh
        InstanceConfig row, but the same object is reused across
        save_revision/publish/hydrate in tests and in any future code
        that holds the row across a commit.
        """
        cached = getattr(self, '_nodes_for_serialization', None)
        if cached is not None:
            return cached
        fresh = list(self.nodes.get_queryset().for_serialization())
        self._nodes_for_serialization = fresh
        return fresh

    @cached_property
    def log(self) -> Logger:
        return logger.bind(instance=self.identifier, markup=True)


class InstanceHostnameManager(Manager['InstanceHostname']):
    def get_by_natural_key(self, instance_identifier, hostname, base_path):
        instance = InstanceConfig.objects.get_by_natural_key(instance_identifier)
        return self.get(instance=instance, hostname=hostname, base_path=base_path)


class InstanceHostname(models.Model):
    instance = models.ForeignKey(
        InstanceConfig,
        on_delete=models.CASCADE,
        related_name='hostnames',
    )
    hostname = models.CharField(max_length=100)
    base_path = models.CharField(max_length=100, blank=True, default='')

    extra_script_urls = ArrayField(models.URLField(max_length=300), default=list)

    objects = InstanceHostnameManager()

    class Meta:
        verbose_name = _('Instance hostname')
        verbose_name_plural = _('Instance hostnames')
        unique_together = (('instance', 'hostname'), ('hostname', 'base_path'))
        ordering = ['instance', 'hostname', 'base_path']

    def __str__(self):
        return '%s at %s [basepath %s]' % (self.instance, self.hostname, self.base_path)

    def natural_key(self):
        return self.instance.natural_key() + (self.hostname, self.base_path)


class InstanceTokenManager(Manager['InstanceToken']):
    def get_by_natural_key(self, instance_identifier, token, created_at):
        instance = InstanceConfig.objects.get_by_natural_key(instance_identifier)
        return self.get(instance=instance, token=token, created_at=created_at)


class InstanceToken(models.Model):
    instance = models.ForeignKey(
        InstanceConfig,
        on_delete=models.CASCADE,
        related_name='tokens',
    )
    token = models.CharField(max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)

    objects = InstanceTokenManager()

    class Meta:
        verbose_name = _('Instance token')
        verbose_name_plural = _('Instance tokens')
        ordering = ['instance', '-created_at']

    def __str__(self) -> str:
        return 'Token for %s' % str(self.instance)

    def natural_key(self):
        return self.instance.natural_key() + (self.token, self.created_at)


class NodeConfigQuerySet(MultilingualQuerySet['NodeConfig'], PathsQuerySet['NodeConfig']):  # type: ignore[override, misc]
    def active(self) -> Self:
        return self.filter(is_stale=False)

    def with_spec(self) -> Self:
        return self.defer(None)

    def annotate_ports(self) -> Self:
        """
        Attach the node's port bindings as ``PortBindingDef``-shaped JSON.

        Served from the authoritative ``NodeInputPortBinding`` table. This is
        the projection behind ``port_edge_bindings`` / ``port_dataset_bindings``.
        """
        edge_bindings = (
            NodeInputPortBinding.objects
            .filter(Q(node=OuterRef('pk')) | Q(source_node=OuterRef('pk')), source_node__isnull=False)
            .order_by('node_id', 'port_id', 'position')
            .annotate(
                obj=JSONObject(
                    id=F('uuid'),
                    from_ref=JSONObject(
                        node_uuid=F('source_node__uuid'),
                        node_id=F('source_node__identifier'),
                        port_id=F('source_port_id'),
                    ),
                    port_ref=JSONObject(
                        node_uuid=F('node__uuid'),
                        node_id=F('node__identifier'),
                        port_id=F('port_id'),
                    ),
                    position=F('position'),
                    transformations=F('transformations'),
                    tags=F('tags'),
                ),
            )
            .values('obj')
        )
        dataset_bindings = (
            NodeInputPortBinding.objects
            .filter(node=OuterRef('pk'), dataset__isnull=False)
            .order_by('port_id', 'position')
            .annotate(
                obj=JSONObject(
                    id=F('uuid'),
                    port_ref=JSONObject(
                        node_uuid=F('node__uuid'),
                        node_id=F('node__identifier'),
                        port_id=F('port_id'),
                    ),
                    position=F('position'),
                    dataset_uuid=F('dataset__uuid'),
                    metric_uuid=F('metric__uuid'),
                    dataset_is_external_placeholder=F('dataset__is_external_placeholder'),
                    dataset_external_ref=F('dataset__external_ref'),
                    external_dataset_id=F('dataset__identifier'),
                    external_metric_id=F('metric__name'),
                    transformations=F('transformations'),
                    tags=F('tags'),
                ),
            )
            .values('obj')
        )
        return self.annotate(
            _annotated_port_edge_bindings=ArraySubquery(edge_bindings),
            _annotated_port_dataset_bindings=ArraySubquery(dataset_bindings),
        )

    def for_serialization(self) -> Self:
        return self.active().with_spec().select_related('indicator_node', 'copy_of', 'layout').annotate_ports()


_NodeConfigManager = models.Manager.from_queryset(NodeConfigQuerySet)


class NodeConfigManager(MLModelManager['NodeConfig', NodeConfigQuerySet], _NodeConfigManager):  # pyright: ignore
    """Model manager for NodeConfig."""

    def get_queryset(self) -> NodeConfigQuerySet:
        return super().get_queryset().defer('spec')

    def get_by_natural_key(self, instance_identifier, identifier):
        instance = InstanceConfig.objects.get_by_natural_key(instance_identifier)
        return self.get(instance=instance, identifier=identifier)

    if TYPE_CHECKING:

        def active(self) -> NodeConfigQuerySet: ...
        def with_spec(self) -> NodeConfigQuerySet: ...
        def annotate_ports(self) -> NodeConfigQuerySet: ...


del _NodeConfigManager


def make_empty_node_spec() -> NodeSpec:
    return NodeSpec()


class EditableInstanceChild(
    UUIDIdentifiedModel,
    UserModifiableModel,
    RevisionMixin,
    ClusterableModel,
):
    """
    Abstract superclass for ORM rows that compose an editable InstanceConfig.

    Bundles:
      * ``UUIDIdentifiedModel`` — stable UUID for cross-system references
      * ``UserModifiableModel`` — ``created_at`` / ``created_by`` /
        ``last_modified_at`` / ``last_modified_by`` for ordering + future
        ``is_creator(obj)``-style permission conditions
      * ``RevisionMixin`` — per-row Wagtail revision history (redundant
        with IMLE audit, but cheap and valued for recovery)
      * ``ClusterableModel`` — Wagtail form/revision machinery

    Subclasses declare ``snapshot_model``, a ``ModelSnapshot`` subtype that
    mirrors the row's state. ``serializable_data()`` (overridden from
    Wagtail's default) dumps through ``snapshot_model.from_model(self)`` so
    the stored revision content is the snapshot-shaped dict.

    ``apply_snapshot`` is the inverse — used by undo/revert to bring a row
    (looked up by uuid) back to a prior snapshot. Signature differs from
    Wagtail's ``from_serializable_data`` because we need the parent
    ``InstanceConfig`` to bind FKs; subclasses implement it when the
    upsert rules become relevant (Phase 5+).
    """

    snapshot_model: ClassVar[type[ModelSnapshot]]

    class Meta:
        abstract = True

    def serializable_data(self) -> dict[str, Any]:
        return self.snapshot_model.from_model(self).model_dump(mode='json')

    @classmethod
    def apply_snapshot(
        cls,
        data: dict[str, Any],
        *,
        instance_config: InstanceConfig,
    ) -> Self:
        msg = f'{cls.__name__}.apply_snapshot is not implemented yet'
        raise NotImplementedError(msg)


class NodeConfigPermissionPolicy(
    ParentInheritedPolicy['NodeConfig', InstanceConfig, NodeConfigQuerySet, InstanceConfig],
):
    """Instance-inherited permissions with a superuser-bypassable edit lock."""

    def __init__(self):
        super().__init__(NodeConfig, InstanceConfig, 'instance', create_context_type=InstanceConfig)

    def construct_perm_q(self, user: User, action: BaseObjectAction) -> Q | None:
        q = super().construct_perm_q(user, action)
        if q is None or action not in ('change', 'delete'):
            return q
        return q & Q(is_editable=True)

    def user_has_perm(self, user: User, action: ObjectSpecificAction, obj: NodeConfig) -> bool:
        if action in ('change', 'delete') and not obj.is_editable and not user.is_superuser:
            return False
        return super().user_has_perm(user, action, obj)


class NodeConfig(PathsModel[InstanceConfig], EditableInstanceChild, index.Indexed):
    instance: FK[InstanceConfig] = models.ForeignKey(
        InstanceConfig,
        on_delete=models.CASCADE,
        related_name='nodes',
        editable=False,
    )
    identifier = IdentifierField(max_length=200)
    is_stale = models.BooleanField(default=False, help_text='Whether the node is stale and should be deleted')
    name = models.CharField(max_length=200, null=True, blank=True)
    short_name = models.CharField(max_length=200, null=True, blank=True)
    order = models.IntegerField(
        null=True,
        blank=True,
        verbose_name=_('Order'),
    )
    is_visible = models.BooleanField(default=True)
    is_editable = models.BooleanField(
        default=True,
        help_text=_('Whether non-superusers may modify this node and its inputs'),
    )
    goal = RichTextField[str | None, str | None](
        null=True,
        blank=True,
        verbose_name=_('Goal'),
        editor='very-limited',
        max_length=1000,
    )
    short_description = RichTextField[str | None, str | None](
        null=True,
        blank=True,
        verbose_name=_('Short description'),
        editor='limited',
    )
    description = RichTextField[str | None, str | None](
        null=True,
        blank=True,
        verbose_name=_('Description'),
    )  # -> StreamField
    body = StreamField(
        [
            ('card_list', CardListBlock()),
            ('paragraph', blocks.RichTextBlock()),
        ],
        use_json_field=True,
        blank=True,
    )

    indicator_node: FK[NodeConfig | None] = models.ForeignKey(
        'self',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='indicates_nodes',
    )

    copy_of: FK[NodeConfig | None] = models.ForeignKey(
        'self',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='copies',
        editable=False,
        help_text=_('The node this one was copied from, if any.'),
    )
    copy_of_id: int | None

    datasets: M2M[DatasetModel, NodeDataset] = models.ManyToManyField(DatasetModel, through='NodeDataset', related_name='nodes')

    color: CharField[str, str] = ColorField(max_length=20, blank=True)
    input_data = models.JSONField(null=True, editable=False)
    params = models.JSONField(null=True, editable=False)

    spec = SchemaField(schema=NodeSpec, null=True, blank=True)

    # Audit timestamps (``created_at`` / ``last_modified_at``) + user FKs
    # come from ``UserModifiableModel`` via ``EditableInstanceChild``.

    i18n = TranslationField(
        fields=('name', 'short_name', 'short_description', 'description', 'goal'),
        default_language_field='instance__primary_language',
    )
    name_i18n: str | None
    short_name_i18n: str | None
    short_description_i18n: str | None
    description_i18n: str | None
    goal_i18n: str | None
    indicates_nodes: RevMany[NodeConfig]
    layout: RevOne[NodeConfig, NodeLayout]
    incoming_edges: RevMany[NodeEdge]
    outgoing_edges: RevMany[NodeEdge]
    input_bindings: RevMany[NodeInputPortBinding]
    output_bindings: RevMany[NodeInputPortBinding]

    search_fields = [
        index.AutocompleteField('identifier'),
        index.AutocompleteField('name'),
        index.SearchField('name'),
        index.SearchField('identifier'),
        index.FilterField('instance'),
    ]
    search_auto_update = False
    wagtail_reference_index_ignore = True

    objects: ClassVar[NodeConfigManager] = NodeConfigManager()

    snapshot_model: ClassVar[type[ModelSnapshot]] = NodeSnapshot

    _node: Node | None
    _annotated_port_edge_bindings: list[dict[str, Any]] | None
    _annotated_port_dataset_bindings: list[dict[str, Any]] | None
    _annotated_dataset_models_by_uuid: dict[UUID, DatasetModel]

    class Meta:
        verbose_name = _('Node')
        verbose_name_plural = _('Nodes')
        unique_together = (('instance', 'identifier'),)
        ordering = ['instance', 'order', 'pk']
        base_manager_name = 'objects'

    @classmethod
    def permission_policy(cls) -> NodeConfigPermissionPolicy:
        return NodeConfigPermissionPolicy()

    def get_node(self, visible_for_user: UserOrAnon | None = None) -> Node | None:
        if hasattr(self, '_node'):
            return self._node

        instance = self.instance.get_instance()
        # FIXME: Node visibility restrictions
        node = instance.context.nodes.get(self.identifier)
        self._node = node
        return node

    def update_node_from_config(self, node: Node, keep_ref: bool = False):
        node.database_id = self.pk
        if keep_ref:
            node.db_obj = self
        if self.order is not None:
            node.order = self.order

        node._spec = self.spec

        if self.input_data:
            assert len(node.input_dataset_instances) == 1
            # disable legacy input data stuff
            # node.replace_input_data(self.input_data)

        # FIXME: Override params

    def update_from_node(self, node: Node, overwrite=False, skip_descriptions=False, update_relations=True):
        """Set attributes of this instance from revelant fields of the given node but does not save."""

        overwritten = False

        conf = node.as_node_config_attributes()
        i18n = conf.pop('i18n', None)
        for k, v in conf.items():
            if overwrite or getattr(self, k, None) is None:
                if skip_descriptions and k in ['short_description', 'description']:
                    continue
                setattr(self, k, v)
                overwritten = True

        if i18n is not None:
            if not self.i18n:
                self.i18n = {}
            assert isinstance(self.i18n, dict)
            translated = cast('dict[str, str]', i18n)
            if overwrite:
                self.i18n |= translated
            else:
                for key, value in translated.items():
                    self.i18n.setdefault(key, value)

        if overwritten:
            self.instance.log.info('Overwrote contents in node %s' % str(node))

        if self.pk and update_relations:
            self.update_relations_from_node(node)

    def update_relations_from_node(self, node: Node):
        from nodes.datasets import DBDataset

        current_dss = {ds.pk for ds in self.datasets.all()}
        for dataset in node.input_dataset_instances:
            if not isinstance(dataset, DBDataset):
                continue
            obj = dataset.db_dataset_obj
            if obj is None:
                continue
            if obj.pk not in current_dss:
                self.instance.log.info('Adding dataset %s to node %s' % (obj.identifier, self))
                self.datasets.add(obj)
            else:
                current_dss.remove(obj.pk)

        # The remaining current_dss are the ones to delete
        self.datasets.remove(*current_dss)

    def can_edit_data(self):
        node = self.get_node()
        if node is None:
            return False
        if len(node.input_dataset_instances) != 1:
            return False
        return True

    def __str__(self) -> str:
        return self.name or '<no name>'

    def __rich_repr__(self):
        yield self.name
        yield 'pk', self.pk
        yield 'identifier', self.identifier
        yield 'instance', self.instance.identifier

    def get_name_with_icon(self):
        node = self.get_node()
        prefix = ''
        if node is None:
            prefix = '⚠️ '
            name = ''
        else:
            icon = node.get_icon()
            if icon is not None:
                prefix = f'{icon} '
            name = str(node.name)

        if self.name:
            name = self.name
        return f'{prefix}{name}'

    @copy_signature(models.Model.save)
    def save(self, **kwargs) -> None:
        if self.i18n:
            for key in self.i18n.keys():
                regex_match = re.search(r'_([a-z]{2}([_-][a-z]{2})?$)', key, re.IGNORECASE)
                if regex_match is None:
                    error_message = f'No language code found in i18n key "{key}".'
                    raise RuntimeError(error_message)
                lang = regex_match.group(1)
                if lang != convert_language_code(lang, 'modeltrans'):
                    error_message = f'Language code "{lang}" in i18n key "{key}" is not in "modeltrans" format.'
                    raise RuntimeError(error_message)

        if not isinstance(self.uuid, uuid.UUID):
            self.uuid = uuid.uuid4()

        return super().save(**kwargs)

    def natural_key(self):
        return self.instance.natural_key() + (self.identifier,)

    @property
    def port_edge_bindings(self) -> list[EdgeBindingDef]:
        if not hasattr(self, '_annotated_port_edge_bindings'):
            raise RuntimeError('NodeConfig.port_edge_bindings requires NodeConfigQuerySet.annotate_ports()')
        raw = self._annotated_port_edge_bindings or []
        return [EdgeBindingDef.model_validate(port) for port in raw]

    @property
    def port_dataset_bindings(self) -> list[DatasetBindingDef]:
        if not hasattr(self, '_annotated_port_dataset_bindings'):
            raise RuntimeError('NodeConfig.port_dataset_bindings requires NodeConfigQuerySet.annotate_ports()')
        raw = self._annotated_port_dataset_bindings or []
        return [DatasetBindingDef.model_validate(port) for port in raw]


class NodeLayoutSource(models.TextChoices):
    AUTO = 'auto', _('Auto')
    USER = 'user', _('User')


class NodeLayout(UserModifiableModel):
    """Shared model-editor position for one node card."""

    node: OneToOne[NodeConfig] = models.OneToOneField(
        NodeConfig,
        on_delete=models.CASCADE,
        related_name='layout',
    )
    x = models.FloatField()
    y = models.FloatField()
    source = TextChoicesField(
        choices_enum=NodeLayoutSource,  # pyright: ignore[reportCallIssue]
        default=NodeLayoutSource.AUTO,
    )

    class Meta:
        verbose_name = _('Node layout')
        verbose_name_plural = _('Node layouts')
        ordering = ['node']

    def __str__(self) -> str:
        return f'{self.node.identifier}: ({self.x}, {self.y})'


class NodeDataset(models.Model):
    node = models.ForeignKey(NodeConfig, on_delete=models.CASCADE, related_name='datasets_edges')
    dataset = models.ForeignKey(DatasetModel, on_delete=models.PROTECT, related_name='nodes_edges')

    class Meta:
        verbose_name = _('Node dataset')
        verbose_name_plural = _('Node datasets')
        unique_together = (('node', 'dataset'),)
        ordering = ['node', 'dataset']

    def __str__(self) -> str:
        node_name = self.node.name or self.node.identifier
        return _('Node: %(node_name)s') % {'node_name': node_name}

    def get_admin_display_title(self) -> str:
        """Return a descriptive title for Wagtail admin views."""
        return str(self)


# --- Model editor models ---


class NodeEdge(EditableInstanceChild):
    """A directed edge in the computation graph."""

    snapshot_model: ClassVar[type[ModelSnapshot]] = EdgeSnapshot

    instance: FK[InstanceConfig] = models.ForeignKey(
        InstanceConfig,
        on_delete=models.CASCADE,
        related_name='edges',
    )
    from_node: FK[NodeConfig] = models.ForeignKey(
        NodeConfig,
        on_delete=models.CASCADE,
        related_name='outgoing_edges',
    )
    from_port = models.UUIDField[UUID, UUID](
        max_length=200,
        default='output',
        help_text='Output port ID on the source node',
    )
    to_node: FK[NodeConfig] = models.ForeignKey(
        NodeConfig,
        on_delete=models.CASCADE,
        related_name='incoming_edges',
    )
    to_port = models.UUIDField[UUID, UUID](
        max_length=200,
        help_text='Input port ID on the target node',
    )
    transformations = SchemaField(schema=list[EdgeTransformOp], default=list, blank=True)
    tags = ArrayField(
        models.CharField(max_length=200),
        default=list,
        blank=True,
    )

    from_node_id: int  # for type checkers
    to_node_id: int

    class Meta:
        ordering = ['instance', 'from_node_id', 'to_node_id', 'to_port']
        verbose_name = _('Node edge')
        verbose_name_plural = _('Node edges')

    def __str__(self) -> str:
        return f'{self.from_node_id} → {self.to_node_id}'


class DatasetPort(EditableInstanceChild):
    """Connects a dataset metric to a node input port."""

    snapshot_model: ClassVar[type[ModelSnapshot]] = DatasetPortSnapshot

    instance: FK[InstanceConfig] = models.ForeignKey(
        InstanceConfig,
        on_delete=models.CASCADE,
        related_name='dataset_ports',
    )
    node: FK[NodeConfig] = models.ForeignKey(
        NodeConfig,
        on_delete=models.CASCADE,
        related_name='dataset_ports',
    )
    port_id = models.UUIDField[UUID, UUID](
        max_length=100,
        help_text='Input port ID on the node (must match a port in node.input_ports)',
    )
    dataset: FK[DatasetModel] = models.ForeignKey(
        DatasetModel,
        on_delete=models.PROTECT,
        related_name='node_ports',
    )
    metric: FK[DatasetMetric] = models.ForeignKey(
        DatasetMetric,
        on_delete=models.PROTECT,
        related_name='node_ports',
    )
    spec = SchemaField(schema=DatasetPortSpec, default=DatasetPortSpec, blank=True)
    dataset_index = models.PositiveIntegerField(
        default=0,
        help_text=(
            "Index of this binding in the owning node's input_dataset_instances list. "
            'Multiple DatasetPort rows can share a dataset_index when a column-less '
            'binding expands to one port per output metric.'
        ),
    )

    # for type checkers
    node_id: int
    dataset_id: int
    metric_id: int

    class Meta:
        ordering = ['node', 'dataset_index', 'metric__order']
        verbose_name = _('Dataset port')
        verbose_name_plural = _('Dataset ports')

    def __str__(self) -> str:
        return f'{self.node_id}:{self.port_id} ← {self.dataset_id}'


class NodeInputPortBinding(EditableInstanceChild):
    """
    One value delivered to a node input port — edge- or dataset-sourced.

    The authoritative unified store replacing ``NodeEdge`` and ``DatasetPort``
    (see docs/architecture/dimension-constraints.md, "One input-binding
    table"; the legacy tables are empty and await removal in plan step 11).
    ``position`` orders bindings within one port across both source kinds,
    which matters because a ``multi`` port may hold both and floating-point
    addition makes delivery order observable. Snapshot-driven writers go
    through ``nodes.input_bindings.reconcile_input_bindings()``; row-level
    editors write directly and keep positions dense.
    """

    snapshot_model: ClassVar[type[ModelSnapshot]] = InputBindingSnapshot

    instance: FK[InstanceConfig] = models.ForeignKey(
        InstanceConfig,
        on_delete=models.CASCADE,
        related_name='input_bindings',
    )
    node: FK[NodeConfig] = models.ForeignKey(
        NodeConfig,
        on_delete=models.CASCADE,
        related_name='input_bindings',
    )
    port_id = models.UUIDField[UUID, UUID](
        help_text='Input port ID on the node (must match a port in node.input_ports)',
    )
    position = models.PositiveIntegerField(
        default=0,
        help_text='Stable order among values delivered to the input port, shared across source kinds.',
    )

    # Exactly one source branch is populated.
    source_node: FK[NodeConfig | None] = models.ForeignKey(
        NodeConfig,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name='output_bindings',
    )
    source_port_id = models.UUIDField[UUID | None, UUID | None](
        null=True,
        blank=True,
        help_text='Output port ID on the source node',
    )
    dataset: FK[DatasetModel | None] = models.ForeignKey(
        DatasetModel,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name='node_input_bindings',
    )
    metric: FK[DatasetMetric | None] = models.ForeignKey(
        DatasetMetric,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name='node_input_bindings',
    )

    transformations = SchemaField(schema=list[StoredPortTransformOp], default=list, blank=True)
    tags = ArrayField(
        models.CharField(max_length=200),
        default=list,
        blank=True,
    )

    # Transitional dataset-branch state (empty/zero on edge bindings). The
    # runtime still consumes DatasetPortSpec whole and groups fanned-out
    # per-metric rows by (node, dataset_index); each spec field's target home
    # is the transform pipeline, and both fields go away in plan step 11 once
    # dataset loading executes the pipeline directly.
    dataset_spec = SchemaField(schema=DatasetPortSpec, default=DatasetPortSpec, blank=True)
    dataset_index = models.PositiveIntegerField(
        default=0,
        help_text=(
            "Order of the binding group in the owning node's input_dataset_instances list. "
            'Rows sharing (node, dataset_index) belong to one column-less binding expanded '
            'to a port per metric.'
        ),
    )

    # for type checkers
    node_id: int
    source_node_id: int | None
    dataset_id: int | None
    metric_id: int | None

    class Meta:
        ordering = ['node', 'port_id', 'position']
        verbose_name = _('Node input binding')
        verbose_name_plural = _('Node input bindings')
        constraints = (
            models.CheckConstraint(
                condition=(
                    Q(
                        source_node__isnull=False,
                        source_port_id__isnull=False,
                        dataset__isnull=True,
                        metric__isnull=True,
                    )
                    | Q(
                        source_node__isnull=True,
                        source_port_id__isnull=True,
                        dataset__isnull=False,
                        metric__isnull=False,
                    )
                ),
                name='node_input_binding_has_one_source',
            ),
            models.UniqueConstraint(
                fields=('node', 'port_id', 'position'),
                name='node_input_binding_position_is_unique',
                deferrable=models.Deferrable.DEFERRED,
            ),
        )

    def __str__(self) -> str:
        if self.source_node_id is not None:
            return f'{self.node_id}:{self.port_id}[{self.position}] ← node {self.source_node_id}'
        return f'{self.node_id}:{self.port_id}[{self.position}] ← dataset {self.dataset_id}'


class DatasetMaterialization(models.Model):
    """Current serialized calculation payload for a DB-backed dataset."""

    dataset: OneToOne[DatasetModel] = models.OneToOneField(
        DatasetModel,
        on_delete=models.CASCADE,
        related_name='paths_materialization',
    )
    content = models.JSONField()
    content_hash = models.CharField(max_length=64)
    generation = models.PositiveBigIntegerField(default=1)
    shape_profiles = models.JSONField(null=True)
    validation_violations = models.JSONField(default=list, blank=True)
    """Current violations of the dataset's metric validation rules (see ``datasets.validation``)."""
    forecast_from = models.IntegerField(null=True, blank=True)
    source_modified_at = models.DateTimeField()
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['dataset_id']
        verbose_name = _('Dataset materialization')
        verbose_name_plural = _('Dataset materializations')

    def __str__(self) -> str:
        return f'{self.dataset_id} @ {self.generation}'


class InstanceRevisionDatasetPin(models.Model):
    """Relational retention manifest for datasets used by an instance revision."""

    instance_config: FK[InstanceConfig] = models.ForeignKey(
        InstanceConfig,
        on_delete=models.CASCADE,
        related_name='dataset_revision_pins',
    )
    instance_revision = models.ForeignKey(
        'wagtailcore.Revision',
        on_delete=models.CASCADE,
        related_name='instance_dataset_pins',
    )
    dataset: FK[DatasetModel] = models.ForeignKey(
        DatasetModel,
        on_delete=models.PROTECT,
        related_name='instance_revision_pins',
    )
    dataset_revision = models.ForeignKey(
        'wagtailcore.Revision',
        on_delete=models.PROTECT,
        related_name='dataset_revision_pins',
    )
    dataset_uuid = models.UUIDField()
    identifier = models.CharField(max_length=100, null=True, blank=True)
    forecast_from = models.IntegerField(null=True, blank=True)
    shape_profiles = models.JSONField(null=True)

    class Meta:
        ordering = ['instance_revision_id', 'dataset_id']
        constraints = [
            models.UniqueConstraint(
                fields=['instance_revision', 'dataset'],
                name='unique_dataset_pin_per_instance_revision',
            ),
        ]
        indexes = [
            models.Index(fields=['instance_config', 'instance_revision']),
            models.Index(fields=['dataset_revision']),
            models.Index(fields=['dataset']),
        ]
        verbose_name = _('Instance revision dataset pin')
        verbose_name_plural = _('Instance revision dataset pins')

    def __str__(self) -> str:
        return f'{self.instance_revision_id}: {self.identifier or self.dataset_id} → {self.dataset_revision_id}'


# --- Change tracking: InstanceChangeOperation + InstanceModelLogEntry ---
#
# Every user-facing edit to an InstanceConfig's model opens exactly one
# InstanceChangeOperation. All resulting row-level writes emit
# InstanceModelLogEntry rows linked to that operation. This is the audit +
# undo substrate; actual mutations are wired through
# ``nodes/change_ops.py::change_operation``.


class InstanceChangeSource(models.TextChoices):
    GRAPHQL = 'graphql', _('GraphQL')
    REST = 'rest', _('REST')
    ADMIN = 'admin', _('Wagtail admin')
    CLI = 'cli', _('CLI')
    MIGRATION = 'migration', _('Data migration')


class InstanceChangeOperation(UUIDIdentifiedModel):
    """
    One row per user-facing edit (create / update / delete / cascade bundle).

    Serves as:
      * grouping anchor for ``InstanceModelLogEntry`` rows
      * audit of who/when/where an edit came from
      * unit of undo (undo targets the operation, not individual entries)
      * undo trail via ``superseded_by``
    """

    instance_config: FK[InstanceConfig] = models.ForeignKey(
        InstanceConfig,
        on_delete=models.CASCADE,
        related_name='change_operations',
    )
    user: FK[User | None] = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='+',
    )
    user_id: int | None
    # In-memory only: model classes recorded through ``record_change`` during
    # this operation, so the write-boundary hook knows whether the
    # input-binding mirror may be affected.
    _touched_models: set[type[models.Model]]
    action = models.CharField(
        max_length=100,
        help_text="Top-level action that triggered the operation, e.g. 'node.delete'.",
    )
    source = models.CharField(
        max_length=20,
        choices=InstanceChangeSource.choices,
        default=InstanceChangeSource.GRAPHQL,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    superseded_by: FK[InstanceChangeOperation | None] = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='supersedes',
        help_text='Set when this operation has been undone by another operation.',
    )

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['instance_config', '-created_at']),
        ]
        verbose_name = _('Instance change operation')
        verbose_name_plural = _('Instance change operations')

    def __str__(self) -> str:
        return f'{self.action} @ {self.created_at:%Y-%m-%d %H:%M:%S} ({self.uuid})'


class InstanceModelLogEntry(UUIDIdentifiedModel):
    """
    One row per row-level write within an ``InstanceChangeOperation``.

    Deliberately standalone (not a subclass of Wagtail's ``ModelLogEntry``)
    to avoid the multi-table-inheritance write overhead and the
    ``LogActionRegistry`` indirection. Shape mirrors ``ModelLogEntry``
    where it makes sense (``content_type`` / ``object_id`` as GFK;
    ``action`` string; ``data`` JSON), but user/timestamp metadata lives
    on the parent ``operation`` to avoid duplication.

    ``target_uuid`` is the durable identity of the affected object. Unlike
    ``object_id``, it remains meaningful after the target row is deleted.

    ``data`` layout::

        {
            'target_uuid': str,         # legacy payload copy; remove after old readers retire
            'before': dict | None,      # None for creates
            'after':  dict | None,      # None for deletes
        }
    """

    operation: FK[InstanceChangeOperation] = models.ForeignKey(
        InstanceChangeOperation,
        on_delete=models.CASCADE,
        related_name='log_entries',
    )
    content_type: FK[ContentType | None] = models.ForeignKey(
        ContentType,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='+',
        help_text='Type of the affected row; GFK with object_id.',
    )
    object_id = models.CharField(max_length=255, null=True, blank=True)
    target_uuid = models.UUIDField(
        null=True,
        blank=True,
        db_index=True,
        help_text='Stable UUID of the affected object; retained after the target row is deleted.',
    )
    action = models.CharField(
        max_length=100,
        help_text="Dotted action id, e.g. 'node.update'.",
    )
    data = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-id']
        indexes = [
            models.Index(fields=['operation']),
            models.Index(fields=['content_type', 'object_id']),
        ]
        verbose_name = _('Instance model log entry')
        verbose_name_plural = _('Instance model log entries')

    def __str__(self) -> str:
        return f'{self.action} on {self.content_type}:{self.object_id}'


INVITATION_TTL_DAYS = 30


def _generate_invitation_token() -> str:
    return secrets.token_urlsafe(32)


def _default_invitation_expiry() -> datetime:
    return timezone.now() + timedelta(days=INVITATION_TTL_DAYS)


class InstanceInvitationPermissionPolicy(
    ParentInheritedPolicy['InstanceInvitation', InstanceConfig, 'InstanceInvitationQuerySet']
):
    def __init__(self):
        super().__init__(InstanceInvitation, InstanceConfig, 'instance_config')


class InstanceInvitationQuerySet(PermissionedQuerySet['InstanceInvitation']):
    def active(self) -> Self:
        return self.filter(is_soft_deleted=False, accepted_at__isnull=True, expires_at__gt=timezone.now())


class InstanceInvitationManager(PermissionedManager['InstanceInvitation']):
    """Default manager hides soft-deleted invitations."""

    def get_queryset(self) -> InstanceInvitationQuerySet:
        return InstanceInvitationQuerySet(self.model, using=self._db).exclude(is_soft_deleted=True)


class InstanceInvitationManagerIncludingDeleted(PermissionedManager['InstanceInvitation']):
    def get_queryset(self) -> InstanceInvitationQuerySet:
        return InstanceInvitationQuerySet(self.model, using=self._db)


class InstanceInvitation(UserModifiableModel, PermissionedModel):
    """
    An invitation extended to an email address to join an :class:`InstanceConfig`.

    On acceptance via the ``registerUser`` mutation, the invited email is
    promoted to a real ``User`` and granted the instance admin role.
    Soft-deletion is used for revocation so the audit trail survives.
    """

    instance_config: FK[InstanceConfig] = models.ForeignKey(
        InstanceConfig,
        on_delete=models.CASCADE,
        related_name='invitations',
    )
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    email = models.EmailField()
    token = models.CharField(max_length=64, unique=True, default=_generate_invitation_token, editable=False)
    expires_at = models.DateTimeField(default=_default_invitation_expiry)
    accepted_at = models.DateTimeField(null=True, blank=True, editable=False)
    accepted_by: FK[User | None] = models.ForeignKey(
        'users.User',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='accepted_invitations',
        editable=False,
    )

    is_soft_deleted = models.BooleanField(default=False)
    soft_deleted_at = models.DateTimeField(null=True, blank=True, editable=False)
    soft_deleted_by: FK[User | None] = models.ForeignKey(
        'users.User',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='soft_deleted_invitations',
        editable=False,
    )

    objects: ClassVar[InstanceInvitationManager] = InstanceInvitationManager()
    objects_including_soft_deleted: ClassVar[InstanceInvitationManagerIncludingDeleted] = (
        InstanceInvitationManagerIncludingDeleted()
    )

    class Meta:
        verbose_name = _('Instance invitation')
        verbose_name_plural = _('Instance invitations')
        ordering = ['-created_at']
        constraints = [
            models.UniqueConstraint(
                fields=['instance_config', 'email'],
                condition=Q(is_soft_deleted=False, accepted_at__isnull=True),
                name='unique_active_invitation_per_instance_email',
            ),
        ]

    def __str__(self) -> str:
        return f'Invitation for {self.email} to {self.instance_config.identifier}'

    def save(self, *args, **kwargs):
        self.email = self.email.lower()
        super().save(*args, **kwargs)

    @classmethod
    def permission_policy(cls) -> InstanceInvitationPermissionPolicy:
        return InstanceInvitationPermissionPolicy()

    def is_valid(self) -> bool:
        return not self.is_soft_deleted and self.accepted_at is None and self.expires_at > timezone.now()

    def soft_delete(self, user: User | None) -> None:
        self.is_soft_deleted = True
        self.soft_deleted_at = timezone.now()
        self.soft_deleted_by = user
        self.save(update_fields=['is_soft_deleted', 'soft_deleted_at', 'soft_deleted_by', 'last_modified_at', 'last_modified_by'])

    def mark_accepted(self, user: User) -> None:
        self.accepted_at = timezone.now()
        self.accepted_by = user
        self.save(update_fields=['accepted_at', 'accepted_by', 'last_modified_at', 'last_modified_by'])
