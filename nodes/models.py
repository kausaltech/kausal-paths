from __future__ import annotations

import re
import threading
import uuid
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self, TypedDict, cast
from urllib.parse import urlparse

from django.conf import settings
from django.contrib.auth.models import Group
from django.contrib.contenttypes.fields import GenericRelation
from django.contrib.contenttypes.models import ContentType
from django.contrib.postgres.fields import ArrayField
from django.core.exceptions import ObjectDoesNotExist
from django.db import models, transaction
from django.db.models import CharField, Q
from django.utils import timezone
from django.utils.translation import get_language, gettext, gettext_lazy as _, override
from modelcluster.models import ClusterableModel
from modeltrans.fields import TranslationField
from modeltrans.manager import MultilingualQuerySet
from wagtail import blocks
from wagtail.fields import RichTextField, StreamField
from wagtail.models import Locale, Page, RevisionMixin
from wagtail.models.sites import Site
from wagtail.search import index

import sentry_sdk
from asgiref.sync import sync_to_async
from channels.consumer import async_to_sync
from channels.layers import get_channel_layer
from loguru import logger
from wagtail_color_panel.fields import ColorField

from kausal_common.datasets.models import (
    Dataset as DatasetModel,
    Dimension as DatasetDimensionModel,
    DimensionCategory,
    DimensionScope,
)
from kausal_common.i18n.helpers import convert_language_code
from kausal_common.models.permission_policy import ModelPermissionPolicy, ParentInheritedPolicy
from kausal_common.models.permissions import PermissionedQuerySet
from kausal_common.models.types import FK, M2M, MLModelManager, RevMany, RevOne, copy_signature
from kausal_common.models.uuid import UUIDIdentifiedModel

from paths.const import INSTANCE_CHANGE_GROUP, INSTANCE_CHANGE_TYPE
from paths.types import CacheablePathsModel, PathsModel, PathsQuerySet
from paths.utils import (
    ChoiceArrayField,
    IdentifierField,
    InstanceIdentifierValidator,
    get_default_language,
    get_supported_languages,
)

from common.i18n import get_modeltrans_attrs_from_str
from orgs.models import Organization
from pages.blocks import CardListBlock

if TYPE_CHECKING:
    from django.http import HttpRequest

    from loguru import Logger

    from kausal_common.models.permission_policy import BaseObjectAction, ObjectSpecificAction
    from kausal_common.users import UserOrAnon

    from frameworks.models import FrameworkConfig
    from nodes.dimensions import Dimension as NodeDimension
    from nodes.node import Node
    from pages.config import OutcomePage as OutcomePageConfig
    from pages.models import ActionListPage, InstanceSiteContent
    from users.models import User

    from .instance import Instance


instance_cache_lock = threading.Lock()


def get_instance_identifier_from_wildcard_domain(
    hostname: str, request: HttpRequest | None = None, wildcard_domains: list[str] | None = None,
) -> tuple[str, str] | tuple[None, None]:
    # Get instance identifier from hostname for development and testing
    parts = hostname.lower().split('.', maxsplit=1)
    if wildcard_domains is None:
        if request is not None:
            req_wildcards: list[str] = getattr(request, 'wildcard_domains', None) or []
        else:
            req_wildcards = []
        settings_wildcards: list[str] = cast('list[str]', settings.HOSTNAME_INSTANCE_DOMAINS) or []
        wd_domains = [*settings_wildcards, *req_wildcards]
    else:
        wd_domains = wildcard_domains
    if len(parts) == 2 and parts[1].lower() in wd_domains:
        return (parts[0], parts[1])
    return (None, None)


class InstanceConfigQuerySet(MultilingualQuerySet['InstanceConfig'], PermissionedQuerySet['InstanceConfig']):  # type: ignore[override]
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

_InstanceConfigManager = models.Manager.from_queryset(InstanceConfigQuerySet)

class InstanceConfigManager(MLModelManager['InstanceConfig', InstanceConfigQuerySet], _InstanceConfigManager):  # pyright: ignore[reportIncompatibleMethodOverride]
    def get_by_natural_key(self, identifier: str) -> InstanceConfig:
        return self.get(identifier=identifier)
del _InstanceConfigManager


class InstanceConfigPermissionPolicy(ModelPermissionPolicy['InstanceConfig', Any, InstanceConfigQuerySet]):
    def __init__(self):
        from frameworks.roles import framework_admin_role, framework_viewer_role

        from .roles import instance_admin_role, instance_viewer_role

        self.admin_role = instance_admin_role
        self.viewer_role = instance_viewer_role
        self.fw_admin_role = framework_admin_role
        self.fw_viewer_role = framework_viewer_role
        super().__init__(InstanceConfig)

    def is_admin(self, user: User, obj: InstanceConfig) -> bool:
        return user.has_instance_role(self.admin_role, obj)

    def is_viewer(self, user: User, obj: InstanceConfig) -> bool:
        return user.has_instance_role(self.viewer_role, obj)

    def is_framework_admin(self, user: User, obj: InstanceConfig) -> bool:
        if not obj.has_framework_config():
            return False
        return user.has_instance_role(self.fw_admin_role, obj.framework_config.framework)

    def is_framework_viewer(self, user: User, obj: InstanceConfig) -> bool:
        if not obj.has_framework_config():
            return False
        return user.has_instance_role(self.fw_viewer_role, obj.framework_config.framework)

    def construct_perm_q(self, user: User, action: ObjectSpecificAction, include_implicit_public: bool = True) -> models.Q | None:
        is_admin = self.admin_role.role_q(user)
        is_viewer = self.viewer_role.role_q(user)
        is_fw_admin = self.fw_admin_role.role_q(user, prefix='framework_config__framework')
        is_fw_viewer = self.fw_viewer_role.role_q(user, prefix='framework_config__framework')
        if action == 'view':
            q = is_viewer | is_admin | is_fw_admin | is_fw_viewer
            if include_implicit_public:
                q |= Q(framework_config__isnull=True)
            return q
        return is_admin | is_fw_admin

    def construct_perm_q_anon(self, action: BaseObjectAction) -> Q | None:
        if action == 'view':
            # If it's a framework-based config, require authentication for viewing
            return Q(framework_config__isnull=True)
        return None

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
        if action == 'delete':
            return self.is_framework_admin(user, obj)
        if action == 'view':
            if self.anon_has_perm('view', obj):
                return True
            if self.is_viewer(user, obj):
                return True
            if self.is_framework_viewer(user, obj):
                return True
        return self.is_admin(user, obj) or self.is_framework_admin(user, obj)

    def anon_has_perm(self, action: ObjectSpecificAction, obj: InstanceConfig) -> bool:
        if action != 'view':
            return False
        if not obj.has_framework_config():
            # FIXME: Add checking for a "published" status here
            return True
        return False

    def user_can_create(self, user: User, context: Any) -> bool:
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


class InstanceConfig(CacheablePathsModel[None], UUIDIdentifiedModel, models.Model):  # , RevisionMixin)
    """Metadata for one Paths computational model instance."""

    identifier = IdentifierField(max_length=100, unique=True, validators=[InstanceIdentifierValidator()])
    name = models.CharField(max_length=150, verbose_name=_('name'))
    lead_title = models.CharField(blank=True, max_length=100, verbose_name=_('Lead title'))
    lead_title_i18n: str
    lead_paragraph = RichTextField[str | None, str | None](null=True, blank=True, verbose_name=_('Lead paragraph'))
    lead_paragraph_i18n: str | None
    site_url = models.URLField(verbose_name=_('Site URL'), null=True)
    site = models.OneToOneField(Site, null=True, on_delete=models.PROTECT, editable=False, related_name='instance')
    organization: FK[Organization] = models.ForeignKey(
        Organization, related_name='instances', on_delete=models.PROTECT, verbose_name=_('organization'),
        help_text=_('The main organization for the instance'),
    )

    is_protected = models.BooleanField(default=False)
    protection_password = models.CharField(max_length=50, null=True, blank=True)

    created_at = models.DateTimeField(default=timezone.now)
    modified_at = models.DateTimeField(auto_now=True)
    cache_invalidated_at = models.DateTimeField(default=timezone.now)

    primary_language = models.CharField[str, str](max_length=8, choices=get_supported_languages, default=get_default_language)  # type: ignore[arg-type]
    other_languages = ChoiceArrayField(
        models.CharField(max_length=8, choices=get_supported_languages, default=get_default_language),  # type: ignore[arg-type]
        default=list,
    )

    viewer_group: FK[Group | None] = models.ForeignKey(
        Group, on_delete=models.PROTECT, editable=False, related_name='viewer_instances',
        null=True,
    )
    viewer_group_id: int | None
    admin_group: FK[Group | None] = models.ForeignKey(
        Group, on_delete=models.PROTECT, editable=False, related_name='admin_instances',
        null=True,
    )
    admin_group_id: int | None

    """
    model_cache = JSONField[InstanceModelCache | None, InstanceModelCache | None](
        verbose_name='cached model data', null=True, blank=True,
    )
    """
    """Used to store data to speed up model runs"""

    i18n = TranslationField(fields=('name', 'lead_title', 'lead_paragraph'))

    objects: ClassVar[InstanceConfigManager] = InstanceConfigManager()

    dataset_schema_scopes = GenericRelation(
        'datasets.DatasetSchemaScope', related_query_name='instance_config',
        content_type_field='scope_content_type', object_id_field='scope_id',
    )

    # Type annotations
    nodes: RevMany[NodeConfig]
    hostnames: RevMany[InstanceHostname]
    dimensions: RevMany[DatasetDimensionModel]
    datasets: RevMany[DatasetModel]
    framework_config: RevOne[InstanceConfig, FrameworkConfig]
    framework_config_id: int | None
    organization_id: int
    site_content: RevOne[InstanceConfig, InstanceSiteContent]

    search_fields = [
        index.SearchField('identifier'),
        index.SearchField('name_i18n'),
    ]

    class Meta:  # pyright: ignore
        verbose_name = _('Instance')
        verbose_name_plural = _('Instances')

    def __str__(self) -> str:
        return self.get_name()

    def __rich_repr__(self):
        yield self.identifier
        yield 'id', self.pk
        yield 'name', self.name

    def save(self, *args, **kwargs):
        if self.uuid is None:
            self.uuid = uuid.uuid4()

        if self.site is not None:
            # TODO: Update Site and root page attributes
            pass

        super().save(*args, **kwargs)

    @transaction.atomic
    @copy_signature(models.Model.delete)
    def delete(self, **kwargs):
        site = self.site
        if site is not None:
            rp = site.root_page
            self.site = None
            self.save()
            site.delete()
            rp.get_descendants(inclusive=True).delete()

        pp = self.permission_policy()
        pp.admin_role.delete_instance_group(self)
        pp.viewer_role.delete_instance_group(self)
        self.nodes.all().delete()
        super().delete(**kwargs)

    def natural_key(self):
        return (self.identifier,)

    @classmethod
    def permission_policy(cls) -> InstanceConfigPermissionPolicy:
        return InstanceConfigPermissionPolicy()

    @classmethod
    def create_for_instance(cls, instance: Instance, **kwargs) -> InstanceConfig:
        assert not cls.objects.filter(identifier=instance.id).exists()

        org = Organization.objects.get(name="Kausal") # TODO: Define the organization better when we have a better idea?
        return cls.objects.create(identifier=instance.id, site_url=instance.site_url, organization=org, **kwargs)

    def has_framework_config(self) -> bool:
        try:
            _ = self.framework_config
        except ObjectDoesNotExist:
            return False
        else:
            return True

    def update_instance_from_configs(self, instance: Instance, node_refs: bool = False):
        for node_config in self.nodes.all():
            node = instance.context.nodes.get(node_config.identifier)
            if node is None:
                continue
            node_config.update_node_from_config(node, keep_ref=node_refs)

    def update_from_instance(self, instance: Instance, overwrite=False):
        """Update lead_title and lead_paragraph from instance but do not call save()."""

        for field_name in ('lead_title', 'lead_paragraph', 'name'):
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

    def _create_from_config(self) -> Instance:
        from .instance_loader import InstanceLoader

        if self.has_framework_config():
            fwc = self.framework_config
            instance = fwc.create_model_instance(self)
        else:
            config_fn = Path(settings.BASE_DIR, 'configs', '%s.yaml' % self.identifier)
            loader = InstanceLoader.from_yaml(config_fn)
            instance = loader.instance
        return instance

    def _initialize_instance(self, node_refs: bool = False) -> Instance:
        with sentry_sdk.start_span(name='create-instance-from-config: %s' % self.identifier, op='function'):
            instance = self._create_from_config()

        with sentry_sdk.start_span(name='update-instance-from-configs: %s' % self.identifier, op='function'):
            self.update_instance_from_configs(instance, node_refs=node_refs)
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
    def enter_instance_context(self):
        if self.identifier in _pytest_instances:
            instance = _pytest_instances[self.identifier]
        else:
            instance = self._initialize_instance(node_refs=True)

        token = instance_context.set(instance)
        try:
            with sentry_sdk.new_scope() as scope, logger.contextualize(instance=self.identifier):
                self.set_instance_scope(scope)
                yield instance
        finally:
            instance_context.reset(token)

    @asynccontextmanager
    async def enter_instance_context_async(self):
        if self.identifier in _pytest_instances:
            instance = _pytest_instances[self.identifier]
        else:
            instance = await sync_to_async(self._initialize_instance)(node_refs=True)

        token = instance_context.set(instance)
        try:
            with sentry_sdk.new_scope() as scope:
                self.set_instance_scope(scope)
                yield instance
        finally:
            instance_context.reset(token)


    def _get_instance(self, node_refs: bool = False) -> Instance:
        if self.identifier in _pytest_instances:
            return _pytest_instances[self.identifier]

        current_instance = instance_context.get()
        if current_instance is not None and current_instance.id == self.identifier:
            return current_instance

        self.log.info("Creating new instance")
        with instance_cache_lock:
            instance = self._initialize_instance(node_refs=node_refs)
        return instance

    def get_instance(self, node_refs: bool = False) -> Instance:
        # Unit tests will set the Instance to `_instance` so that we don't need
        # to read the YAML configs
        instance = self._get_instance(node_refs=node_refs)
        return instance

    def get_name(self) -> str:
        if self.name:
            return self.name
        instance = self.get_instance()
        return str(instance.name)

    @property
    def default_language(self) -> str:
        return self.primary_language

    @property
    def supported_languages(self) -> list[str]:
        return [self.primary_language, *self.other_languages]

    @cached_property
    def root_page(self) -> Page:
        assert self.site is not None
        return self.site.root_page

    @cached_property
    def action_list_page(self) -> ActionListPage | None:
        from pages.models import ActionListPage
        qs = self.root_page.get_descendants().type(ActionListPage).specific()
        return cast('ActionListPage | None', qs.first())

    def get_translated_root_page(self) -> Page:
        """Return root page in activated language, fall back to default language."""
        root: Page = self.root_page
        language = get_language()
        language = convert_language_code(language, 'wagtail')
        try:
            locale = Locale.objects.get(language_code=language)
            root = root.get_translation(locale)  # type: ignore
        except (Locale.DoesNotExist, Page.DoesNotExist):
            pass
        return root

    def sync_nodes(self, update_existing=False, delete_stale=False, overwrite=False):
        from nodes.datasets import DBDataset

        instance = self.get_instance()
        node_configs = {n.identifier: n for n in self.nodes.all()}
        found_nodes = set()
        for node in instance.context.nodes.values():
            node_config = node_configs.get(node.id)
            if node_config is None:
                node_config = NodeConfig(instance=self, **node.as_node_config_attributes())
                self.log.info("Creating node config for node %s" % node.id)
                node_config.save()
                has_db_datasets = any(isinstance(ds, DBDataset) for ds in node.input_dataset_instances)
                if has_db_datasets:
                    node_config.update_relations_from_node(node)
                node.database_id = node_config.pk
            else:
                found_nodes.add(node.id)
                if update_existing:
                    node_config.update_from_node(node, overwrite=overwrite)
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
            update_existing=False,
            delete_stale=False,
        ):
        found_cats = set()
        instance = self.get_instance()
        default_lang = instance.default_language
        assert scope.identifier is not None
        dim = instance.context.dimensions[scope.identifier]

        cats = {cat.identifier: cat for cat in dataset_dim.categories.all()}
        for cat in dim.categories:
            cat_obj = cats.get(cat.id)
            if cat_obj is None:
                label, i18n = get_modeltrans_attrs_from_str(cat.label, 'label', default_lang)
                cat_obj = DimensionCategory.objects.create(
                    dimension=dataset_dim,
                    identifier=cat.id,
                    label=label,
                    i18n=i18n
                )
                print("Creating category %s" % cat.id)
            else:
                found_cats.add(cat_obj.pk)
                label, i18n = get_modeltrans_attrs_from_str(cat.label, 'label', default_lang)
                if i18n != cat_obj.i18n or cat_obj.label != label:
                    cat_obj.label, cat_obj.i18n = label, i18n
                    print('Updating category %s' % cat.id)
                    cat_obj.save()

        if delete_stale:
            for cat_obj in cats.values():
                if cat_obj.pk in found_cats:
                    continue
                print("Deleting stale category %s" % cat_obj)
                cat_obj.delete()

    def sync_dimension(self, dim: NodeDimension, update_existing=False, delete_stale=False) -> DatasetDimensionModel:
        scope = DimensionScope.objects.filter(
            scope_content_type=ContentType.objects.get_for_model(self),
            scope_id=self.pk,
            identifier=dim.id,
        ).first()
        label, i18n = get_modeltrans_attrs_from_str(dim.label, 'label', self.primary_language)
        if scope is None:
            dim_obj = DatasetDimensionModel.objects.create(name=label, i18n=i18n)
            scope = DimensionScope.objects.create(
                scope_content_type=ContentType.objects.get_for_model(self),
                scope_id=self.pk,
                identifier=dim.id,
                dimension=dim_obj
            )
            print("Creating dimension %s" % dim.id)
        else:
            dim_obj = scope.dimension

        if update_existing and (dim_obj.name != label or dim_obj.i18n != i18n):
            if dim_obj.pk:
                print('Updating dimension %s' % dim.id)
            dim_obj.name = label
            dim_obj.i18n = i18n
            dim_obj.save()

        self.sync_categories(dataset_dim=dim_obj, scope=scope, update_existing=update_existing, delete_stale=delete_stale)
        return dim_obj

    def sync_dimensions(self, update_existing=False, delete_stale=False) -> None:
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
        pks = [node.database_id for node in root_nodes]
        return list(self.nodes.filter(pk__in=pks))

    def _create_default_pages(self) -> Page:
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
        assert home_page_conf is not None
        assert home_page_conf.outcome_node is not None

        root_node: Page = cast('Page', Page.get_first_root_node())
        with override(self.primary_language):
            locale, _ = Locale.objects.get_or_create(language_code=self.primary_language)
            try:
                home_page = home_pages.get(slug=self.identifier)
            except Page.DoesNotExist:
                assert home_page_conf.outcome_node is not None
                home_page = root_node.add_child(instance=OutcomePage(
                    locale=locale,
                    title=self.get_name(),
                    slug=self.identifier,
                    url_path='',
                    outcome_node=outcome_nodes[home_page_conf.outcome_node],
                ))

            action_list_pages: models.QuerySet[ActionListPage] = home_page.get_children().type(ActionListPage)  # type: ignore
            if not action_list_pages.exists():
                home_page.add_child(instance=ActionListPage(
                    title=gettext("Actions"), slug='actions', show_in_menus=True, show_in_footer=True,
                ))

            for page_config in instance.pages:
                slug = page_config.id
                if slug == 'home':
                    continue

                page = cast('OutcomePage', home_page.get_children().filter(slug=slug).first())
                if page is not None:
                    continue

                assert page_config.outcome_node is not None
                home_page.add_child(instance=OutcomePage(
                    locale=locale,
                    title=str(page_config.name),
                    slug=slug,
                    url_path=page_config.path,
                    outcome_node=outcome_nodes[page_config.outcome_node],
                    show_in_menus=page_config.show_in_menus,
                    show_in_footer=page_config.show_in_footer,
                ))

        return home_page

    def create_default_content(self):
        pp = self.permission_policy()
        pp.admin_role.create_or_update_instance_group(self)
        pp.viewer_role.create_or_update_instance_group(self)

        root_page = self._create_default_pages()
        if self.site is None and self.site_url is not None:
            o = urlparse(self.site_url)
            site = Site(site_name=self.get_name(), hostname=o.hostname, root_page=root_page)
            site.save()
            self.site = site
            self.save(update_fields=['site'])

    def invalidate_cache(self, save: bool = True):
        self.cache_invalidated_at = timezone.now()
        self.log.info("Invalidating cache")
        self.save(update_fields=['cache_invalidated_at'])

    def notify_change(self):
        self.update_modified_at(save=False)
        self.invalidate_cache(save=False)
        self.log.info("Instance modified")
        self.save(update_fields=['modified_at', 'cache_invalidated_at'])
        cl = get_channel_layer()
        if cl is None:
            return
        async_to_sync(cl.group_send)(INSTANCE_CHANGE_GROUP, {
            'type': INSTANCE_CHANGE_TYPE,
            'pk': self.pk,
        })

    @cached_property
    def log(self) -> Logger:
        return logger.bind(instance=self.identifier, markup=True)


class InstanceHostnameManager(models.Manager):
    def get_by_natural_key(self, instance_identifier, hostname, base_path):
        instance = InstanceConfig.objects.get_by_natural_key(instance_identifier)
        return self.get(instance=instance, hostname=hostname, base_path=base_path)


class InstanceHostname(models.Model):
    instance = models.ForeignKey(
        InstanceConfig, on_delete=models.CASCADE, related_name='hostnames',
    )
    hostname = models.CharField(max_length=100)
    base_path = models.CharField(max_length=100, blank=True, default='')

    extra_script_urls = ArrayField(models.URLField(max_length=300), default=list)

    objects = InstanceHostnameManager()

    class Meta:
        verbose_name = _('Instance hostname')
        verbose_name_plural = _('Instance hostnames')
        unique_together = (('instance', 'hostname'), ('hostname', 'base_path'))

    def __str__(self):
        return '%s at %s [basepath %s]' % (self.instance, self.hostname, self.base_path)

    def natural_key(self):
        return self.instance.natural_key() + (self.hostname, self.base_path)


class InstanceTokenManager(models.Manager):
    def get_by_natural_key(self, instance_identifier, token, created_at):
        instance = InstanceConfig.objects.get_by_natural_key(instance_identifier)
        return self.get(instance=instance, token=token, created_at=created_at)


class InstanceToken(models.Model):
    instance = models.ForeignKey(
        InstanceConfig, on_delete=models.CASCADE, related_name='tokens',
    )
    token = models.CharField(max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)

    objects = InstanceTokenManager()

    class Meta:
        verbose_name = _('Instance token')
        verbose_name_plural = _('Instance tokens')

    def __str__(self) -> str:
        return 'Token for %s' % str(self.instance)

    def natural_key(self):
        return self.instance.natural_key() + (self.token, self.created_at)


class NodeConfigQuerySet(MultilingualQuerySet['NodeConfig'], PathsQuerySet['NodeConfig']):  # type: ignore[override]
    pass


_NodeConfigManager = models.Manager.from_queryset(NodeConfigQuerySet)
class NodeConfigManager(MLModelManager['NodeConfig', NodeConfigQuerySet], _NodeConfigManager):  # pyright: ignore
    """Model manager for NodeConfig."""

    def get_by_natural_key(self, instance_identifier, identifier):
        instance = InstanceConfig.objects.get_by_natural_key(instance_identifier)
        return self.get(instance=instance, identifier=identifier)

del _NodeConfigManager

class NodeConfig(PathsModel, RevisionMixin, ClusterableModel, index.Indexed, UUIDIdentifiedModel):
    instance: FK[InstanceConfig] = models.ForeignKey(
        InstanceConfig, on_delete=models.CASCADE, related_name='nodes', editable=False,
    )
    identifier = IdentifierField(max_length=200)
    name = models.CharField(max_length=200, null=True, blank=True)
    order = models.PositiveIntegerField(
        null=True, blank=True, verbose_name=_('Order'),
    )
    is_visible = models.BooleanField(default=True)
    goal = RichTextField[str | None, str | None](
        null=True, blank=True, verbose_name=_('Goal'), editor='very-limited',
        max_length=1000,
    ) # pyright: ignore
    short_description = RichTextField[str | None, str | None](
        null=True, blank=True, verbose_name=_('Short description'), editor='limited',
    ) # pyright: ignore
    description = RichTextField[str | None, str | None](
        null=True, blank=True, verbose_name=_('Description'),
    ) # -> StreamField
    body = StreamField([
        ('card_list', CardListBlock()),
        ('paragraph', blocks.RichTextBlock()),
    ], use_json_field=True, blank=True)

    indicator_node: FK[NodeConfig | None] = models.ForeignKey(
        'self', null=True, blank=True, on_delete=models.SET_NULL, related_name='indicates_nodes',
    )

    datasets: M2M[DatasetModel, NodeDataset] = models.ManyToManyField(DatasetModel, through='NodeDataset', related_name='nodes')

    color: CharField[str, str] = ColorField(max_length=20, blank=True)
    input_data = models.JSONField(null=True, editable=False)
    params = models.JSONField(null=True, editable=False)

    created_at = models.DateTimeField(default=timezone.now)
    modified_at = models.DateTimeField(auto_now=True)

    i18n = TranslationField(
        fields=('name', 'short_description', 'description', 'goal'),
        default_language_field='instance__primary_language',
    )  # pyright: ignore
    name_i18n: str | None
    short_description_i18n: str | None
    description_i18n: str | None
    goal_i18n: str | None

    search_fields = [
        index.AutocompleteField('identifier'),
        index.AutocompleteField('name'),
        index.SearchField('name'),
        index.SearchField('identifier'),
        index.FilterField('instance'),
    ]
    search_auto_update = False
    wagtail_reference_index_ignore = True

    objects: ClassVar[NodeConfigManager] = NodeConfigManager()  # pyright: ignore

    _node: Node | None

    indicates_nodes: RevMany[NodeConfig]

    class Meta:  # pyright: ignore
        verbose_name = _('Node')
        verbose_name_plural = _('Nodes')
        unique_together = (('instance', 'identifier'),)

    @classmethod
    def permission_policy(cls) -> ParentInheritedPolicy[Self, InstanceConfig, NodeConfigQuerySet]:
        return ParentInheritedPolicy(cls, InstanceConfig, 'instance', disallowed_actions=('add', 'delete'))

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

        if self.input_data:
            assert len(node.input_dataset_instances) == 1
            # disable legacy input data stuff
            # node.replace_input_data(self.input_data)

        # FIXME: Override params

    def update_from_node(self, node: Node, overwrite=False):
        """Set attributes of this instance from revelant fields of the given node but does not save."""

        overwritten = False

        conf = node.as_node_config_attributes()
        i18n = conf.pop('i18n', None)
        for k, v in node.as_node_config_attributes().items():
            if overwrite or getattr(self, k, None) is None:
                setattr(self, k, v)
                overwritten = True

        if i18n is not None:
            if not self.i18n:
                self.i18n = {}
            assert isinstance(self.i18n, dict)
            self.i18n |= cast('dict', i18n)

        if overwritten:
            self.instance.log.info('Overwrote contents in node %s' % str(node))

        if self.pk:
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

        if self.uuid is None:
            self.uuid = uuid.uuid4()

        return super().save(**kwargs)

    def natural_key(self):
        return self.instance.natural_key() + (self.identifier,)


class NodeDataset(models.Model):
    node = models.ForeignKey(NodeConfig, on_delete=models.CASCADE, related_name='datasets_edges')
    dataset = models.ForeignKey(DatasetModel, on_delete=models.PROTECT, related_name='nodes_edges')

    class Meta:
        verbose_name = _('Node dataset')
        verbose_name_plural = _('Node datasets')

    def __str__(self) -> str:
        return f'{self.node.identifier} -> {self.dataset}'
