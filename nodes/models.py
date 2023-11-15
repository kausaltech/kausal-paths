from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, Optional, Sequence, Tuple, Union, cast
from urllib.parse import urlparse

from django.conf import settings
from django.contrib.auth.models import Group
from django.contrib.postgres.fields import ArrayField
from django.db import models
from django.utils import timezone
from django.utils.translation import get_language, gettext, override
from django.utils.translation import gettext_lazy as _
from loguru import logger
from modelcluster.models import ClusterableModel
from modeltrans.fields import TranslationField
from wagtail import blocks
from wagtail.fields import RichTextField, StreamField
from wagtail.models import Locale, Page, RevisionMixin
from wagtail.models.sites import Site
from wagtail.search import index
from wagtail_color_panel.fields import ColorField  # type: ignore

from common.i18n import get_modeltrans_attrs_from_str
from nodes.node import Node
from pages.blocks import CardListBlock
from paths.permissions import PathsPermissionPolicy
from paths.types import PathsModel, UserOrAnon
from paths.utils import (
    ChoiceArrayField,
    IdentifierField,
    UserModifiableModel,
    UUIDIdentifierField,
    get_default_language,
    get_supported_languages,
)

from .instance import Instance, InstanceLoader

if TYPE_CHECKING:
    from loguru import Logger

    from datasets.models import Dataset as DatasetModel
    from datasets.models import Dimension as DimensionModel
    from pages.models import ActionListPage
    from users.models import User


instance_cache_lock = threading.Lock()
instance_cache: dict[str, Instance] = {}


def get_instance_identifier_from_wildcard_domain(hostname: str) -> Union[Tuple[str, str], Tuple[None, None]]:
    # Get instance identifier from hostname for development and testing
    parts = hostname.lower().split('.', maxsplit=1)
    if len(parts) == 2:
        if parts[1] in settings.HOSTNAME_INSTANCE_DOMAINS:
            return (parts[0], parts[1])
    return (None, None)


class InstanceConfigQuerySet(models.QuerySet['InstanceConfig']):
    def for_hostname(self, hostname: str):
        hostname = hostname.lower()
        hostnames = InstanceHostname.objects.filter(hostname=hostname)
        lookup = models.Q(id__in=hostnames.values_list('instance'))

        # Get instance identifier from hostname for development and testing
        identifier, _ = get_instance_identifier_from_wildcard_domain(hostname)
        if identifier:
            lookup |= models.Q(identifier=identifier)
        return self.filter(lookup)

    def adminable_for(self, user: User):
        return InstanceConfig.permission_policy.adminable_instances(user)

class InstancePermissionPolicy(PathsPermissionPolicy['InstanceConfig', InstanceConfigQuerySet]):
    def __init__(self):
        super().__init__(InstanceConfig, auth_model=None)

    def instances_user_has_any_permission_for(self, user: User, actions: Sequence[str]) -> InstanceConfigQuerySet:
        qs = super().instances_user_has_any_permission_for(user, actions)
        if not user.is_superuser:
            qs = qs.filter(admin_group__in=user.groups.all())
        return qs

    def adminable_instances(self, user: User) -> InstanceConfigQuerySet:
        return self.instances_user_has_any_permission_for(user, ['change'])


class InstanceConfigManager(models.Manager['InstanceConfig']):
    def for_hostname(self, hostname: str) -> InstanceConfigQuerySet: ...  # type: ignore


class InstanceConfig(PathsModel):
    identifier = IdentifierField(max_length=100, unique=True)
    uuid = models.UUIDField(verbose_name=_('UUID'), editable=False, null=True, unique=True)
    name = models.CharField(max_length=150, verbose_name=_('name'), null=True)
    lead_title = models.CharField(blank=True, max_length=100, verbose_name=_('Lead title'))
    lead_paragraph = RichTextField(null=True, blank=True, verbose_name=_('Lead paragraph'))
    site_url = models.URLField(verbose_name=_('Site URL'), null=True)
    site = models.OneToOneField(Site, null=True, on_delete=models.PROTECT, editable=False, related_name='instance')

    is_protected = models.BooleanField(default=False)
    protection_password = models.CharField(max_length=50, null=True, blank=True)

    created_at = models.DateTimeField(default=timezone.now)
    modified_at = models.DateTimeField(auto_now=True)
    cache_invalidated_at = models.DateTimeField(default=timezone.now)

    primary_language = models.CharField(max_length=8, choices=get_supported_languages(), default=get_default_language)
    other_languages = ChoiceArrayField(
        models.CharField(max_length=8, choices=get_supported_languages(), default=get_default_language),
        default=list,
    )

    admin_group = models.ForeignKey(
        Group, on_delete=models.PROTECT, editable=False, related_name='admin_instances',
        null=True
    )

    i18n = TranslationField(fields=('name', 'lead_title', 'lead_paragraph'))  # pyright: ignore

    objects: InstanceConfigManager = InstanceConfigQuerySet.as_manager()  # type: ignore

    # Type annotations
    nodes: models.manager.RelatedManager[NodeConfig]
    hostnames: models.manager.RelatedManager[InstanceHostname]
    dimensions: models.manager.RelatedManager['DimensionModel']
    datasets: models.manager.RelatedManager['DatasetModel']

    permission_policy: ClassVar[InstancePermissionPolicy]
    _instance: Instance

    search_fields = [
        index.SearchField('identifier'),
        index.SearchField('name_i18n'),
    ]

    class Meta:
        verbose_name = _('Instance')
        verbose_name_plural = _('Instances')

    @classmethod
    def create_for_instance(cls, instance: Instance, **kwargs) -> InstanceConfig:
        assert not cls.objects.filter(identifier=instance.id).exists()
        return cls.objects.create(identifier=instance.id, site_url=instance.site_url, **kwargs)

    def update_instance_from_configs(self, instance: Instance):
        for node_config in self.nodes.all():
            node = instance.context.nodes.get(node_config.identifier)
            if node is None:
                continue
            node_config.update_node_from_config(node)

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
        other_langs = set(instance.supported_languages) - set([self.primary_language])
        if set(self.other_languages or []) != other_langs:
            self.log.info('Updating instance.other_languages to [%s]' % ', '.join(other_langs))
            self.other_languages = list(other_langs)

    def _get_instance_from_memory(self):
        if self.identifier not in instance_cache:
            return
        instance: Instance = instance_cache[self.identifier]
        assert instance.modified_at is not None
        if self.modified_at > instance.modified_at:
            return

        if not self.nodes.exists():
            return instance
        latest_node_edit = self.nodes.all().order_by('-modified_at').values_list('modified_at', flat=True).first()
        assert isinstance(latest_node_edit, datetime)
        more_recent_nodes = latest_node_edit > instance.modified_at
        ic_recently_saved = self.modified_at > instance.modified_at
        if more_recent_nodes or ic_recently_saved:
            return
        return instance

    def _create_new_instance(self) -> Instance:
        config_fn = os.path.join(settings.BASE_DIR, 'configs', '%s.yaml' % self.identifier)
        loader = InstanceLoader.from_yaml(config_fn)
        instance = loader.instance
        self.update_instance_from_configs(instance)
        instance.modified_at = timezone.now()
        instance.context.load_all_dvc_datasets()
        if settings.ENABLE_PERF_TRACING:
            instance.context.perf_context.enabled = True
        return instance

    def _get_instance(self) -> Instance:
        if hasattr(self, '_instance'):
            return self._instance

        instance = self._get_instance_from_memory()
        if instance:
            setattr(self, '_instance', instance)
            return instance

        self.log.info("Creating new instance")
        instance = self._create_new_instance()
        instance_cache[self.identifier] = instance
        self._instance = instance
        return instance

    def get_instance(self, generate_baseline: bool = False) -> Instance:
        assert not generate_baseline
        with instance_cache_lock:
            instance = self._get_instance()
            if generate_baseline:
                instance.context.generate_baseline_values()
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
        qs = self.root_page.get_descendants().type(ActionListPage)
        return qs.first()

    def get_translated_root_page(self) -> Page:
        """Return root page in activated language, fall back to default language."""
        root: Page = self.root_page
        language = get_language()
        try:
            locale = Locale.objects.get(language_code=language)
            root = root.get_translation(locale)  # type: ignore
        except (Locale.DoesNotExist, Page.DoesNotExist):
            pass
        return root

    def sync_nodes(self, update_existing=False, delete_stale=False, overwrite=False):
        instance = self.get_instance()
        node_configs = {n.identifier: n for n in self.nodes.all()}
        found_nodes = set()
        for node in instance.context.nodes.values():
            node_config = node_configs.get(node.id)
            if node_config is None:
                node_config = NodeConfig(instance=self, **node.as_node_config_attributes())
                self.log.info("Creating node config for node %s" % node.id)
                node_config.save()
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

    def sync_dimensions(self, update_existing=False, delete_stale=False):
        from datasets.models import Dimension as DimensionModel

        DimensionModel.sync_dimensions(self, update_existing=update_existing, delete_stale=delete_stale)

    def update_modified_at(self, save=True):
        self.modified_at = timezone.now()
        if save:
            self.save(update_fields=['modified_at'])

    def get_outcome_nodes(self) -> list[NodeConfig]:
        instance = self.get_instance()
        root_nodes = [node for node in instance.context.nodes.values() if node.is_outcome]
        pks = [node.database_id for node in root_nodes]
        return list(self.nodes.filter(pk__in=pks))

    def _create_default_pages(self) -> Page:
        from pages.config import OutcomePage as OutcomePageConfig
        from pages.models import ActionListPage, OutcomePage

        root = cast(Page, Page.get_first_root_node())
        home_pages: models.QuerySet['Page'] = root.get_children()

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

        root_node: Page = cast(Page, Page.get_first_root_node())
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
                    outcome_node=outcome_nodes[home_page_conf.outcome_node]
                ))

            action_list_pages = home_page.get_children().type(ActionListPage)
            if not action_list_pages.exists():
                home_page.add_child(instance=ActionListPage(
                    title=gettext("Actions"), slug='actions', show_in_menus=True, show_in_footer=True
                ))

            for page_config in instance.pages:
                id = page_config.id
                if id == 'home':
                    continue

                page = home_page.get_children().filter(slug=id).first()
                if page is not None:
                    continue

                assert page_config.outcome_node is not None
                home_page.add_child(instance=OutcomePage(
                    locale=locale,
                    title=str(page_config.name),
                    slug=id,
                    url_path=page_config.path,
                    outcome_node=outcome_nodes[page_config.outcome_node],
                    show_in_menus=page_config.show_in_menus,
                    show_in_footer=page_config.show_in_footer,
                ))

        return home_page

    def create_default_content(self):
        root_page = self._create_default_pages()
        if self.site is None and self.site_url is not None:
            o = urlparse(self.site_url)
            site = Site(site_name=self.get_name(), hostname=o.hostname, root_page=root_page)
            site.save()
            self.site = site
            self.save(update_fields=['site'])

    def delete(self, **kwargs):
        site = self.site
        if site is not None:
            rp = site.root_page
            self.site = None
            self.save()
            site.delete()
            rp.get_descendants(inclusive=True).delete()
        self.nodes.all().delete()
        super().delete(**kwargs)

    def save(self, *args, **kwargs):
        if self.uuid is None:
            self.uuid = uuid.uuid4()

        if self.site is not None:
            # TODO: Update Site and root page attributes
            pass

        if self.admin_group is None:
            from admin_site.perms import AdminRole
            role = AdminRole()
            role.update_instance(self)

        super().save(*args, **kwargs)

    def invalidate_cache(self):
        self.cache_invalidated_at = timezone.now()
        self.log.info("Invalidating cache")
        self.save(update_fields=['cache_invalidated_at'])

    @cached_property
    def log(self) -> Logger:
        return logger.bind(instance=self.identifier, markup=True)

    def __str__(self) -> str:
        return self.get_name()


class InstanceHostname(models.Model):
    instance = models.ForeignKey(
        InstanceConfig, on_delete=models.CASCADE, related_name='hostnames'
    )
    hostname = models.CharField(max_length=100)
    base_path = models.CharField(max_length=100, blank=True, default='')

    extra_script_urls = ArrayField(models.URLField(max_length=300), default=list)

    class Meta:
        verbose_name = _('Instance hostname')
        verbose_name_plural = _('Instance hostnames')
        unique_together = (('instance', 'hostname'), ('hostname', 'base_path'))

    def __str__(self):
        return '%s at %s [basepath %s]' % (self.instance, self.hostname, self.base_path)


class InstanceToken(models.Model):
    instance = models.ForeignKey(
        InstanceConfig, on_delete=models.CASCADE, related_name='tokens'
    )
    token = models.CharField(max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = _('Instance token')
        verbose_name_plural = _('Instance tokens')


class DataSource(UserModifiableModel):
    """
    A DataSource represents a reusable reference to some published data source
    and is used to track where specific data values in datasets have come from.
    """
    instance = models.ForeignKey(
        InstanceConfig, on_delete=models.CASCADE, related_name='data_sources', editable=True,
        verbose_name=_('instance')
    )
    uuid = UUIDIdentifierField(null=False, blank=False)
    name = models.CharField(max_length=200, null=False, blank=False, verbose_name=_('name'))
    edition = models.CharField(max_length=100, null=True, blank=True, verbose_name=_('edition'))

    authority = models.CharField(
        max_length=200, verbose_name=_('authority'), help_text=_('The organization responsible for the data source'),
        null=True, blank=True
    )
    description = models.TextField(null=True, blank=True, verbose_name=_('description'))
    url = models.URLField(verbose_name=_('URL'), null=True, blank=True)

    def get_label(self):
        name, *rest = [p for p in (self.name, self.authority, self.edition) if p is not None]
        return f'{name}, {" ".join(rest)}'

    def __str__(self):
        return self.get_label()

    class Meta:
        verbose_name = _('Data source')
        verbose_name_plural = _('Data sources')


class NodeConfig(RevisionMixin, ClusterableModel, index.Indexed):
    instance = models.ForeignKey(
        InstanceConfig, on_delete=models.CASCADE, related_name='nodes', editable=False
    )
    identifier = IdentifierField(max_length=200)
    uuid = models.UUIDField(verbose_name=_('UUID'), editable=False, null=True, unique=True)
    name = models.CharField(max_length=200, null=True, blank=True)
    order = models.PositiveIntegerField(
        null=True, blank=True, verbose_name=_('Order')
    )
    goal = RichTextField(
        null=True, blank=True, verbose_name=_('Goal'), editor='very-limited',
        max_length=200,
    ) # pyright: ignore
    short_description = RichTextField(
        null=True, blank=True, verbose_name=_('Short description'), editor='limited',
    ) # pyright: ignore
    description = RichTextField(
        null=True, blank=True, verbose_name=_('Description')
    ) # -> StreamField
    body = StreamField([
        ('card_list', CardListBlock()),
        ('paragraph', blocks.RichTextBlock()),
    ], use_json_field=True, blank=True)

    indicator_node = models.ForeignKey(
        'self', null=True, blank=True, on_delete=models.SET_NULL, related_name='indicates_nodes',
    )

    color = ColorField(max_length=20, null=True, blank=True)
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

    class Meta:
        verbose_name = _('Node')
        verbose_name_plural = _('Nodes')
        unique_together = (('instance', 'identifier'),)

    def get_node(self, visible_for_user: UserOrAnon | None = None) -> Optional[Node]:
        if hasattr(self, '_node'):
            return getattr(self, '_node')

        instance = self.instance.get_instance()
        # FIXME: Node visibility restrictions
        node = instance.context.nodes.get(self.identifier)
        setattr(self, '_node', node)
        return node

    def update_node_from_config(self, node: Node):
        node.database_id = self.pk
        node.db_obj = self
        if self.order is not None:
            node.order = self.order

        if self.input_data:
            assert len(node.input_dataset_instances) == 1
            node.replace_input_data(self.input_data)

        # FIXME: Override params

    def update_from_node(self, node: Node, overwrite=False):
        """Sets attributes of this instance from revelant fields of the given node but does not save."""

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
            self.i18n |= cast(dict, i18n)

        if overwritten:
            self.instance.log.info('Overwrote contents in node %s' % str(node))

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

    def save(self, **kwargs):
        if self.uuid is None:
            self.uuid = uuid.uuid4()
        return super().save(**kwargs)


InstanceConfig.permission_policy = InstancePermissionPolicy()
