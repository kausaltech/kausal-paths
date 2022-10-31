from __future__ import annotations
from datetime import datetime

import os
from typing import Optional, Tuple, Union
from urllib.parse import urlparse

from django.conf import settings
from django.db import models
from django.utils import timezone
from django.utils.translation import get_language, gettext_lazy as _, override
from modelcluster.models import ClusterableModel
from modeltrans.fields import TranslationField
from wagtail.core.fields import RichTextField
from wagtail.core.models import Locale, Page
from wagtail.core.models.sites import Site
from common.i18n import get_modeltrans_attrs_from_str

from nodes.node import Node
from paths.utils import IdentifierField

from .instance import Instance, InstanceLoader

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


class InstanceConfig(models.Model):
    identifier = IdentifierField(max_length=100)
    name = models.CharField(max_length=150, verbose_name=_('name'), null=True)
    lead_title = models.CharField(blank=True, max_length=100, verbose_name=_('Lead title'))
    lead_paragraph = RichTextField(null=True, blank=True, verbose_name=_('Lead paragraph'))
    site_url = models.URLField(verbose_name=_('Site URL'), null=True)
    site = models.OneToOneField(Site, null=True, on_delete=models.PROTECT, editable=False, related_name='instance')

    is_protected = models.BooleanField(default=False)
    protection_password = models.CharField(max_length=50, null=True, blank=True)

    created_at = models.DateTimeField(default=timezone.now)
    modified_at = models.DateTimeField(auto_now=True)

    i18n = TranslationField(fields=('name', 'lead_title', 'lead_paragraph'))

    objects = InstanceConfigQuerySet.as_manager()

    # Type annotations
    nodes: models.manager.RelatedManager[NodeConfig]
    hostnames: models.manager.RelatedManager[InstanceHostname]

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

    def get_instance(self) -> Instance:
        if self.identifier in instance_cache:
            instance = instance_cache[self.identifier]
            if not self.nodes.exists():
                return instance
            latest_node_edit = self.nodes.all().order_by('-modified_at').values_list('modified_at', flat=True).first()
            assert isinstance(latest_node_edit, datetime)
            assert instance.modified_at is not None
            if latest_node_edit <= instance.modified_at and self.modified_at <= instance.modified_at:
                return instance

        config_fn = os.path.join(settings.BASE_DIR, 'configs', '%s.yaml' % self.identifier)
        loader = InstanceLoader.from_yaml(config_fn)
        instance = loader.instance
        self.update_instance_from_configs(instance)
        instance.modified_at = timezone.now()
        instance.context.generate_baseline_values()
        instance_cache[self.identifier] = instance
        return instance

    def get_name(self) -> str:
        if self.name:
            return self.name
        instance = self.get_instance()
        return str(instance.name)

    @property
    def default_language(self) -> str:
        return self.get_instance().default_language

    @property
    def root_page(self) -> Page:
        assert self.site is not None
        return self.site.root_page

    def get_translated_root_page(self):
        """Return root page in activated language, fall back to default language."""
        root = self.root_page
        language = get_language()
        try:
            locale = Locale.objects.get(language_code=language)
            root = root.get_translation(locale)
        except (Locale.DoesNotExist, Page.DoesNotExist):
            pass
        return root

    def sync_nodes(self, update_existing=False, delete_stale=False):
        instance = self.get_instance()
        node_configs = {n.identifier: n for n in self.nodes.all()}
        found_nodes = set()
        for node in instance.context.nodes.values():
            node_config = node_configs.get(node.id)
            if node_config is None:
                node_config = NodeConfig(instance=self, **node.as_node_config_attributes())
                print("Creating node config for node %s" % node.id)
                node_config.save()
            else:
                found_nodes.add(node.id)
                if update_existing:
                    node_config.update_from_node(node)
                    node_config.save()

        for node in list(node_configs.values()):
            if node.identifier in found_nodes:
                continue

            print("Node %s exists in database, but it's not found in node graph" % node.identifier)
            if delete_stale:
                node.delete()

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
        from pages.models import ActionListPage, OutcomePage

        root_pages: models.QuerySet['Page'] = Page.get_first_root_node().get_children()
        # Create default pages only in default language for now
        # TODO: Also create translations to other supported languages
        with override(self.default_language):
            try:
                root_page = root_pages.get(slug=self.identifier)
            except Page.DoesNotExist:
                outcome_nodes = self.get_outcome_nodes()
                locale, locale_created = Locale.objects.get_or_create(language_code=self.default_language)
                root_page = Page.get_first_root_node().add_child(instance=OutcomePage(
                    locale=locale,
                    title=self.get_name(),
                    slug=self.identifier,
                    url_path='',
                    outcome_node=outcome_nodes[0],
                ))
            action_list_pages = root_page.get_children().type(ActionListPage)
            if not action_list_pages.exists():
                root_page.add_child(instance=ActionListPage(
                    title=_("Actions"), slug='actions', show_in_menus=True, show_in_footer=True
                ))
        return root_page

    def create_default_content(self):
        if self.site is None and self.site_url is not None:
            root_page = self._create_default_pages()
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
        super().delete(**kwargs)

    def save(self, *args, **kwargs):
        if self.site is not None:
            # TODO: Update Site and root page attributes
            pass

        super().save(*args, **kwargs)

    def __str__(self) -> str:
        return self.get_name()


class InstanceHostname(models.Model):
    instance = models.ForeignKey(
        InstanceConfig, on_delete=models.CASCADE, related_name='hostnames'
    )
    hostname = models.CharField(max_length=100)
    base_path = models.CharField(max_length=100, blank=True, default='')

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


class NodeConfig(ClusterableModel):
    instance = models.ForeignKey(
        InstanceConfig, on_delete=models.CASCADE, related_name='nodes', editable=False
    )
    identifier = IdentifierField(max_length=200)
    name = models.CharField(max_length=200, null=True, blank=True)
    order = models.PositiveIntegerField(
        null=True, blank=True, verbose_name=_('Order')
    )
    short_description = RichTextField(
        null=True, blank=True, verbose_name=_('Short description')
    )
    description = RichTextField(
        null=True, blank=True, verbose_name=_('Description')
    )

    color = models.CharField(max_length=20, null=True, blank=True)
    input_data = models.JSONField(null=True, editable=False)
    params = models.JSONField(null=True, editable=False)

    created_at = models.DateTimeField(default=timezone.now)
    modified_at = models.DateTimeField(auto_now=True)

    i18n = TranslationField(fields=('name', 'short_description', 'description'))

    class Meta:
        verbose_name = _('Node')
        verbose_name_plural = _('Nodes')
        unique_together = (('instance', 'identifier'),)

    def get_node(self) -> Optional[Node]:
        instance = self.instance.get_instance()
        return instance.context.nodes.get(self.identifier)

    def update_node_from_config(self, node: Node):
        node.database_id = self.pk
        node.order = self.order

        if self.input_data:
            assert len(node.input_dataset_instances) == 1
            node.replace_input_data(self.input_data)

        # FIXME: Override params

    def update_from_node(self, node: Node, overwrite=False):
        """Sets attributes of this instance from revelant fields of the given node but does not save."""
        for k, v in node.as_node_config_attributes().items():
            if overwrite or getattr(self, k, None) is None:
                setattr(self, k, v)

    def can_edit_data(self):
        node = self.get_node()
        if node is None:
            return False
        if len(node.input_dataset_instances) != 1:
            return False
        return True

    def __str__(self) -> str:
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
