from __future__ import annotations

import os
from typing import Optional
from urllib.parse import urlparse

from django.conf import settings
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from modelcluster.models import ClusterableModel
from modeltrans.fields import TranslationField
from wagtail.core.fields import RichTextField
from wagtail.core.models import Page
from wagtail.core.models.sites import Site

from nodes.node import Node
from paths.utils import IdentifierField

from .instance import Instance, InstanceLoader

instance_cache: dict[str, Instance] = {}


class InstanceQuerySet(models.QuerySet):
    def for_hostname(self, hostname):
        hostname = hostname.lower()

        # Support localhost-based URLs for development
        parts = hostname.split('.')
        if len(parts) == 2 and parts[1] == 'localhost':
            return self.filter(identifier__iexact=parts[0])

        return self.filter(hostnames__hostname=hostname)


class InstanceConfig(models.Model):
    identifier = IdentifierField()
    name = models.CharField(max_length=150, verbose_name=_('name'), null=True)
    lead_title = models.CharField(blank=True, max_length=100, verbose_name=_('Lead title'))
    lead_paragraph = RichTextField(null=True, blank=True, verbose_name=_('Lead paragraph'))
    site_url = models.URLField(verbose_name=_('Site URL'), null=True)
    site = models.OneToOneField(Site, null=True, on_delete=models.PROTECT, editable=False, related_name='instance')

    created_at = models.DateTimeField(default=timezone.now)
    modified_at = models.DateTimeField(default=timezone.now)

    i18n = TranslationField(fields=('name',))

    objects = models.Manager.from_queryset(InstanceQuerySet)()

    # Type annotations
    nodes: "models.manager.RelatedManager[NodeConfig]"

    class Meta:
        verbose_name = _('Instance')
        verbose_name_plural = _('Instances')

    @classmethod
    def create_for_instance(cls, instance: Instance) -> InstanceConfig:
        assert not cls.objects.filter(identifier=instance.id).exists()
        return cls.objects.create(identifier=instance.id, site_url=instance.site_url)

    def update_instance_from_configs(self, instance: Instance):
        for node_config in self.nodes.all():
            node = instance.context.nodes.get(node_config.identifier)
            if node is None:
                continue
            node_config.update_node_from_config(node)

    def get_instance(self) -> Instance:
        if self.identifier in instance_cache:
            instance = instance_cache[self.identifier]
            if not self.nodes.exists():
                return instance
            latest_node_edit = self.nodes.all().order_by('-modified_at').values_list('modified_at', flat=True).first()
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
        return self.site.root_page

    def sync_nodes(self):
        instance = self.get_instance()
        node_configs = {n.identifier: n for n in self.nodes.all()}
        found_nodes = set()
        new_nodes = []
        for node in instance.context.nodes.values():
            node_config = node_configs.get(node.id)
            if node_config is None:
                new_nodes.append(node)
            else:
                found_nodes.add(node.id)

        for node in new_nodes:
            node_obj = NodeConfig(instance=self, identifier=node.id)
            print("Creating node config for node %s" % node.id)
            node_obj.save()

        for node in node_configs.values():
            if node.identifier not in found_nodes:
                print("Node %s exists in database, but it's not found in node graph" % node.identifier)

    def update_modified_at(self, save=True):
        self.modified_at = timezone.now()
        if save:
            self.save(update_fields=['modified_at'])

    def get_outcome_nodes(self) -> list[NodeConfig]:
        instance = self.get_instance()
        root_nodes = instance.context.get_root_nodes()
        pks = [node.database_id for node in root_nodes]
        return self.nodes.filter(pk__in=pks)

    def _create_default_pages(self) -> Page:
        from pages.models import ActionListPage, OutcomePage

        root_pages = Page.get_first_root_node().get_children()
        try:
            root_page = root_pages.get(slug=self.identifier)
        except Page.DoesNotExist:
            outcome_nodes = self.get_outcome_nodes()
            root_page = Page.get_first_root_node().add_child(instance=OutcomePage(
                title=self.get_name(), slug=self.identifier, url_path='', outcome_node=outcome_nodes[0]
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
    hostname = models.CharField(max_length=100, unique=True)
    base_path = models.CharField(max_length=100, null=True, blank=True)

    class Meta:
        verbose_name = _('Instance hostname')
        verbose_name_plural = _('Instance hostnames')


class NodeConfig(ClusterableModel):
    instance = models.ForeignKey(
        InstanceConfig, on_delete=models.CASCADE, related_name='nodes', editable=False
    )
    identifier = IdentifierField()
    name = models.CharField(max_length=200, null=True, blank=True)
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
    modified_at = models.DateTimeField(default=timezone.now)

    i18n = TranslationField(fields=('name',))

    class Meta:
        verbose_name = _('Node')
        verbose_name_plural = _('Nodes')
        unique_together = (('instance', 'identifier'),)

    def get_node(self) -> Optional[Node]:
        instance = self.instance.get_instance()
        return instance.context.nodes.get(self.identifier)

    def update_node_from_config(self, node: Node):
        node.database_id = self.pk

        if self.input_data:
            assert len(node.input_dataset_instances) == 1
            node.replace_input_data(self.input_data)

        # FIXME: Override params

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
