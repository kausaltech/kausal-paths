from __future__ import annotations
import os
from typing import Optional

from django.conf import settings
from django.db import models
from django.utils.translation import gettext_lazy as _
from django.utils import timezone
from modeltrans.fields import TranslationField
import pandas as pd
from nodes.context import Context
from nodes.datasets import FixedDataset
from nodes.node import Node
from wagtail.core.fields import RichTextField

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

    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)

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
        return cls.objects.create(identifier=instance.id)

    def update_instance_from_configs(self, instance: Instance):
        for node_config in self.nodes.all():
            node = instance.context.nodes.get(node_config.identifier)
            if node is None:
                continue
            node_config.update_node_from_config(node, instance.context)

    def get_instance(self) -> Instance:
        if self.identifier in instance_cache:
            return instance_cache[self.identifier]
        config_fn = os.path.join(settings.BASE_DIR, 'configs', '%s.yaml' % self.identifier)
        loader = InstanceLoader.from_yaml(config_fn)
        instance = loader.instance
        instance.context.generate_baseline_values()
        self.update_instance_from_configs(instance)
        instance_cache[self.identifier] = instance
        return instance

    def get_name(self) -> str:
        if self.name:
            return self.name
        instance = self.get_instance()
        return instance.name

    @property
    def default_language(self) -> str:
        return self.get_instance().default_language

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


class NodeConfig(models.Model):
    instance = models.ForeignKey(
        InstanceConfig, on_delete=models.CASCADE, related_name='nodes', editable=False
    )
    identifier = IdentifierField()
    name = models.CharField(max_length=200, null=True, blank=True)
    short_description = RichTextField(
        null=True, blank=True, verbose_name=_('Short description')
    )
    body = RichTextField(
        null=True, blank=True, verbose_name=_('Body')
    )

    color = models.CharField(max_length=20, null=True, blank=True)
    input_data = models.JSONField(null=True, editable=False)
    params = models.JSONField(null=True, editable=False)

    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)

    i18n = TranslationField(fields=('name',))

    class Meta:
        verbose_name = _('Node')
        verbose_name_plural = _('Nodes')
        unique_together = (('instance', 'identifier'),)

    def get_node(self) -> Optional[Node]:
        instance = self.instance.get_instance()
        return instance.context.nodes.get(self.identifier)

    def update_node_from_config(self, node: Node, context: Context):
        node.database_id = self.pk

        if self.input_data:
            assert len(node.input_dataset_instances) == 1
            node.replace_input_data(self.input_data)

        # FIXME: Override params

    def can_edit_data(self):
        node = self.get_node()
        if node is None:
            return False
        # FIXME
        return True

    def __str__(self) -> str:
        node = self.get_node()
        if node is None:
            prefix = '⚠️ '
            name = ''
        else:
            prefix = ''
            name = node.name
        if self.name:
            name = self.name
        return f'{prefix}{name}'

