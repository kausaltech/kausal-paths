import os

from django.conf import settings
from django.db import models
from django.utils.translation import gettext_lazy as _

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

    objects = models.Manager.from_queryset(InstanceQuerySet)()

    class Meta:
        verbose_name = _('Instance')
        verbose_name_plural = _('Instances')

    def get_instance(self) -> Instance:
        if self.identifier in instance_cache:
            return instance_cache[self.identifier]
        config_fn = os.path.join(settings.BASE_DIR, 'configs', '%s.yaml' % self.identifier)
        loader = InstanceLoader.from_yaml(config_fn)
        instance = loader.instance
        instance.context.generate_baseline_values()
        instance_cache[self.identifier] = instance
        return instance


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
        InstanceConfig, on_delete=models.CASCADE, related_name='nodes'
    )
    identifier = IdentifierField()

    color = models.CharField(max_length=20, null=True, blank=True)
    forecast_values = models.JSONField(null=True)
    historical_values = models.JSONField(null=True)
    params = models.JSONField(null=True)

    class Meta:
        verbose_name = _('Node')
        verbose_name_plural = _('Nodes')
        unique_together = (('instance', 'identifier'),)


class NodeValue(models.Model):
    node = models.ForeignKey(
        NodeConfig, on_delete=models.CASCADE, related_name='values',
    )
    year = models.PositiveIntegerField()
    is_forecast = models.BooleanField()
    value = models.FloatField()
