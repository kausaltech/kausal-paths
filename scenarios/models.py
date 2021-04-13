from django.db import models
from django.utils.translation import gettext_lazy as _


class Instance(models.Model):
    name = models.CharField(verbose_name=_('Name'), max_length=100)

    def __str__(self):
        return self.name


class NodeConfig(models.Model):
    instance = models.ForeignKey(Instance, on_delete=models.CASCADE)
    identifier = models.CharField(max_length=100)
    node_type = models.CharField(max_length=100)  # validate against implemented node classes
    unit = models.CharField(max_length=20)  # in format compatible with pint
    input_nodes = models.ManyToManyField('self', blank=True, null=True)
    input_datasets = models.JSONField(null=True, blank=True)
    values = models.JSONField(null=True, blank=True)

    class Meta:
        unique_together = (('instance', 'identifier'),)

    def __str__(self):
        return self.identifier
