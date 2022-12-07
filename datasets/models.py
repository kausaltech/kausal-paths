from django.db import models
from django.utils.translation import gettext_lazy as _

from modeltrans.fields import TranslationField
from modelcluster.models import ClusterableModel, ParentalKey, ParentalManyToManyField
from wagtail.admin.panels import FieldPanel, InlinePanel
from wagtail.models import Orderable

from paths.utils import IdentifierField
from nodes.models import InstanceConfig
from nodes.constants import YEAR_COLUMN, FORECAST_COLUMN, VALUE_COLUMN


class ColumnType(models.TextChoices):
    YEAR = YEAR_COLUMN, _('Year')
    FORECAST = FORECAST_COLUMN, _('Forecast')
    VALUE = VALUE_COLUMN, _('Value')


class Dataset(ClusterableModel):
    instance = models.ForeignKey(
        InstanceConfig, on_delete=models.CASCADE, related_name='datasets'
    )
    identifier = IdentifierField()
    name = models.CharField(max_length=200)

    #dimensions = ParentalManyToManyField(
    #    'DatasetDimension', related_name='datasets', blank=True
    #)
    data = models.JSONField(null=True)

    panels = [
        FieldPanel('instance'),
        FieldPanel('identifier'),
        FieldPanel('name'),
        #InlinePanel('columns'),
    ]

    def __str__(self):
        return '%s [%s]' % (self.name, self.identifier)


class DatasetDimension(models.Model):
    instance = models.ForeignKey('nodes.InstanceConfig', on_delete=models.CASCADE, related_name='dataset_dimensions', editable=False)
    identifier = IdentifierField()
    name = models.CharField(max_length=50)

    i18n = TranslationField(fields=('name',))

    panels = [
        FieldPanel('instance'),
        FieldPanel('identifier'),
        FieldPanel('name'),
    ]

    def __str__(self):
        return self.name
