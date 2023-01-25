from __future__ import annotations
from datetime import date

import pandas as pd
import pint_pandas
from django.db import models
from django.utils.translation import gettext_lazy as _
from django.contrib.postgres.fields import ArrayField

from modeltrans.fields import TranslationField
from modelcluster.models import ClusterableModel, ParentalManyToManyField
from wagtail.admin.panels import FieldPanel

from paths.utils import IdentifierField, OrderedModel, UUIDIdentifierField, UnitField, UserModifiableModel
from nodes.models import InstanceConfig
from nodes.constants import YEAR_COLUMN
from nodes.datasets import JSONDataset


class Dataset(ClusterableModel, UserModifiableModel):
    instance = models.ForeignKey(
        InstanceConfig, on_delete=models.CASCADE, related_name='datasets'
    )
    identifier = IdentifierField()
    uuid = UUIDIdentifierField()
    years = ArrayField(models.IntegerField(), null=True, blank=True)
    name = models.CharField(max_length=200)

    dimensions = ParentalManyToManyField(
        'Dimension', related_name='datasets', blank=True, through='DatasetDimension'
    )
    metrics: models.manager.RelatedManager[DatasetMetric]

    table = models.JSONField(null=True)

    class Meta:
        unique_together = (('instance', 'identifier'),)
        ordering = ('instance', 'name')

    panels = [
        FieldPanel('instance'),
        FieldPanel('identifier'),
        FieldPanel('name'),
        #InlinePanel('columns'),
    ]

    def __str__(self):
        return '%s [%s]' % (self.name, self.identifier)

    def generate_empty_table(self) -> dict:
        dtypes = {}
        metric_cols = []
        for m in self.metrics.all():
            dtypes[m.identifier] = pint_pandas.PintType(m.unit)
            metric_cols.append(m.identifier)

        ctx = self.instance.get_instance().context
        dims = [ctx.dimensions[dim.identifier] for dim in self.dimensions.all()]
        dim_cats = [dim.get_cat_ids_ordered() for dim in dims]
        today = date.today()
        years = self.years if self.years is not None else range(2000, today.year + 1)
        index = pd.MultiIndex.from_product([years, *dim_cats], names=[YEAR_COLUMN, *[dim.id for dim in dims]])
        df = pd.DataFrame(index=index)
        for m in metric_cols:
            df[m] = pd.Series(dtype=dtypes[m])
        data = JSONDataset.serialize_df(df)
        return data


class DatasetMetric(OrderedModel):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='metrics')
    identifier = IdentifierField()
    label = models.CharField(verbose_name=_('label'), max_length=80)
    uuid = UUIDIdentifierField()
    unit = UnitField()

    i18n = TranslationField(fields=('label',))

    class Meta:
        unique_together = (('dataset', 'identifier'),)
        ordering = ('order',)

    def filter_siblings(self, qs: models.QuerySet['DatasetMetric']):
        return qs.filter(dataset=self.dataset)

    def __str__(self):
        return self.label


class DatasetComment(UserModifiableModel):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='comments')
    uuid = UUIDIdentifierField()
    row_uuid = models.UUIDField()
    text = models.TextField()

    class Meta:
        ordering = ('dataset', 'created_at')

    def __str__(self):
        return 'Comment on %s (created by %s at %s)' % (self.dataset, self.created_by, self.created_at)


class Dimension(UserModifiableModel):
    instance = models.ForeignKey(InstanceConfig, on_delete=models.CASCADE, related_name='dimensions', editable=False)
    identifier = IdentifierField()
    uuid = UUIDIdentifierField()
    label = models.CharField(verbose_name=_('label'), max_length=50)

    i18n = TranslationField(fields=('label',))

    categories: models.manager.RelatedManager[DimensionCategory]

    class Meta:
        unique_together = (('instance', 'identifier'),)
        ordering = ('instance', 'label')

    panels = [
        FieldPanel('instance'),
        FieldPanel('identifier'),
        FieldPanel('label'),
    ]

    def __str__(self):
        return self.label

    def sync_categories(self, update_existing=False, delete_stale=False):
        found_cats = set()
        instance = self.instance.get_instance()
        dim = instance.context.dimensions[self.identifier]
        cats = {cat.identifier: cat for cat in self.categories.all()}
        for cat in dim.categories:
            cat_obj = cats.get(cat.id)
            if cat_obj is None:
                cat_obj = DimensionCategory(dimension=self, identifier=cat.id, label=cat.label)
                print("Creating category %s" % cat.id)
                cat_obj.save()
            else:
                found_cats.add(cat.id)


class DatasetDimension(OrderedModel):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    dimension = models.ForeignKey(Dimension, on_delete=models.CASCADE)

    class Meta:
        unique_together = (('dataset', 'dimension'),)
        ordering = ('dataset', 'order')

    def __str__(self):
        return '%s in %s' % (self.dimension, self.dataset)

    def filter_siblings(self, qs: models.QuerySet['DatasetDimension']):
        return qs.filter(dataset=self.dataset)


class DimensionCategory(UserModifiableModel, OrderedModel):
    dimension = models.ForeignKey(Dimension, on_delete=models.CASCADE, related_name='categories')
    identifier = IdentifierField()
    uuid = UUIDIdentifierField()
    label = models.CharField(max_length=50)

    i18n = TranslationField(fields=('label',))

    class Meta:
        ordering = ('dimension', 'order')
        unique_together = (('dimension', 'identifier'),)

    def __str__(self):
        return self.label

    def filter_siblings(self, qs: models.QuerySet['DimensionCategory']):
        return qs.filter(dimension=self.dimension)
