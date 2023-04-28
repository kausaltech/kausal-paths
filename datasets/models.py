from __future__ import annotations
from datetime import date

import pandas as pd
import polars as pl
import pint_pandas
from django.db import models
from django.utils.translation import gettext_lazy as _
from django.contrib.postgres.fields import ArrayField

from modeltrans.fields import TranslationField
from modelcluster.models import ClusterableModel, ParentalKey
from wagtail.admin.panels import FieldPanel, InlinePanel

from paths.utils import IdentifierField, OrderedModel, UUIDIdentifierField, UnitField, UserModifiableModel
from nodes.models import InstanceConfig
from nodes.constants import YEAR_COLUMN
from nodes.datasets import JSONDataset
from nodes.dimensions import Dimension as NodeDimension
from common.i18n import get_modeltrans_attrs_from_str


class Dataset(ClusterableModel, UserModifiableModel):
    instance = models.ForeignKey(
        InstanceConfig, on_delete=models.CASCADE, related_name='datasets'
    )
    identifier = IdentifierField(max_length=150)
    uuid = UUIDIdentifierField()
    years = ArrayField(models.IntegerField())
    name = models.CharField(max_length=200)

    metrics: models.manager.RelatedManager[DatasetMetric]

    table = models.JSONField()

    i18n = TranslationField(fields=('name',))

    class Meta:
        unique_together = (('instance', 'identifier'),)
        ordering = ('instance', 'created_at')
        verbose_name = _('dataset')
        verbose_name_plural = _('datasets')

    panels = [
        FieldPanel('instance'),
        FieldPanel('identifier'),
        FieldPanel('name'),
        # InlinePanel('columns'),
    ]

    def __str__(self):
        return '%s [%s]' % (self.name, self.identifier)

    def generate_years_from_data(self) -> list[int]:
        assert self.table is not None
        data = self.table['data']
        df = pl.DataFrame(data)
        return list(df[YEAR_COLUMN].unique().sort())

    def generate_empty_table(self) -> dict:
        dtypes = {}
        metric_cols = []
        for m in self.metrics.all():
            dtypes[m.identifier] = pint_pandas.PintType(m.unit)
            metric_cols.append(m.identifier)

        ctx = self.instance.get_instance().context
        dims = [ctx.dimensions[dim_sel.dimension.identifier] for dim_sel in self.dimension_selections.all()]
        dim_cats = [[cat.identifier for cat in dim_sel.selected_categories.all()]
                    for dim_sel in self.dimension_selections.all()]
        today = date.today()
        years = self.years if self.years is not None else range(2000, today.year + 1)
        if len(dim_cats):
            index = pd.MultiIndex.from_product([years, *dim_cats], names=[YEAR_COLUMN, *[dim.id for dim in dims]])
        else:
            index = pd.Index(years, name=YEAR_COLUMN)
        df = pd.DataFrame(index=index)
        for m in metric_cols:
            df[m] = pd.Series(dtype=dtypes[m])
        data = JSONDataset.serialize_df(df, add_uuids=True)
        return data

    @classmethod
    def annotate_nr_unresolved_comments(cls, qs: models.QuerySet[Dataset]):
        unresolved = models.Count(
            'datasets_datasetcomment',
            filter=(
                models.Q(datasets_datasetcomment__type=DatasetComment.CommentType.REVIEW) &
                ~models.Q(datasets_datasetcomment__state=DatasetComment.State.RESOLVED)
            )
        )
        qs = qs.annotate(nr_unresolved_comments=unresolved)
        return qs


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


class CellMetadata(UserModifiableModel):
    cell_path = models.CharField(null=True, blank=True, max_length=300)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="%(app_label)s_%(class)s")

    class Meta:
        abstract = True


class DatasetSourceReference(CellMetadata):
    data_source = models.ForeignKey('nodes.DataSource', on_delete=models.PROTECT, related_name='references')

    def __str__(self):
        return f"{self.dataset.identifier} [{self.cell_path or 'all'}]: " + str(self.data_source)

    class Meta:
        ordering = ('dataset', 'cell_path')
        unique_together = (('dataset', 'cell_path'),)
        verbose_name = _('data source reference')
        verbose_name_plural = _('data source references')


class DatasetComment(CellMetadata):
    class CommentType(models.TextChoices):
        REVIEW = 'review', _('Review comment'),
        STICKY = 'sticky', _('Sticky comment'),

    class State(models.TextChoices):
        RESOLVED = 'resolved', _('Resolved'),
        UNRESOLVED = 'unresolved', _('Unresolved'),

    uuid = UUIDIdentifierField(null=True, blank=True)
    text = models.TextField()
    type = models.CharField(
        null=True,
        blank=True,
        max_length=20,
        choices=CommentType.choices
    )
    state = models.CharField(
        null=True,
        blank=True,
        max_length=20,
        choices=State.choices
    )
    resolved_at = models.DateTimeField(
        verbose_name=_('resolved at'), editable=False, null=True
    )
    resolved_by = models.ForeignKey(
        'users.User', null=True, on_delete=models.SET_NULL, related_name='resolved_comments'
    )

    def __str__(self):
        return 'Comment on %s (created by %s at %s)' % (self.dataset, self.created_by, self.created_at)

    class Meta:
        ordering = ('dataset', 'cell_path', '-created_at')
        verbose_name = _('comment')
        verbose_name_plural = _('comments')


class Dimension(ClusterableModel, UserModifiableModel):
    instance = models.ForeignKey(InstanceConfig, on_delete=models.CASCADE, related_name='dimensions', editable=True)
    identifier = IdentifierField()
    uuid = UUIDIdentifierField()
    label = models.CharField(verbose_name=_('label'), max_length=50)

    i18n = TranslationField(fields=('label',), default_language_field='instance__primary_language')

    categories: models.manager.RelatedManager[DimensionCategory]

    class Meta:
        unique_together = (('instance', 'identifier'),)
        ordering = ('instance', 'label')

    panels = [
        FieldPanel('instance'),
        FieldPanel('identifier'),
        FieldPanel('label'),
        InlinePanel('categories', [
            FieldPanel('label'),
            FieldPanel('identifier'),
        ]),
    ]

    def __str__(self):
        return self.label

    def sync_categories(self, update_existing=False, delete_stale=False):
        found_cats = set()
        instance = self.instance.get_instance()
        default_lang = instance.default_language
        dim = instance.context.dimensions[self.identifier]
        cats = {cat.identifier: cat for cat in self.categories.all()}
        for cat in dim.categories:
            cat_obj = cats.get(cat.id)
            if cat_obj is None:
                cat_obj = DimensionCategory(dimension=self, identifier=cat.id, label=cat.label)
                print("Creating category %s" % cat.id)
                cat_obj.save()
            else:
                found_cats.add(cat_obj.id)
                if not cat_obj.i18n or cat_obj.label != cat.label.i18n.get(default_lang):
                    cat_obj.label, cat_obj.i18n = get_modeltrans_attrs_from_str(cat.label, 'label', default_lang)
                    print('Updating category %s' % cat.id)
                    cat_obj.save()

        for cat_obj in cats.values():
            if cat_obj.id in found_cats:
                continue
            print("Deleting stale category %s" % cat_obj)
            cat_obj.delete()

    @classmethod
    def sync_dimension(cls, ic: InstanceConfig, dim: NodeDimension, update_existing=False, delete_stale=False):
        instance = ic.get_instance()
        dim_obj = ic.dimensions.filter(identifier=dim.id).first()
        if dim_obj is None:
            dim_obj = cls(instance=ic, identifier=dim.id)
            print("Creating dimension %s" % dim.id)

        label, i18n = get_modeltrans_attrs_from_str(dim.label, 'label', instance.default_language)  #type: ignore
        if update_existing and (dim_obj.label != label or dim_obj.i18n != i18n):
            if dim_obj.pk:
                print('Updating dimension %s' % dim.id)
            dim_obj.label = label
            dim_obj.i18n = i18n  # type: ignore
        dim_obj.save()
        dim_obj.sync_categories(update_existing=update_existing, delete_stale=delete_stale)
        return dim_obj

    @classmethod
    def sync_dimensions(cls, ic: InstanceConfig, update_existing=False, delete_stale=False):
        instance = ic.get_instance()
        # dims = {dim.identifier: dim for dim in self.dimensions.all()}
        found_dims = set()
        for dim in instance.context.dimensions.values():
            obj = cls.sync_dimension(ic, dim, update_existing=update_existing, delete_stale=delete_stale)
            found_dims.add(obj)

        if delete_stale:
            for dim_obj in ic.dimensions.all():
                if dim_obj not in found_dims:
                    dim_obj.delete()


class DatasetDimension(OrderedModel):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='dimension_selections')
    dimension = models.ForeignKey(Dimension, on_delete=models.CASCADE)
    selected_categories = models.ManyToManyField(
        to='DimensionCategory',
        through='DatasetDimensionSelectedCategory'
    )

    class Meta:
        unique_together = (('dataset', 'dimension'),)
        ordering = ('dataset', 'order')

    def __str__(self):
        return '%s in %s' % (self.dimension, self.dataset)

    def filter_siblings(self, qs: models.QuerySet['DatasetDimension']):
        return qs.filter(dataset=self.dataset)


class DimensionCategory(UserModifiableModel, OrderedModel):
    dimension = ParentalKey(Dimension, on_delete=models.CASCADE, related_name='categories')
    identifier = IdentifierField()
    uuid = UUIDIdentifierField()
    label = models.CharField(max_length=50)

    i18n = TranslationField(fields=('label',), default_language_field='dimension__instance__primary_language')

    class Meta:
        ordering = ('dimension', 'order')
        unique_together = (('dimension', 'identifier'),)

    def __str__(self):
        return self.label

    def filter_siblings(self, qs: models.QuerySet['DimensionCategory']):
        return qs.filter(dimension=self.dimension)


class DatasetDimensionSelectedCategory(OrderedModel):
    dataset_dimension = models.ForeignKey(DatasetDimension, on_delete=models.CASCADE)
    category = models.ForeignKey(DimensionCategory, on_delete=models.PROTECT)

    def save(self, *args, **kwargs):
        if self.category not in self.dataset_dimension.dimension.categories.all():
            raise ValueError(f'{self.category} is not part of {self.dataset_dimension.dimension}')
        super().save(*args, **kwargs)

    def filter_siblings(self, qs: models.QuerySet['DatasetDimensionSelectedCategory']):
        return qs.filter(dataset_dimension=self.dataset_dimension)

    class Meta:
        ordering = ('order', )
        unique_together = (('dataset_dimension', 'category'), )
