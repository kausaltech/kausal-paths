from typing import Optional, Sequence

from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
import graphene
from modelcluster.fields import ParentalKey
from wagtail import blocks
from wagtail.admin.panels import (
    FieldPanel, MultiFieldPanel, Panel,
)
from wagtail.fields import RichTextField, StreamField
from wagtail.models import Page, Site

from graphene_django.converter import convert_choices_to_named_enum_with_descriptions
from grapple.models import (
    GraphQLBoolean, GraphQLField, GraphQLForeignKey, GraphQLImage, GraphQLStreamfield,
    GraphQLString
)
from nodes.blocks import OutcomeBlock

from nodes.node import Node
from nodes.models import InstanceConfig, NodeConfig


class PathsPage(Page):
    i18n = models.JSONField(blank=True, null=True)
    show_in_footer = models.BooleanField(default=False, verbose_name=_('show in footer'),
                                         help_text=_('Should the page be shown in the footer?'),)

    content_panels = [
        FieldPanel('title', classname="full title"),
    ]
    common_settings_panels = [
        FieldPanel('seo_title'),
        FieldPanel('show_in_menus'),
        FieldPanel('show_in_footer'),
        FieldPanel('search_description'),
    ]
    settings_panels = [
        MultiFieldPanel([
            FieldPanel('slug'),
            *common_settings_panels
        ], _('Common page configuration')),
    ]
    promote_panels: list[Panel] = []

    graphql_fields = [
        GraphQLBoolean('show_in_menus', required=True),
        GraphQLBoolean('show_in_footer', required=True),
        GraphQLString('title', required=True)
    ]

    class Meta:
        abstract = True

    @classmethod
    def get_subclasses(cls):
        """Get implementations of this abstract base class"""
        content_types = ContentType.objects.filter(app_label=cls._meta.app_label)
        model_classes = [ct.model_class() for ct in content_types]
        return [model for model in model_classes if (model is not None and issubclass(model, cls) and model is not cls)]

    def get_url_parts(self, request=None):
        # Find the root page for this sub-page
        root_page = self.get_ancestors(inclusive=True).filter(depth=2).specific().first()
        site = Site.objects.filter(root_page=root_page).first()
        instance = InstanceConfig.objects.filter(site=site).first()
        if not site or not instance:
            return super().get_url_parts(request)

        return (site.id, instance.site_url, self.url_path)


class InstanceRootPage(PathsPage):
    body = StreamField([
        ('outcome', OutcomeBlock()),
    ], block_counts={
        'outcome': {'min_num': 1, 'max_num': 1},
    }, use_json_field=True)

    content_panels = PathsPage.content_panels + [
        FieldPanel('body')
    ]

    parent_page_types: Sequence[type[Page] | str] = []


class StaticPage(PathsPage):
    body = StreamField([
        ('paragraph', blocks.RichTextBlock(label=_('Paragraph'))),
        ('outcome', OutcomeBlock()),
    ], blank=True, null=True, use_json_field=True)

    content_panels = PathsPage.content_panels + [
        FieldPanel('body')
    ]

    graphql_fields = PathsPage.graphql_fields + [
        GraphQLStreamfield('body')
    ]

class OutcomePage(PathsPage):
    outcome_node = ParentalKey(NodeConfig, on_delete=models.PROTECT, related_name='pages')
    lead_title = models.CharField(blank=True, max_length=100, verbose_name=_('Lead title'))
    lead_paragraph = RichTextField(blank=True, verbose_name=_('Lead paragraph'))

    content_panels = PathsPage.content_panels + [
        FieldPanel('outcome_node'),
        FieldPanel('lead_title'),
        FieldPanel('lead_paragraph'),
    ]

    graphql_fields = PathsPage.graphql_fields + [
        GraphQLField('outcome_node', 'nodes.schema.NodeType', required=True),  #type: ignore
        GraphQLString('lead_title'),
        GraphQLString('lead_paragraph'),
    ]

    class Meta:
        verbose_name = _('Outcome page')
        verbose_name_plural = _('Outcome pages')


class ActionListPage(PathsPage):
    class ActionSortOrder(models.TextChoices):
        STANDARD = "standard", _("Standard")
        IMPACT = "impact", _("Impact")
        CUM_IMPACT = "cum_impact", _("Cumulative impact")

    lead_title = models.CharField(blank=True, max_length=100, verbose_name=_('Lead title'))
    lead_paragraph = RichTextField(blank=True, verbose_name=_('Lead paragraph'))
    # standard, impact, cumulative impact??
    default_sort_order = models.CharField(max_length=40, choices=ActionSortOrder.choices, default=ActionSortOrder.STANDARD)
    show_cumulative_impact = models.BooleanField(default=True, verbose_name=_('Show cumulative impact'))

    content_panels = PathsPage.content_panels + [
        FieldPanel('lead_title'),
        FieldPanel('lead_paragraph'),
        FieldPanel('default_sort_order'),
        FieldPanel('show_cumulative_impact'),
    ]

    ActionSortOrderEnum = convert_choices_to_named_enum_with_descriptions('ActionSortOrder', choices=ActionSortOrder.choices)

    graphql_fields = PathsPage.graphql_fields + [
        GraphQLString('lead_title'),
        GraphQLString('lead_paragraph'),
        GraphQLField('default_sort_order', ActionSortOrderEnum, required=True),
    ]

    parent_page_type = [InstanceRootPage]

    class Meta:
        verbose_name = _('Action list page')
        verbose_name_plural = _('Action list pages')


class NodePage(Page):
    description = RichTextField()
    node = models.CharField(max_length=100, unique=True)

    content_panels = Page.content_panels + [
        FieldPanel('description', classname="full"),
        FieldPanel('node'),
    ]

    parent_page_types = ['wagtailcore.Page']
    subpage_types = []
