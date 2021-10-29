from typing import Optional

from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from modelcluster.fields import ParentalKey
from wagtail.admin.edit_handlers import (
    FieldPanel, MultiFieldPanel, StreamFieldPanel
)
from wagtail.core.fields import RichTextField
from wagtail.core.models import Page, Site

from grapple.models import (
    GraphQLBoolean, GraphQLField, GraphQLForeignKey, GraphQLImage, GraphQLStreamfield,
    GraphQLString
)

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
    promote_panels = []

    graphql_fields = [
        GraphQLBoolean('show_in_footer'),
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


class OutcomePage(PathsPage):
    outcome_node = ParentalKey(NodeConfig, on_delete=models.PROTECT, related_name='pages')
    lead_title = models.CharField(blank=True, max_length=100, verbose_name=_('Lead title'))
    lead_paragraph = RichTextField(null=True, blank=True, verbose_name=_('Lead paragraph'))

    content_panels = PathsPage.content_panels + [
        FieldPanel('outcome_node'),
        FieldPanel('lead_title'),
        FieldPanel('lead_paragraph'),
    ]
    graphql_fields = PathsPage.graphql_fields + [
        # FIXME how to resolve
        GraphQLField('outcome_node', 'nodes.schema.NodeType', required=True),
        GraphQLString('lead_title'),
        GraphQLString('lead_paragraph'),
    ]

    class Meta:
        verbose_name = _('Outcome page')
        verbose_name_plural = _('Outcome pages')


class ActionListPage(PathsPage):
    lead_title = models.CharField(blank=True, max_length=100, verbose_name=_('Lead title'))
    lead_paragraph = RichTextField(null=True, blank=True, verbose_name=_('Lead paragraph'))

    content_panels = PathsPage.content_panels + [
        FieldPanel('lead_title'),
        FieldPanel('lead_paragraph'),
    ]
    graphql_fields = PathsPage.graphql_fields + [
        GraphQLString('lead_title'),
        GraphQLString('lead_paragraph'),
    ]

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
