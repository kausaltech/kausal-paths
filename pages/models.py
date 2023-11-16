from functools import cached_property
from typing import Sequence, cast

from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.utils.translation import gettext_lazy as _
from modelcluster.fields import ParentalKey
from wagtail import blocks
from wagtail.admin.forms import WagtailAdminPageForm
from wagtail.admin.panels import (
    FieldPanel, MultiFieldPanel, Panel,
)
from wagtail.fields import RichTextField, StreamField
from wagtail.models import Page, Site

from graphene_django.converter import convert_choices_to_named_enum_with_descriptions
from grapple.models import (
    GraphQLBoolean, GraphQLField, GraphQLStreamfield,
    GraphQLString
)
from nodes.blocks import OutcomeBlock

from nodes.models import InstanceConfig, NodeConfig


class PathsAdminPageForm(WagtailAdminPageForm):
    instance: 'PathsPage'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        process_form = getattr(self._meta.model, 'process_form', None)
        if process_form:
            process_form(self)

    @cached_property
    def admin_instance(self):
        if self.instance is not None and self.instance.id is not None:
            pp = self.instance
        else:
            pp = cast(PathsPage, self.parent_page)
        for page in pp.get_ancestors(inclusive=True).filter(depth__gte=2):
            site = page.sites_rooted_here.first()
            if site is not None:
                break
        else:
            raise Exception("No sites found for page: %s" % self.instance)
        return site.instance


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

    base_form_class = PathsAdminPageForm

    # type annotations
    id: int | None

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

        return (site.pk, instance.site_url, self.url_path)


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

    @classmethod
    def process_form(cls, form: PathsAdminPageForm):
        f = form.fields.get('outcome_node')
        if f is not None:
            f.queryset = f.queryset.filter(instance=form.admin_instance)


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
    show_action_comparison = models.BooleanField(default=True, verbose_name=_('Show action comparison'))
    show_only_municipal_actions = models.BooleanField(default=False, verbose_name=_('Show only municipal actions'))

    content_panels = PathsPage.content_panels + [
        FieldPanel('lead_title'),
        FieldPanel('lead_paragraph'),
        FieldPanel('default_sort_order'),
        FieldPanel('show_cumulative_impact'),
        FieldPanel('show_action_comparison'),
        FieldPanel('show_only_municipal_actions'),
    ]

    ActionSortOrderEnum = convert_choices_to_named_enum_with_descriptions('ActionSortOrder', choices=ActionSortOrder.choices)

    graphql_fields = PathsPage.graphql_fields + [
        GraphQLString('lead_title'),
        GraphQLString('lead_paragraph'),
        GraphQLField('default_sort_order', ActionSortOrderEnum, required=True),
        GraphQLBoolean('show_cumulative_impact'),
        GraphQLBoolean('show_action_comparison'),
        GraphQLBoolean('show_only_municipal_actions'),
    ]

    parent_page_type = [InstanceRootPage]

    class Meta:
        verbose_name = _('Action list page')
        verbose_name_plural = _('Action list pages')
