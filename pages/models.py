from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, Self, cast

from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.utils.translation import gettext_lazy as _
from graphene_django.converter import convert_choices_to_named_enum_with_descriptions
from modelcluster.fields import ParentalKey
from wagtail import blocks
from wagtail.admin.forms.pages import WagtailAdminPageForm
from wagtail.admin.panels import (
    FieldPanel,
    MultiFieldPanel,
)
from wagtail.fields import RichTextField, StreamField, StreamValue
from wagtail.models import Page, Site
from wagtail.query import PageQuerySet

from grapple.models import GraphQLBoolean, GraphQLField, GraphQLStreamfield, GraphQLString
from wagtail_color_panel.edit_handlers import NativeColorPanel
from wagtail_color_panel.fields import ColorField

from kausal_common.models.types import PageModelManager

from nodes.blocks import OutcomeBlock
from nodes.models import InstanceConfig, NodeConfig
from pages.blocks import DashboardCardBlock

if TYPE_CHECKING:
    from collections.abc import Sequence

    from django.db.models.expressions import Combinable
    from django.db.models.fields import AutoField
    from django.forms import ModelChoiceField
    from modelcluster.fields import PK
    from wagtail.admin.panels import (
        Panel,
    )

    from paths.context import InstanceSpecificCache


class PathsAdminPageForm(WagtailAdminPageForm):
    instance: PathsPage

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        process_form = getattr(self._meta.model, 'process_form', None)
        if process_form:
            process_form(self)

    @cached_property
    def admin_instance(self) -> InstanceConfig:
        if self.instance is not None and self.instance.id is not None:
            pp = self.instance
        else:
            pp = cast('PathsPage', self.parent_page)
        for page in pp.get_ancestors(inclusive=True).filter(depth__gte=2):
            site = page.sites_rooted_here.first()
            if site is not None:
                break
        else:
            raise Exception('No sites found for page: %s' % self.instance)

        return cast('InstanceConfig', site.instance)  # pyright: ignore


class PathsPageManager[PageT: PathsPage](PageModelManager[PageT, PageQuerySet[PageT]]):
    model: type[PageT]

    def get_queryset(self) -> PageQuerySet[PageT]:
        return super().get_queryset()

    def get_by_natural_key(self, *slugs: str) -> PageT:
        next_level = Page.get_root_nodes()
        page = None
        rest = list(slugs)
        while rest:
            slug, *rest = rest
            page = next_level.get(slug=slug)
            next_level = page.get_children()
        assert page
        page = cast('PageT', page.specific)
        assert isinstance(page, self.model)
        return page


class PathsPage(Page):
    i18n = models.JSONField(blank=True, null=True)
    show_in_footer = models.BooleanField[bool, bool](
        default=False,
        verbose_name=_('show in footer'),
        help_text=_('Should the page be shown in the footer?'),
    )
    show_in_additional_links = models.BooleanField(
        default=False,
        verbose_name=_('show in additional links'),
        help_text=_('Should the page be shown in the additional links?'),
    )

    content_panels: Sequence[Panel] = [
        FieldPanel('title', classname='full title'),
    ]
    common_settings_panels = [
        FieldPanel('seo_title'),
        FieldPanel('show_in_menus'),
        FieldPanel('show_in_footer'),
        FieldPanel('show_in_additional_links'),
        FieldPanel('search_description'),
    ]
    settings_panels = [
        MultiFieldPanel(
            [
                FieldPanel('slug'),
                *common_settings_panels,
            ],
            _('Common page configuration'),
        ),
    ]
    promote_panels: Sequence[Panel] = []

    graphql_fields = [
        GraphQLBoolean('show_in_menus', required=True),
        GraphQLBoolean('show_in_footer', required=True),
        GraphQLBoolean('show_in_additional_links', required=True),
        GraphQLString('title', required=True),
    ]

    base_form_class = PathsAdminPageForm

    objects: ClassVar[PathsPageManager[Self]] = PathsPageManager()
    id: AutoField[Combinable | int | str | None, int]

    class Meta:
        abstract = True

    @classmethod
    def get_subclasses(cls) -> Sequence[type[Self]]:
        """Get implementations of this abstract base class."""
        content_types = ContentType.objects.filter(app_label=cls._meta.app_label)
        model_classes = [ct.model_class() for ct in content_types]
        return tuple(model for model in model_classes if (model is not None and issubclass(model, cls) and model is not cls))

    def get_url_parts(self, request=None):
        # Find the root page for this sub-page
        root_page = self.get_ancestors(inclusive=True).filter(depth=2).specific().first()
        site = Site.objects.filter(root_page=root_page).first()
        instance = InstanceConfig.objects.filter(site=site).first()
        if not site or not instance:
            return super().get_url_parts(request)

        return (site.pk, instance.site_url, self.url_path)

    def natural_key(self):
        page: Page | None = self
        key: tuple[str, ...] = ()
        while page:
            key = (page.slug,) + key
            page = page.get_parent()
        return key

    def get_visible_parent(self, cache: InstanceSpecificCache) -> PathsPage | None:
        depth = int(len(self.path) / self.steplen)
        if cache.translated_root_page is None:
            return None
        if depth <= cache.translated_root_page.depth:
            return None

        parent_path = self._get_basepath(self.path, depth=depth - 1)
        for page in cache.visible_pages:
            if page.path == parent_path:
                return page
        return None

    def get_visible_descendants(self, cache: InstanceSpecificCache, in_menu: bool = False, max_depth: int | None = None):
        descendants: list[PathsPage] = []
        for page in cache.visible_pages:
            if page.depth <= self.depth:
                continue
            if not page.path.startswith(self.path):
                continue
            if in_menu and not page.show_in_menus:
                continue
            if max_depth is not None and page.depth > self.depth + max_depth:
                continue
            descendants.append(page)
        return descendants

    def get_visible_children(self, cache: InstanceSpecificCache, in_menu: bool = False):
        return self.get_visible_descendants(cache, in_menu, max_depth=1)

    def get_visible_ancestors(self, cache: InstanceSpecificCache):
        ancestors: list[PathsPage] = []
        for page in cache.visible_pages:
            if page.depth >= self.depth:
                continue
            if not self.path.startswith(page.path):
                continue
            ancestors.append(page)
        return ancestors

    @cached_property
    def instance_config(self):
        root: PathsPage | None = None
        if self.depth == 2:
            root = cast('PathsPage', self)
        else:
            root = cast('PathsPage | None', self.get_ancestors(inclusive=False).filter(depth=2).specific().first())
            assert root is not None
        site = Site.objects.filter(root_page__translation_key=root.translation_key).first()
        assert site is not None
        return getattr(site, 'instance')  # noqa: B009

    def __rich_repr__(self):
        yield self.title
        yield 'id', self.pk
        yield 'instance', self.instance_config.identifier
        yield 'path', self.path


class InstanceRootPage(PathsPage):
    body = StreamField(
        [
            ('outcome', OutcomeBlock()),
        ],
        block_counts={
            'outcome': {'min_num': 1, 'max_num': 1},
        },
    )

    content_panels = [
        *PathsPage.content_panels,
        FieldPanel('body'),
    ]

    parent_page_types: Sequence[type[Page] | str] = []


class StaticPage(PathsPage):
    body = StreamField[StreamValue | None](
        [
            ('paragraph', blocks.RichTextBlock(label=_('Paragraph'))),
            ('outcome', OutcomeBlock()),
        ],
        blank=True,
        null=True,
    )

    content_panels = [
        *PathsPage.content_panels,
        FieldPanel('body'),
    ]

    graphql_fields = PathsPage.graphql_fields + [
        GraphQLStreamfield('body'),
    ]


class OutcomePageQuerySet(PageQuerySet['OutcomePage']):
    pass


class OutcomePageManager(PathsPageManager['OutcomePage']):
    pass


class OutcomePage(PathsPage):
    outcome_node: PK[NodeConfig] = ParentalKey(NodeConfig, on_delete=models.PROTECT, related_name='pages')
    lead_title = models.CharField(blank=True, max_length=100, verbose_name=_('Lead title'))
    lead_paragraph = RichTextField(blank=True, verbose_name=_('Lead paragraph'))

    objects: ClassVar[OutcomePageManager] = OutcomePageManager()
    _default_manager: ClassVar[OutcomePageManager]

    content_panels = [
        *PathsPage.content_panels,
        FieldPanel('outcome_node'),
        FieldPanel('lead_title'),
        FieldPanel('lead_paragraph'),
    ]

    graphql_fields = PathsPage.graphql_fields + [
        GraphQLField('outcome_node', 'nodes.schema.NodeType', required=True),
        GraphQLString('lead_title'),
        GraphQLString('lead_paragraph'),
    ]

    class Meta:
        verbose_name = _('Outcome page')
        verbose_name_plural = _('Outcome pages')

    @classmethod
    def process_form(cls, form: PathsAdminPageForm) -> None:
        f = cast('ModelChoiceField[NodeConfig] | None', form.fields.get('outcome_node'))
        if f is not None:
            qs = cast('models.QuerySet[NodeConfig]', f.queryset)
            f.queryset = qs.filter(instance=form.admin_instance)


class ActionListPage(PathsPage):
    class ActionSortOrder(models.TextChoices):
        STANDARD = 'standard', _('Standard')
        IMPACT = 'impact', _('Impact')
        CUM_IMPACT = 'cum_impact', _('Cumulative impact')

    lead_title = models.CharField[str, str](verbose_name=_('Lead title'), blank=True, max_length=100)
    lead_paragraph = RichTextField(blank=True, verbose_name=_('Lead paragraph'))
    # standard, impact, cumulative impact??
    default_sort_order = models.CharField(max_length=40, choices=ActionSortOrder.choices, default=ActionSortOrder.STANDARD)
    show_cumulative_impact = models.BooleanField(default=True, verbose_name=_('Show cumulative impact'))
    show_action_comparison = models.BooleanField(default=True, verbose_name=_('Show action comparison'))
    show_only_municipal_actions = models.BooleanField(default=False, verbose_name=_('Show only municipal actions'))

    content_panels = [
        *PathsPage.content_panels,
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


class DashboardPage(PathsPage):
    lead_title = models.CharField(blank=True, max_length=100, verbose_name=_('Lead title'))
    lead_paragraph = RichTextField(blank=True, verbose_name=_('Lead paragraph'))
    background_color = ColorField(blank=True, verbose_name=_('Background color'))
    dashboard_cards = StreamField(block_types=[('card', DashboardCardBlock())])

    content_panels = [
        *PathsPage.content_panels,
        FieldPanel('lead_title'),
        FieldPanel('lead_paragraph'),
        NativeColorPanel('background_color'),
        FieldPanel('dashboard_cards'),
    ]

    graphql_fields = PathsPage.graphql_fields + [
        GraphQLString('lead_title'),
        GraphQLString('lead_paragraph'),
        GraphQLString('background_color'),
        GraphQLStreamfield('dashboard_cards'),
    ]

    parent_page_type = [InstanceRootPage]

    class Meta:
        verbose_name = _('Dashboard page')
        verbose_name_plural = _('Dashboard pages')


class InstanceSiteContentManager(models.Manager['InstanceSiteContent']):
    def get_by_natural_key(self, instance_identifier):
        instance = InstanceConfig.objects.get_by_natural_key(instance_identifier)
        return self.get(instance=instance)


class InstanceSiteContent(models.Model):
    instance = models.OneToOneField(InstanceConfig, on_delete=models.CASCADE, related_name='site_content')

    intro_content = StreamField(
        block_types=[
            ('title', blocks.RichTextBlock(label=_('Title'), features=['h2', 'h3', 'h4', 'bold', 'italic'])),
            (
                'paragraph',
                blocks.RichTextBlock(
                    label=_('Introductory content to show in the UI'),
                    features=['h2', 'h3', 'h4', 'bold', 'italic', 'embed'],
                ),
            ),
        ],
        block_counts={'title': {'max_num': 1}, 'paragraph': {'max_num': 1}},
        blank=True,
        verbose_name=_('Introductory content'),
    )

    graphql_fields = [GraphQLStreamfield('intro_content')]

    objects = InstanceSiteContentManager()

    class Meta:
        verbose_name = _('Site content')
        verbose_name_plural = _('Site contents')
        ordering = ['instance', 'id']

    def __str__(self) -> str:
        return 'Site contents for %s' % self.instance.name

    def natural_key(self):
        return self.instance.natural_key()
