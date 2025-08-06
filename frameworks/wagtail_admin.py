from __future__ import annotations

from typing import TYPE_CHECKING

from django.db.models import QuerySet
from django.utils.translation import gettext_lazy as _
from wagtail import hooks
from wagtail.admin.menu import AdminOnlyMenuItem
from wagtail.admin.panels.field_panel import FieldPanel
from wagtail.admin.ui.tables import BooleanColumn
from wagtail.admin.viewsets.base import ViewSetGroup

from admin_site.forms import PathsAdminModelForm
from admin_site.viewsets import PathsViewSet

from .models import MeasureTemplate, Section, SectionQuerySet

if TYPE_CHECKING:
    from django.http.request import HttpRequest


class SectionForm(PathsAdminModelForm[Section]):
    model = Section



class SectionViewSet(PathsViewSet[Section, SectionQuerySet]):
    model = Section
    icon = 'folder-open-1'
    choose_one_text = _("Choose a section")
    choose_another_text = _("Choose another section")
    list_display = [
        'indented_name',
    ]
    panels = [
        FieldPanel('name'),
        FieldPanel('description'),
        FieldPanel('available_years'),
    ]
    menu_item_class = AdminOnlyMenuItem

    def get_queryset(self, request: HttpRequest) -> SectionQuerySet:
        return super().get_queryset(request).filter(depth__gte=2)


class MeasureTemplateViewSet(PathsViewSet[MeasureTemplate, QuerySet]):
    model = MeasureTemplate
    icon = 'kausal-plans'
    form_fields = ['name', 'priority', 'min_value', 'max_value', 'hidden']
    menu_label = _("Measure templates")
    list_display = [
        'name',
        'section',
        'unit',
        'uuid',
        'priority',
        BooleanColumn('hidden'),
    ]
    list_filter = ['section__framework', 'section', 'priority', 'hidden']
    search_fields = ['name', 'section__name', 'uuid']
    inspect_view_enabled = False
    copy_view_enabled = False

    menu_item_class = AdminOnlyMenuItem

    panels = [
        FieldPanel('name'),
        FieldPanel('section'),
        FieldPanel('priority'),
        FieldPanel('min_value'),
        FieldPanel('max_value'),
        FieldPanel('hidden'),
    ]


class FrameworksViewSetGroup(ViewSetGroup):
    menu_label = "Frameworks"
    menu_icon = "folder-inverse"
    menu_order = 200
    items = (SectionViewSet(), MeasureTemplateViewSet())


frameworks_viewset_group = FrameworksViewSetGroup()
hooks.register('register_admin_viewset', lambda: frameworks_viewset_group)
