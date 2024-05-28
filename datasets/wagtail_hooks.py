from django.utils.translation import gettext_lazy as _
from wagtail.snippets.models import register_snippet

from admin_site.viewsets import PathsViewSet
from nodes.models import DataSource

from .models import Dimension


class DimensionViewSet(PathsViewSet):
    model = Dimension
    menu_label = _('Data dimensions')
    icon = 'kausal-dimensions'
    menu_order = 10
    add_to_settings_menu = True
    list_display = ('label',)

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        qs = qs.filter(instance=request.admin_instance)
        return qs


class DataSourceViewSet(PathsViewSet):
    model = DataSource
    menu_label = _('Data sources')
    icon = 'doc-full'
    menu_order = 11
    add_to_settings_menu = True

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        qs = qs.filter(instance=request.admin_instance)
        return qs


register_snippet(DataSourceViewSet)
register_snippet(DimensionViewSet)
