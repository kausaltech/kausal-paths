from django.utils.translation import gettext_lazy as _

from wagtail_modeladmin.options import (
    ModelAdmin, modeladmin_register
)
from .models import Dimension
from nodes.models import DataSource


class DimensionAdmin(ModelAdmin):
    model = Dimension
    menu_label = _('Data dimensions')
    menu_icon = 'kausal-dimensions'
    menu_order = 10
    add_to_settings_menu = True
    list_display = ('label',)

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        qs = qs.filter(instance=request.admin_instance)
        return qs


class DataSourceAdmin(ModelAdmin):
    model = DataSource
    menu_label = _('Data sources')
    menu_icon = 'doc-full'
    menu_order = 11
    add_to_settings_menu = True

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        qs = qs.filter(instance=request.admin_instance)
        return qs


modeladmin_register(DataSourceAdmin)
modeladmin_register(DimensionAdmin)
