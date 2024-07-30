from wagtail import hooks
from django.db.models import QuerySet

from admin_site.viewsets import PathsViewSet
from paths.types import PathsAdminRequest
from .models import MeasureTemplate


class MeasureTemplateViewSet(PathsViewSet[MeasureTemplate, QuerySet]):
    model = MeasureTemplate
    icon = "chart"  # Assuming there's a suitable icon in Wagtail's icon library
    add_to_admin_menu = True
    menu_label = "Measure Templates"
    menu_order = 200  # Adjust as needed
    list_display = ['name', 'section', 'unit', 'priority']
    list_filter = ['section', 'priority']
    search_fields = ['name', 'section__name']
    form_fields = ['section', 'name', 'unit', 'priority', 'min_value', 'max_value', 'time_series_max', 'default_value_source']
    inspect_view_enabled = True
    inspect_view_fields = ['name', 'section', 'unit', 'priority', 'min_value', 'max_value', 'time_series_max', 'default_value_source']
    copy_view_enabled = False

    #def get_queryset(self, request: PathsAdminRequest):
    #    return super().get_queryset(request).select_related('section')

measure_template_viewset = MeasureTemplateViewSet(name="measure_templates")
hooks.register('register_admin_viewset', lambda: measure_template_viewset)
