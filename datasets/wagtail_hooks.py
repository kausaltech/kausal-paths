from django.utils.translation import gettext_lazy as _

from wagtail.contrib.modeladmin.options import (
    ModelAdmin, modeladmin_register
)
from .models import Dimension


class DimensionAdmin(ModelAdmin):
    model = Dimension
    menu_label = _('Dimensions')
    menu_icon = 'table'
    menu_order = 10
    add_to_settings_menu = True
    list_display = ('label',)


modeladmin_register(DimensionAdmin)
