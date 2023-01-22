from wagtail.contrib.modeladmin.options import (
    ModelAdmin, modeladmin_register
)
from .models import Dimension


class DimensionAdmin(ModelAdmin):
    model = Dimension
    menu_label = 'Dimensions'
    menu_icon = 'pilcrow'
    menu_order = 200
    add_to_settings_menu = True
    list_display = ('label',)


modeladmin_register(DimensionAdmin)
