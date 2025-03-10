from __future__ import annotations

from django.utils.translation import gettext_lazy as _
from wagtail.snippets.models import register_snippet
from wagtail.snippets.views.snippets import SnippetViewSet

from kausal_common.datasets.config import dataset_config
from kausal_common.datasets.models import DatasetSchema


class DatasetSchemaViewSet(SnippetViewSet):
    model = DatasetSchema
    icon = 'table'
    add_to_admin_menu = dataset_config.SHOW_SCHEMAS_IN_MENU
    menu_order = 200
    menu_label = _('Dataset schemas')
    list_display = ('name_i18n',)
    search_fields = ['name_i18n']
    panels = DatasetSchema.panels


register_snippet(DatasetSchemaViewSet)
