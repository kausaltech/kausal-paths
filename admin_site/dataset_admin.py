from __future__ import annotations

from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from wagtail.snippets.models import register_snippet
from wagtail.snippets.views.snippets import SnippetViewSet, CreateView

from kausal_paths_extensions.dataset_editor import DatasetViewSet
from kausal_common.datasets.config import dataset_config
from kausal_common.datasets.models import DatasetSchema


class DatasetSchemaCreateView(CreateView):
    def get_success_url(self):
        if not dataset_config.SCHEMA_HAS_SINGLE_DATASET:
            return super().get_success_url()
        if not self.form.instance.datasets.exists():
            return super().get_success_url()
        only_dataset = self.form.instance.datasets.first()
        return reverse(DatasetViewSet().get_url_name('edit'), args=(only_dataset.pk,))


class DatasetSchemaViewSet(SnippetViewSet):
    model = DatasetSchema
    icon = 'table'
    add_to_admin_menu = dataset_config.SHOW_SCHEMAS_IN_MENU
    menu_order = 200
    menu_label = _('Dataset schemas')
    list_display = ('name_i18n',)
    search_fields = ['name_i18n']
    panels = DatasetSchema.panels
    add_view_class = DatasetSchemaCreateView


register_snippet(DatasetSchemaViewSet)
