from __future__ import annotations

from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from wagtail.admin.forms.models import WagtailAdminModelForm
from wagtail.snippets.models import register_snippet
from wagtail.snippets.views.snippets import CreateView

from kausal_common.datasets.config import dataset_config
from kausal_common.datasets.models import DatasetSchema, DatasetSchemaScope

from admin_site.viewsets import PathsViewSet
from kausal_paths_extensions.dataset_editor import DatasetViewSet
from users.models import User


class DatasetSchemaCreateView(CreateView[DatasetSchema, WagtailAdminModelForm[DatasetSchema, User]]):
    def get_success_url(self):
        if not dataset_config.SCHEMA_HAS_SINGLE_DATASET:
            return super().get_success_url()
        if not self.form.instance.datasets.exists():
            return super().get_success_url()
        only_dataset = self.form.instance.datasets.first()
        if only_dataset is None:
            return super().get_success_url()
        return reverse(DatasetViewSet().get_url_name('edit'), args=(only_dataset.pk,))

    def save_instance(self) -> DatasetSchema:
        instance = super().save_instance()
        callback = dataset_config.SCHEMA_DEFAULT_SCOPE_FUNCTION
        if callback is not None:
            default_scope_model_instance = callback()
            DatasetSchemaScope.objects.create(
                schema=instance,
                scope=default_scope_model_instance
            )
        return instance


class DatasetSchemaViewSet(PathsViewSet):
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
