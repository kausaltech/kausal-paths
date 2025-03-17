from __future__ import annotations

from typing import TYPE_CHECKING

from django.forms import BaseInlineFormSet
from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from wagtail.admin.forms.models import WagtailAdminModelForm
from wagtail.snippets.models import register_snippet
from wagtail.snippets.views.snippets import CreateView

from kausal_common.datasets.config import dataset_config
from kausal_common.datasets.models import (
    Dataset,
    DatasetSchema,
    DatasetSchemaScope,
    DimensionScope,
)

from paths.context import realm_context

from admin_site.viewsets import PathsViewSet
from kausal_paths_extensions.dataset_editor import DatasetViewSet
from users.models import User

if TYPE_CHECKING:
    from nodes.models import InstanceConfig


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
        callback = getattr(dataset_config, 'SCHEMA_DEFAULT_SCOPE_FUNCTION', None)
        if callback is not None:
            default_scope_model_instance = callback()
            DatasetSchemaScope.objects.create(
                schema=instance,
                scope=default_scope_model_instance
            )
            if dataset_config.SCHEMA_HAS_SINGLE_DATASET:
                Dataset.objects.get_or_create(schema=instance, defaults={'scope': default_scope_model_instance})
        return instance

class DatasetSchemaFormWithDimensionFormSet(BaseInlineFormSet):
    active_instance: InstanceConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_instance = realm_context.get().realm

    def get_queryset(self):
        queryset = super().get_queryset()
        return queryset.filter(
            dimension__scopes__in=DimensionScope.objects.get_queryset().for_instance_config(self.active_instance)
        )

    def add_fields(self, form, index):
        super().add_fields(form, index)
        if 'dimension' in form.fields:
            form.fields['dimension'].queryset = form.fields['dimension'].queryset.filter(
                scopes__in=DimensionScope.objects.get_queryset().for_instance_config(self.active_instance)
            )

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

    def get_form_class(self, for_update=False):
        form_class = super().get_form_class(for_update)
        class DatasetSchemaWithDimensionForm(form_class):  # type: ignore[valid-type, misc]
            class Meta(form_class.Meta):
                model = DatasetSchema
                fields = form_class.Meta.fields
                formsets = getattr(form_class.Meta, 'formsets', {}).copy()
                formsets.update({
                    'dimensions': {
                        'formset': DatasetSchemaFormWithDimensionFormSet,
                        'fields': ['dimension'],
                        'min_num': 0,
                        'validate_min': False,
                        'can_order': False,
                    }
                })

        return DatasetSchemaWithDimensionForm

register_snippet(DatasetSchemaViewSet)
