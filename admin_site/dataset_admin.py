from __future__ import annotations

from typing import TYPE_CHECKING

from django.forms import BaseInlineFormSet, ValidationError
from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from wagtail.admin.forms.models import WagtailAdminModelForm
from wagtail.snippets.models import register_snippet

from kausal_paths_extensions.dataset_editor import DatasetViewSet

from kausal_common.datasets.config import dataset_config
from kausal_common.datasets.models import (
    DataPoint,
    DataPointDimensionCategory,
    Dataset,
    DatasetSchema,
    DatasetSchemaScope,
    Dimension,
    DimensionScope,
)

from paths.context import realm_context

from admin_site.viewsets import PathsCreateView, PathsEditView, PathsViewSet
from users.models import User

if TYPE_CHECKING:
    from nodes.models import InstanceConfig


class DatasetSchemaCreateView(PathsCreateView[DatasetSchema, WagtailAdminModelForm[DatasetSchema, User]]):
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

    def clean(self):
        super().clean()
        for form in self.forms:
            if form.instance and form.instance.pk and form not in self.deleted_forms and 'dimension' in form.changed_data:
                old_dimension_pk = form.initial.get('dimension')
                new_dimension = form.cleaned_data.get('dimension')

                if old_dimension_pk and new_dimension and old_dimension_pk != new_dimension.pk:

                    old_dimension = Dimension.objects.get(pk=old_dimension_pk)

                    data_point_count = DataPointDimensionCategory.objects.filter(
                        dimension_category__dimension=old_dimension,
                        data_point__dataset__schema=form.instance.schema,
                    ).count()

                    if data_point_count > 0:
                        form.add_error('dimension', _(
                            'Cannot change from "%(old)s" because it is used by %(count)d data point(s).'
                        ) % {
                            'old': str(old_dimension),
                            'count': data_point_count,
                        })

        errors = []
        for form in self.deleted_forms:
            if form.instance and form.instance.pk:
                dimension_pk = form.initial.get('dimension')
                if dimension_pk:
                    dimension = Dimension.objects.get(pk=dimension_pk)
                    schema = form.instance.schema

                    data_point_count = DataPointDimensionCategory.objects.filter(
                        dimension_category__dimension=dimension,
                        data_point__dataset__schema=schema,
                    ).count()

                    if data_point_count > 0:
                        errors.append(_(
                            'Cannot remove dimension "%(dimension)s" because it is used by %(count)d data point(s). '
                            'Please remove the data first.'
                        ) % {
                            'dimension': str(dimension),
                            'count': data_point_count,
                        })

        if errors:
            raise ValidationError(errors)


class DatasetSchemaMetricFormSet(BaseInlineFormSet):
    def clean(self):
        super().clean()

        errors = []

        # Check that at least one metric is present
        valid_forms = [
            form for form in self.forms
            if form not in self.deleted_forms and not form.cleaned_data.get('DELETE', False)
        ]
        non_empty_forms = [
            form for form in valid_forms
            if form.cleaned_data and not form.cleaned_data.get('DELETE', False)
        ]
        if len(non_empty_forms) == 0:
            raise ValidationError(
                _('At least one metric must be defined for the dataset schema.')
            )

        for form in self.deleted_forms:
            if form.instance and form.instance.pk:
                metric = form.instance

                data_point_count = DataPoint.objects.filter(metric=metric).count()

                if data_point_count > 0:
                    errors.append(_(
                        'Cannot remove metric "%(metric)s" because it is used by %(count)d data point(s). '
                        'Please remove the data first.'
                    ) % {
                        'metric': str(metric),
                        'count': data_point_count,
                    })

        if errors:
            raise ValidationError(errors)


class DatasetSchemaEditView(PathsEditView):
    def get_success_url(self):
        return reverse(DatasetViewSet().get_url_name('list'))


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
    edit_view_class = DatasetSchemaEditView

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

                if 'metrics' in formsets:
                    formsets['metrics']['formset'] = DatasetSchemaMetricFormSet

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                if self.instance and self.instance.pk:  # Only disable in edit mode
                    if 'start_date' in self.fields:
                        self.fields['start_date'].disabled = True
                    if 'time_resolution' in self.fields:
                        self.fields['time_resolution'].disabled = True

        return DatasetSchemaWithDimensionForm

register_snippet(DatasetSchemaViewSet)
