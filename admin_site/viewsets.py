from __future__ import annotations

from typing import Generic, Type, TypeVar

from django.db import models
from django.core.exceptions import FieldDoesNotExist
from wagtail.snippets.views.chooser import ChooseResultsView, ChooseView, SnippetChooserViewSet
from wagtail.snippets.views.snippets import SnippetViewSet, EditView, CreateView
from wagtail.admin.panels import Panel
from admin_site.forms import PathsAdminModelForm
from paths.admin_context import get_admin_instance

from paths.types import PathsAdminRequest


M = TypeVar('M', bound=models.Model)


class PathsEditView(EditView, Generic[M]):
    request: PathsAdminRequest
    model: Type[M]

    def get_form_kwargs(self):
        return {
            **super().get_form_kwargs(),
            'admin_instance': get_admin_instance()
        }


class PathsCreateView(CreateView, Generic[M]):
    request: PathsAdminRequest
    model: Type[M]

    def get_form_kwargs(self):
        return {
            **super().get_form_kwargs(),
            'admin_instance': get_admin_instance()
        }


class PathsChooseViewMixin(Generic[M]):
    request: PathsAdminRequest
    model: type[M]

    def get_object_list(self):
        qs = super().get_object_list()  # type: ignore
        try:
            field = self.model._meta.get_field('instance')
        except FieldDoesNotExist:
            field = None
        if field is not None:
            qs = qs.filter(instance=self.request.admin_instance)
        return qs


class PathsChooseView(PathsChooseViewMixin[M], ChooseView):
    pass


class PathsChooseResultsView(PathsChooseViewMixin[M], ChooseResultsView):
    pass


class PathsChooserViewSet(SnippetChooserViewSet, Generic[M]):
    choose_results_view_class = PathsChooseResultsView
    choose_view_class = PathsChooseView
    parent_viewset: PathsViewSet

    def __init__(self, *args, **kwargs):
        self.parent_viewset = kwargs.pop('parent_viewset')
        super().__init__(*args, **kwargs)


class PathsViewSet(SnippetViewSet, Generic[M]):
    model: Type[M]
    request: PathsAdminRequest
    add_view_class = PathsCreateView
    edit_view_class = PathsEditView
    add_to_admin_menu = True
    chooser_viewset_class = PathsChooserViewSet

    def get_queryset(self, request: PathsAdminRequest) -> models.QuerySet[M]:
        base_qs = super().get_queryset(request)
        if base_qs is None:
            qs = self.model.objects.get_queryset()  # type: ignore[attr-defined]
        else:
            qs = base_qs
        #if issubclass(self.model, PlanRelatedModel):
        #    qs = self.model.filter_by_plan(request.get_active_admin_plan(), qs)
        return qs

    def get_edit_handler(self) -> Panel:
        return super().get_edit_handler()

    def get_form_class(self, for_update: bool = False):
        if not self._edit_handler.base_form_class:
            self._edit_handler.base_form_class = PathsAdminModelForm
        return super().get_form_class(for_update)

    @property
    def chooser_viewset(self):
        return self.chooser_viewset_class(
            self.get_chooser_admin_url_namespace(),
            model=self.model,
            url_prefix=self.get_chooser_admin_base_path(),
            icon=self.icon,
            per_page=self.chooser_per_page,
            parent_viewset=self
        )
