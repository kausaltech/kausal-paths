from __future__ import annotations

from typing import ClassVar, Generic, Type, TypeVar

from django.db import models
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


class PathsViewSet(SnippetViewSet, Generic[M]):
    model: Type[M]
    request: PathsAdminRequest
    add_view_class = PathsCreateView
    edit_view_class = PathsEditView
    add_to_admin_menu = True

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