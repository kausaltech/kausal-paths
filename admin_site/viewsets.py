from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Generic, TypeVar, cast

from django.core.exceptions import FieldDoesNotExist
from django.db.models import Model, QuerySet
from django.forms import BaseModelForm
from wagtail.admin.forms.models import WagtailAdminModelForm
from wagtail.snippets.views.chooser import ChooseResultsView, ChooseView, SnippetChooserViewSet
from wagtail.snippets.views.snippets import CreateView, EditView, SnippetViewSet

from paths.admin_context import get_admin_instance
from paths.types import PathsAdminRequest, PathsModel

from admin_site.forms import PathsAdminModelForm
from users.models import User

if TYPE_CHECKING:
    from django.http import HttpRequest
    from wagtail.admin.panels.group import ObjectList

M = TypeVar('M', bound=Model)
QS = TypeVar('QS', bound=QuerySet)
MF = TypeVar('MF', bound=BaseModelForm)


def admin_req(request: HttpRequest) -> PathsAdminRequest:
    assert request.user is not None
    assert request.user.is_authenticated
    return cast(PathsAdminRequest, request)


class PathsEditView(EditView[M]):
    def get_form_kwargs(self):
        return {
            **super().get_form_kwargs(),
            'admin_instance': get_admin_instance(),
        }


class PathsCreateView(CreateView, Generic[M]):
    def get_form_kwargs(self):
        return {
            **super().get_form_kwargs(),
            'admin_instance': get_admin_instance(),
        }


class PathsChooseViewMixin(Generic[M]):
    model: type[M]
    request: HttpRequest

    def get_object_list(self):
        qs: QuerySet[M] = super().get_object_list()  # type: ignore
        try:
            field = self.model._meta.get_field('instance')
        except FieldDoesNotExist:
            field = None
        if field is not None:
            qs = qs.filter(instance=admin_req(self.request).admin_instance)
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


class PathsModelForm[M: Model](WagtailAdminModelForm[M, User]):
    pass


class PathsViewSet[M: Model | PathsModel](SnippetViewSet[M, PathsModelForm[M]]):
    add_view_class = PathsCreateView
    edit_view_class = PathsEditView
    add_to_admin_menu = True
    chooser_viewset_class = PathsChooserViewSet

    @cached_property
    def url_prefix(self) -> str:
        return f"{self.app_label}/{self.model_name}"

    @cached_property
    def url_namespace(self) -> str:
        return f"{self.app_label}_{self.model_name}"

    @property
    def permission_policy(self):
        if issubclass(self.model, PathsModel):
            return self.model.permission_policy()
        return super().permission_policy

    def get_queryset(self, request: HttpRequest) -> QuerySet[M, M]:
        qs = cast(QuerySet[M, M], super().get_queryset(request))
        if issubclass(self.model, PathsModel):
            pass
        return qs

    def get_edit_handler(self) -> ObjectList | None:
        return super().get_edit_handler()

    def get_form_class(self, for_update: bool = False):
        if self._edit_handler and not self._edit_handler.base_form_class:
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
