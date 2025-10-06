from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Generic, cast, override
from typing_extensions import TypeVar

from django.contrib.auth.models import AbstractBaseUser, AnonymousUser
from django.core.exceptions import FieldDoesNotExist
from django.db.models import Model, QuerySet
from django.forms import BaseModelForm
from django.utils.translation import gettext_lazy as _
from wagtail.admin.forms.models import WagtailAdminModelForm
from wagtail.admin.ui.tables import BulkActionsCheckboxColumn
from wagtail.snippets.views.chooser import ChooseResultsView, ChooseView, SnippetChooserViewSet
from wagtail.snippets.views.snippets import (
    CopyView,
    DeleteView,
    EditView,
    HistoryView,
    IndexView,
    InspectView,
    SnippetViewSet,
    UsageView,
)

from kausal_common.admin_site.mixins import HideSnippetsFromBreadcrumbsMixin
from kausal_common.admin_site.permissioned_views import PermissionedCreateView
from kausal_common.models.permission_policy import ModelPermissionPolicy
from kausal_common.models.permissions import PermissionedModel

from admin_site.forms import PathsAdminModelForm
from users.models import User

if TYPE_CHECKING:
    from django.http import HttpRequest
    from wagtail.permission_policies.base import BasePermissionPolicy

    from kausal_common.users import UserOrAnon

    from paths.types import PathsAdminRequest

    from nodes.models import InstanceConfig

_ModelT = TypeVar('_ModelT', bound=Model, default=Model, covariant=True)  # noqa: PLC0105
_QS = TypeVar('_QS', bound=QuerySet[Any, Any], default=QuerySet[_ModelT, _ModelT])


def admin_req(request: HttpRequest) -> PathsAdminRequest:
    assert request.user is not None
    assert request.user.is_authenticated
    return cast('PathsAdminRequest', request)


class PathsModelForm[M: Model](WagtailAdminModelForm[M, User]):
    pass

_FormT = TypeVar('_FormT', bound=BaseModelForm[Any], default=PathsModelForm[_ModelT])


class AdminInstanceMixin:
    @property
    def admin_instance(self) -> InstanceConfig:
        from paths.context import realm_context
        return realm_context.get().realm


def user_has_permission(
    permission_policy: BasePermissionPolicy,
    user: AbstractBaseUser | AnonymousUser,
    permission: str,
    obj: Model
) -> bool:
    assert isinstance(permission_policy, ModelPermissionPolicy)
    if isinstance(user, AnonymousUser):
        return False
    return permission_policy.user_has_permission_for_instance(
        cast('UserOrAnon', user), permission, obj
    )


class PathsEditView(HideSnippetsFromBreadcrumbsMixin, EditView[_ModelT, _FormT], AdminInstanceMixin):
    def user_has_permission(self, permission):
        return user_has_permission(
            self.permission_policy,
            self.request.user,
            permission,
            self.object
        )

    def get_editing_sessions(self):
        return None

    def get_form_kwargs(self):
        return {
            **super().get_form_kwargs(),
            'admin_instance': self.admin_instance,
        }


class PathsDeleteView(DeleteView[_ModelT, _FormT], AdminInstanceMixin):
    def user_has_permission(self, permission):
        return user_has_permission(
            self.permission_policy,
            self.request.user,
            permission,
            self.object
        )


class PathsCreateView(PermissionedCreateView, AdminInstanceMixin):
    def get_form_kwargs(self):
        return {
            **super().get_form_kwargs(),
            'admin_instance': self.admin_instance,
        }

    @override
    def get_create_context(self) -> InstanceConfig:
        return self.admin_instance


class PathsIndexView(HideSnippetsFromBreadcrumbsMixin, IndexView[_ModelT, _QS]):
    def user_can_change_or_delete_model(self) -> bool:
        return self.permission_policy.user_has_any_permission(self.request.user, ('delete', 'change'))

    @cached_property
    def columns(self):
        columns = super().columns
        if self.user_can_change_or_delete_model():
            return columns
        return [c for c in columns if not isinstance(c, BulkActionsCheckboxColumn)]

    def get_list_more_buttons(self, instance):
        buttons = super().get_list_more_buttons(instance)

        # Change "Inspect" to "View"
        for button in buttons:
            if hasattr(button, 'label') and button.label == _("Inspect"):
                button.label = _("View")
                if 'aria-label' in button.attrs:
                    button.attrs['aria-label'] = _("View '%(title)s'") % {"title": str(instance)}
                button.priority = 5
                break

        return buttons


class PathsUsageView(HideSnippetsFromBreadcrumbsMixin, UsageView):
    pass


class PathsHistoryView(HideSnippetsFromBreadcrumbsMixin, HistoryView):
    pass


class PathsCopyView(HideSnippetsFromBreadcrumbsMixin, CopyView[_ModelT, _FormT]):
    pass


class PathsInspectView(HideSnippetsFromBreadcrumbsMixin, InspectView):
    page_title = _("View")


class PathsChooseViewMixin(Generic[_ModelT], AdminInstanceMixin):
    model: type[_ModelT]
    request: HttpRequest

    def get_object_list(self):
        qs: QuerySet[_ModelT] = super().get_object_list()  # type: ignore
        try:
            field = self.model._meta.get_field('instance')
        except FieldDoesNotExist:
            field = None
        if field is not None:
            qs = qs.filter(instance=self.admin_instance)
        return qs


class PathsChooseView(PathsChooseViewMixin[_ModelT], ChooseView):
    pass


class PathsChooseResultsView(PathsChooseViewMixin[_ModelT], ChooseResultsView):
    pass


class PathsChooserViewSet(SnippetChooserViewSet, Generic[_ModelT]):
    choose_results_view_class = PathsChooseResultsView
    choose_view_class = PathsChooseView
    parent_viewset: PathsViewSet

    def __init__(self, *args, **kwargs):
        self.parent_viewset = kwargs.pop('parent_viewset')
        super().__init__(*args, **kwargs)


class PathsViewSet(Generic[_ModelT, _QS, _FormT], SnippetViewSet[_ModelT, _FormT]):
    index_view_class: ClassVar = PathsIndexView[_ModelT, _QS]
    add_view_class: ClassVar = PathsCreateView[_ModelT, _FormT]
    edit_view_class: ClassVar = PathsEditView[_ModelT, _FormT]
    delete_view_class: ClassVar = PathsDeleteView
    usage_view_class: ClassVar = PathsUsageView
    history_view_class: ClassVar = PathsHistoryView
    copy_view_class: ClassVar = PathsCopyView
    inspect_view_class: ClassVar = PathsInspectView

    add_to_admin_menu = True
    chooser_viewset_class = PathsChooserViewSet

    @property
    def admin_instance(self) -> InstanceConfig:
        from paths.context import realm_context
        return realm_context.get().realm

    @cached_property[str]
    def url_prefix(self) -> str:
        return f"{self.app_label}/{self.model_name}"

    @cached_property[str]
    def url_namespace(self) -> str:
        return f"{self.app_label}_{self.model_name}"

    @property
    def permission_policy(self):
        if issubclass(self.model, PermissionedModel):
            return self.model.permission_policy()
        return super().permission_policy

    def get_queryset(self, request: HttpRequest) -> _QS:
        from kausal_common.models.permissions import PermissionedQuerySet

        from paths.types import PathsQuerySet

        qs = self.model._default_manager.get_queryset()
        if isinstance(qs, PermissionedQuerySet):
            qs = qs.viewable_by(admin_req(request).user)
        if isinstance(qs, PathsQuerySet):
            qs = qs.within_realm(self.admin_instance)
        return cast('_QS', qs)

    def get_edit_handler(self):
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
            parent_viewset=self,
        )
