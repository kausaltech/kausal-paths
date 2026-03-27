from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, cast, override

from django.contrib.auth.models import AnonymousUser
from django.core.exceptions import FieldDoesNotExist
from django.db.models import Model, QuerySet
from django.forms import BaseModelForm
from django.utils.translation import gettext_lazy as _
from wagtail.admin.forms.models import WagtailAdminModelForm
from wagtail.admin.ui.menus import MenuItem
from wagtail.admin.ui.tables import BulkActionsCheckboxColumn
from wagtail.admin.widgets import Button
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
from kausal_common.users import user_or_none

from admin_site.forms import PathsAdminModelForm
from users.models import User

if TYPE_CHECKING:
    from django.http import HttpRequest
    from wagtail.permission_policies.base import BasePermissionPolicy

    from kausal_common.users import UserOrAnon

    from paths.types import PathsAdminRequest

    from nodes.models import InstanceConfig


def admin_req(request: HttpRequest) -> PathsAdminRequest:
    assert request.user is not None
    assert request.user.is_authenticated
    return cast('PathsAdminRequest', request)


class PathsModelForm[M: Model](WagtailAdminModelForm[M, User]):
    pass


class AdminInstanceMixin:
    @property
    def admin_instance(self) -> InstanceConfig:
        from paths.context import realm_context

        return realm_context.get().realm


def user_has_permission(permission_policy: BasePermissionPolicy, user: UserOrAnon, permission: str, obj: Model) -> bool:
    assert isinstance(permission_policy, ModelPermissionPolicy)
    if isinstance(user, AnonymousUser):
        return False
    return permission_policy.user_has_permission_for_instance(user, permission, obj)


class PathsEditView[M: Model, FormT: BaseModelForm[Any] = WagtailAdminModelForm[Any]](
    HideSnippetsFromBreadcrumbsMixin, EditView[M, FormT], AdminInstanceMixin
):
    def user_has_permission(self, permission):
        user = user_or_none(self.request.user)
        if user is None:
            return False
        return user_has_permission(self.permission_policy, user, permission, self.object)

    def get_editing_sessions(self):
        return None

    def get_form_kwargs(self):
        return {
            **super().get_form_kwargs(),
            'admin_instance': self.admin_instance,
        }


class PathsDeleteView[M: Model, FormT: BaseModelForm[Any] = WagtailAdminModelForm[Any]](DeleteView[M, FormT], AdminInstanceMixin):
    def user_has_permission(self, permission):
        user = user_or_none(self.request.user)
        if user is None:
            return False
        return user_has_permission(self.permission_policy, user, permission, self.object)


class PathsCreateView[M: Model, FormT: BaseModelForm[Any] = WagtailAdminModelForm[Any]](
    PermissionedCreateView[M, FormT], AdminInstanceMixin
):
    def get_form_kwargs(self):
        return {
            **super().get_form_kwargs(),
            'admin_instance': self.admin_instance,
        }

    @override
    def get_create_context(self) -> InstanceConfig:
        return self.admin_instance


class PathsIndexView[M: Model, QS: QuerySet[Any, Any]](HideSnippetsFromBreadcrumbsMixin, IndexView[M, QS]):
    def user_can_change_or_delete_model(self) -> bool:
        return self.permission_policy.user_has_any_permission(self.request.user, ('delete', 'change'))

    @cached_property
    def columns(self):  # type: ignore[override]
        columns = super().columns
        if self.user_can_change_or_delete_model():
            return columns
        return [c for c in columns if not isinstance(c, BulkActionsCheckboxColumn)]

    def get_list_more_buttons(self, instance):
        buttons = super().get_list_more_buttons(instance)

        inspect_url = self.get_inspect_url(instance)

        # Change "Inspect" to "View"
        for menu_item in buttons:
            if menu_item.url == inspect_url:
                break
        else:
            return buttons

        if isinstance(menu_item, MenuItem):
            inspect_button = Button.from_menu_item(menu_item)
        else:
            inspect_button = menu_item
        inspect_button.label = _('View')
        if 'aria-label' in inspect_button.attrs:
            inspect_button.attrs['aria-label'] = _("View '%(title)s'") % {'title': str(instance)}
        inspect_button.priority = 5
        buttons.remove(menu_item)
        buttons.append(inspect_button)

        return buttons


class PathsUsageView[M: Model](HideSnippetsFromBreadcrumbsMixin, UsageView[M]):
    pass


class PathsHistoryView(HideSnippetsFromBreadcrumbsMixin, HistoryView):
    pass


class PathsCopyView[M: Model](HideSnippetsFromBreadcrumbsMixin, CopyView[M]):
    pass


class PathsInspectView[M: Model](HideSnippetsFromBreadcrumbsMixin, InspectView[M]):
    page_title = _('View')


class PathsChooseViewMixin[M: Model](AdminInstanceMixin):
    model: type[M]
    request: HttpRequest

    def get_object_list(self):
        qs: QuerySet[M] = super().get_object_list()  # type: ignore
        try:
            field = self.model._meta.get_field('instance')
        except FieldDoesNotExist:
            field = None
        if field is not None:
            qs = qs.filter(instance=self.admin_instance)
        return qs


class PathsChooseView[M: Model](PathsChooseViewMixin[M], ChooseView):
    pass


class PathsChooseResultsView[M: Model](PathsChooseViewMixin[M], ChooseResultsView):
    pass


class PathsChooserViewSet[M: Model](SnippetChooserViewSet):
    choose_results_view_class = PathsChooseResultsView
    choose_view_class = PathsChooseView
    parent_viewset: PathsViewSet[Any, Any]

    def __init__(self, *args, **kwargs):
        self.parent_viewset = kwargs.pop('parent_viewset')
        super().__init__(*args, **kwargs)


class PathsViewSet[M: Model, QS: QuerySet[Any, Any] = QuerySet[M], FormT: BaseModelForm[Any] = WagtailAdminModelForm[Any]](
    SnippetViewSet[M, FormT]
):
    index_view_class: ClassVar = PathsIndexView
    add_view_class: ClassVar = PathsCreateView
    edit_view_class: ClassVar = PathsEditView
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
        return f'{self.app_label}/{self.model_name}'

    @cached_property[str]
    def url_namespace(self) -> str:
        return f'{self.app_label}_{self.model_name}'

    @property
    def permission_policy(self):
        if issubclass(self.model, PermissionedModel):
            return self.model.permission_policy()
        return super().permission_policy

    def get_queryset(self, request: HttpRequest) -> QS:
        from kausal_common.models.permissions import PermissionedQuerySet

        from paths.types import PathsQuerySet

        qs = self.model._default_manager.get_queryset()
        if isinstance(qs, PermissionedQuerySet):
            qs = qs.viewable_by(admin_req(request).user)
        if isinstance(qs, PathsQuerySet):
            qs = qs.within_realm(self.admin_instance)
        return cast('QS', qs)

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
