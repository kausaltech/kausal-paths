from __future__ import annotations

from typing import Callable

from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from wagtail.admin.menu import MenuItem
from wagtail.admin.panels import FieldPanel

from kausal_common.models.permission_policy import ParentInheritedPolicy

from paths.context import realm_context

from admin_site.viewsets import PathsEditView, PathsViewSet
from pages.models import InstanceSiteContent


class SiteContentPermissionPolicy(ParentInheritedPolicy):
    def __init__(self):
        from nodes.models import InstanceConfig
        super().__init__(model=InstanceSiteContent, parent_model=InstanceConfig, parent_field='instance')

    def user_has_permission(self, user, action):
        # Disable creating of site content instances
        if action == 'add':
            return False
        return super().user_has_permission(user, action)


class InstanceSiteContentModelMenuItem(MenuItem):

    def __init__(self, view_set, order):
        self.view_set = view_set

        super().__init__(
            label=view_set.menu_label,
            url="",  # This is set in render_component
            name=view_set.menu_name,
            icon_name=view_set.icon,
            order=order,
        )

    def get_one_to_one_field(self, instance):
        return instance.site_content

    def render_component(self, request):
        # When clicking the menu item, use the edit view instead of the index view.
        link_menu_item = super().render_component(request)
        instance = realm_context.get().realm
        field = self.get_one_to_one_field(instance)
        link_menu_item.url = reverse(self.view_set.get_url_name('edit'), kwargs={'pk': field.pk})
        return link_menu_item

    def is_shown(self, request):
        user = request.user
        if user.is_superuser:
            return True
        instance = realm_context.get().realm
        field = self.get_one_to_one_field(instance)
        return self.view_set.permission_policy.user_has_permission_for_instance(request.user, 'change', field)


# TODO: Could be moved to kausal-common, this is used as-is in kausal-watch as well
class SuccessUrlEditPageMixin:
    """After editing a model instance, redirect to the edit page again instead of the index page."""
    get_edit_url: Callable

    def get_success_url(self) -> str:
        return self.get_edit_url()

    def get_success_buttons(self) -> list:
        # Remove the button that takes the user to the edit view from the
        # success message, since we're redirecting back to the edit view already
        return []

    def get_breadcrumbs_items(self):
        # As the idea is to stay only on the edit page, hide the breadcrumb trail
        # that gives access e.g. to the index view
        return []


class SiteContentEditView(SuccessUrlEditPageMixin, PathsEditView):
    pass


class InstanceSiteContentViewSet(PathsViewSet):
    model = InstanceSiteContent
    icon = 'tasks'
    menu_label = _('Site Content')
    menu_order = 101
    add_to_settings_menu = True
    edit_view_class = SiteContentEditView

    panels = [
        FieldPanel('intro_content'),
    ]

    @property
    def permission_policy(self):
        return SiteContentPermissionPolicy()

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        qs = qs.filter(instance=realm_context.get().realm)
        return qs

    def get_menu_item(self, order=None):
        item = InstanceSiteContentModelMenuItem(self, order or self.menu_order)
        return item
