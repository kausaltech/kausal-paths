from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from wagtail.admin.menu import MenuItem
from wagtail.admin.panels import FieldPanel
from wagtail.permission_policies.base import ModelPermissionPolicy
from admin_site.viewsets import PathsViewSet, PathsEditView

from pages.models import InstanceSiteContent


class SiteContentPermissionPolicy(ModelPermissionPolicy):
    def user_has_permission(self, user, action):
        # Disable creating new site content instances
        if action == 'change':
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
        instance = request.admin_instance
        field = self.get_one_to_one_field(instance)
        link_menu_item.url = reverse(self.view_set.get_url_name('edit'), kwargs={'pk': field.pk})
        return link_menu_item

    def is_shown(self, request):
        user = request.user
        if user.is_superuser or user.can_access_admin():
            return True
        instance = request.admin_instance
        field = self.get_one_to_one_field(instance)
        return self.view_set.permission_policy.user_has_permission_for_instance(request.user, 'change', field)


class SiteContentEditView(PathsEditView):
    def get_success_url(self):
        # After editing a model instance, redirect to the edit page again instead of the index page
        return self.get_edit_url()


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
        return SiteContentPermissionPolicy(self.model)

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        qs = qs.filter(instance=request.admin_instance)
        return qs

    def get_menu_item(self, order=None):
        item = InstanceSiteContentModelMenuItem(self, order or self.menu_order)
        return item
