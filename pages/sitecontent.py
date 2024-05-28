from django.utils.translation import gettext_lazy as _
from wagtail.admin.panels import FieldPanel
from wagtail.contrib.modeladmin.options import ModelAdmin, ModelAdminMenuItem
from wagtail.contrib.modeladmin.views import EditView

from pages.models import InstanceSiteContent


class InstanceSiteContentModelMenuItem(ModelAdminMenuItem):
    def get_one_to_one_field(self, instance):
        return instance.site_content

    def render_component(self, request):
        # When clicking the menu item, use the edit view instead of the index view.
        link_menu_item = super().render_component(request)
        instance = request.admin_instance
        field = self.get_one_to_one_field(instance)
        link_menu_item.url = self.model_admin.url_helper.get_action_url('edit', field.pk)
        return link_menu_item

    def is_shown(self, request):
        user = request.user
        if user.is_superuser or user.can_access_admin():
            return True
        instance = request.admin_instance
        field = self.get_one_to_one_field(instance)
        return self.model_admin.permission_helper.user_can_edit_obj(request.user, field)


class SuccessUrlEditPageMixin:
    """After editing a model instance, redirect to the edit page again instead of the index page."""
    def get_success_url(self):
        return self.url_helper.get_action_url('edit', self.instance.pk)


class SiteContentEditView(SuccessUrlEditPageMixin, EditView):
    pass


class InstanceSiteContentAdmin(ModelAdmin):
    model = InstanceSiteContent
    menu_icon = 'tasks'
    menu_label = _('Site Content')
    menu_order = 101
    add_to_settings_menu = True
    edit_view_class = SiteContentEditView

    panels = [
        FieldPanel('intro_content'),
    ]

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        qs = qs.filter(instance=request.admin_instance)
        return qs

    def user_can_create(self):
        return False

    def get_menu_item(self, order=None):
        item = InstanceSiteContentModelMenuItem(self, order or self.get_menu_order())
        return item
