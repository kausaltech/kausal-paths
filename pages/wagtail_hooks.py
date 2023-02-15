from django.utils.translation import gettext_lazy as _
from django.urls import reverse
from wagtail.admin.menu import Menu, MenuItem, SubmenuMenuItem
from wagtail import hooks

from nodes.models import InstanceConfig


class InstanceChooserMenuItem(SubmenuMenuItem):
    def is_shown(self, request):
        if len(self.menu.menu_items_for_request(request)) > 1:
            return True
        return False

    def is_active(self, request):
        return bool(self.menu.active_menu_items(request))


class InstanceItem(MenuItem):
    pass


class InstanceChooserMenu(Menu):
    def menu_items_for_request(self, request):
        user = request.user
        instances = InstanceConfig.objects.all()  # FIXME
        items = []
        for instance in instances:
            url = reverse('change-admin-instance', kwargs=dict(instance_id=instance.id))
            url += '?admin=wagtail'
            icon_name = ''
            if instance == user.get_active_instance():
                icon_name = 'tick'
            text = instance.name or instance.identifier
            item = InstanceItem(text, url, icon_name=icon_name)
            items.append(item)
        return items


instance_chooser = InstanceChooserMenu(None)


@hooks.register('register_admin_menu_item')
def register_instance_chooser():
    return InstanceChooserMenuItem(
        _('Choose instance'), instance_chooser, classnames='icon icon-fa-check-circle-o', order=9000
    )
