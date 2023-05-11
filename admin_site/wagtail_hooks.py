from django.templatetags.static import static
from django.urls import reverse
from django.utils.html import format_html
from django.utils.translation import gettext_lazy as _
from wagtail import hooks
from wagtail.admin.menu import Menu, MenuItem, SubmenuMenuItem
import logging, sys

from nodes.models import InstanceConfig


def global_admin_css():
    return format_html('<link rel="stylesheet" href="{}">', static('css/admin-styles.css'))

hooks.register('insert_global_admin_css', global_admin_css)

@hooks.register("register_icons")
def register_icons(icons):
    return icons + [
        'wagtailadmin/icons/kausal-node.svg',
        'wagtailadmin/icons/kausal-dataset.svg',
        'wagtailadmin/icons/kausal-dimensions.svg',
        'wagtailadmin/icons/kausal-actions.svg',
        'wagtailadmin/icons/kausal-indicators.svg',
        'wagtailadmin/icons/kausal-organisations.svg',
        'wagtailadmin/icons/kausal-plans.svg',
        'wagtailadmin/icons/kausal-spreadsheets.svg',
        'wagtailadmin/icons/kausal-categories.svg',
        'wagtailadmin/icons/kausal-attributes.svg'
    ]

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


def register_instance_chooser():
    return InstanceChooserMenuItem(
        _('Choose instance'), instance_chooser, icon_name='home', order=9000
    )

hooks.register('register_admin_menu_item', register_instance_chooser)
