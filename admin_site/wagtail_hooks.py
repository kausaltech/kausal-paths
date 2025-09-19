from __future__ import annotations

from typing import TYPE_CHECKING

from django.templatetags.static import static
from django.urls import reverse
from django.utils.html import format_html
from django.utils.translation import gettext_lazy as _
from wagtail import hooks
from wagtail.admin.menu import Menu, MenuItem, SubmenuMenuItem

from wagtail_localize.wagtail_hooks import TranslationsReportMenuItem

from kausal_common.users import user_or_bust

from paths.context import realm_context

if TYPE_CHECKING:
    from django.http.request import HttpRequest

    from paths.types import PathsAdminRequest


def global_admin_css():
    return format_html('<link rel="stylesheet" href="{}">', static('css/admin-styles.css'))


hooks.register('insert_global_admin_css', global_admin_css)


@hooks.register('register_icons')
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
        'wagtailadmin/icons/kausal-attributes.svg',
    ]


class InstanceChooserMenuItem(SubmenuMenuItem):
    def is_shown(self, request):
        if len(self.menu.menu_items_for_request(request)) > 1:
            return True
        return False


class InstanceItem(MenuItem):
    pass


class InstanceChooserMenu(Menu):
    def menu_items_for_request(self, request: HttpRequest):
        user = user_or_bust(request.user)
        instances = user.get_adminable_instances()
        if len(instances) < 2:
            return []
        items = []
        for instance in instances:
            url = reverse('change-admin-instance', kwargs=dict(instance_id=instance.pk))
            url += '?admin=wagtail'
            icon_name = ''
            if instance == realm_context.get().realm:
                icon_name = 'tick-inverse'
            text = instance.name or instance.identifier
            item = InstanceItem(text, url, icon_name=icon_name)
            items.append(item)
        return items


instance_chooser = InstanceChooserMenu(None)


def register_instance_chooser():
    return InstanceChooserMenuItem(_('Choose instance'), instance_chooser, icon_name='home', order=9000)


hooks.register('register_admin_menu_item', register_instance_chooser)


@hooks.register('construct_main_menu')
def hide_snippets_menu_item(request, menu_items):
    menu_items[:] = [item for item in menu_items if item.name != 'snippets']


@hooks.register('construct_reports_menu')
def patch_translations_report_menu_item(request: PathsAdminRequest, menu_items: list):
    # We don't want to show the Reports menu to people without rights to it.
    # TranslationsReportMenuItem is always shown by default.
    # Hide from non-superusers until we know who needs this menu item
    if getattr(request.user, 'is_superuser', False):
        return
    menu_items[:] = [menu_item for menu_item in menu_items if not isinstance(menu_item, TranslationsReportMenuItem)]
