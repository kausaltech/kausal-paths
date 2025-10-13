from __future__ import annotations

from django.utils.translation import gettext_lazy as _
from wagtail.admin.forms.models import WagtailAdminModelForm
from wagtail.admin.panels import FieldPanel, InlinePanel
from wagtail.snippets.models import register_snippet

from admin_site.viewsets import PathsCreateView, PathsViewSet
from users.models import User, UserGroup


class UserGroupCreateView(PathsCreateView[UserGroup, WagtailAdminModelForm[UserGroup, User]]):
    def save_instance(self) -> UserGroup:
        instance = self.form.save(commit=False)
        instance.instance = self.admin_instance
        instance.save()
        return instance


class UserGroupSnippetViewSet(PathsViewSet):
    model = UserGroup
    menu_label = _('User groups')  # TODO: Distinguish from Django groups, which have a menu item labeled "groups"
    menu_icon = 'group'
    menu_order = 601  # The menu item for "Users" has 600; (Django) "Groups" has 601; "Sites" has 602
    add_to_settings_menu = True
    add_view_class = UserGroupCreateView

    panels = [
        FieldPanel('name'),
        InlinePanel(
            'users_edges',
            heading=_('Group members'),
            panels = [
                FieldPanel('user'),  # FIXME: Use a chooser for this and MultipleChooserPanel as below
            ],
        ),
        # The following requires a user chooser but is preferable to the InlinePanel above. If we don't have a chooser,
        # this might happen: https://github.com/wagtail/wagtail/issues/10646
        # MultipleChooserPanel(
        #     'users_edges',
        #     chooser_field_name='user',
        #     heading=_('Group members'),
        #     label=_('User'),
        # ),
    ]


register_snippet(UserGroupSnippetViewSet)
