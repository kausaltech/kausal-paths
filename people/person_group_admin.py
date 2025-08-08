from __future__ import annotations

from django.utils.translation import gettext_lazy as _
from wagtail.admin.forms.models import WagtailAdminModelForm
from wagtail.admin.panels import FieldPanel, MultipleChooserPanel
from wagtail.snippets.models import register_snippet

from admin_site.viewsets import PathsCreateView, PathsViewSet
from people.models import PersonGroup
from users.models import User


class PersonGroupCreateView(PathsCreateView[PersonGroup, WagtailAdminModelForm[PersonGroup, User]]):
    def save_instance(self) -> PersonGroup:
        instance = self.form.save(commit=False)
        instance.instance = self.admin_instance
        instance.save()
        return instance


class PersonGroupSnippetViewSet(PathsViewSet):
    model = PersonGroup
    menu_label = _('Person groups')
    menu_icon = 'group'
    menu_order = 601  # The menu item for "Users" has 600; (Django) "Groups" has 601; "Sites" has 602
    add_to_settings_menu = True
    add_view_class = PersonGroupCreateView

    panels = [
        FieldPanel('name'),
        MultipleChooserPanel(
            'persons_edges',
            chooser_field_name='person',
            heading=_('Group members'),
            label=_('Person'),
        ),
    ]


register_snippet(PersonGroupSnippetViewSet)
