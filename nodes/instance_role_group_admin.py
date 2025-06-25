from __future__ import annotations

from django.utils.translation import gettext_lazy as _
from wagtail.admin.forms.models import WagtailAdminModelForm
from wagtail.admin.panels import FieldPanel, InlinePanel, MultipleChooserPanel
from wagtail.admin.panels.group import MultiFieldPanel
from wagtail.snippets.models import register_snippet

from kausal_common.datasets.chooser import DatasetChooser

from admin_site.viewsets import PathsCreateView, PathsViewSet
from nodes.models import InstanceRoleGroup
from users.models import User


class InstanceRoleGroupCreateView(PathsCreateView[InstanceRoleGroup, WagtailAdminModelForm[InstanceRoleGroup, User]]):
    def save_instance(self) -> InstanceRoleGroup:
        instance = self.form.save(commit=False)
        instance.instance = self.admin_instance
        instance.save()
        return instance


class InstanceRoleGroupSnippetViewSet(PathsViewSet):
    model = InstanceRoleGroup
    menu_label = _('Role groups')
    menu_icon = 'group'
    menu_order = 201
    add_to_settings_menu = True
    add_view_class = InstanceRoleGroupCreateView

    panels = [
        FieldPanel('name'),
        MultipleChooserPanel(
            'persons_edges',
            chooser_field_name='person',
            heading=_('Group members'),
            label=_('Person'),
        ),
        InlinePanel(
            'datasets_edges',
            heading=_('Dataset permissions'),
            label=_('Dataset permissions'),
            panels=[
                FieldPanel('dataset', widget=DatasetChooser),
                MultiFieldPanel([
                    FieldPanel('can_view'),
                    FieldPanel('can_edit'),
                    FieldPanel('can_delete'),
                ], heading=_('Permissions')),

            ]
        ),
    ]


register_snippet(InstanceRoleGroupSnippetViewSet)
