from __future__ import annotations

from wagtail import hooks
from wagtail.admin.panels import FieldPanel, ObjectList, TabbedInterface
from wagtail.snippets.models import register_snippet
from wagtail_color_panel.edit_handlers import NativeColorPanel

from admin_site.panels import TranslatedFieldPanel, TranslatedFieldRowPanel
from admin_site.viewsets import PathsViewSet
from nodes.models import NodeConfig, NodeConfigQuerySet


class NodeViewSet(PathsViewSet[NodeConfig, NodeConfigQuerySet]):
    model = NodeConfig
    inspect_view_enabled = True
    icon = 'kausal-node'
    add_to_admin_menu = True
    search_fields = ['name', 'identifier']
    menu_order = 10

    basic_panels = [
        FieldPanel("identifier", read_only=True),
        TranslatedFieldRowPanel("name"),
        NativeColorPanel("color"),
        FieldPanel("is_visible"),
        FieldPanel("indicator_node"),
        TranslatedFieldRowPanel("goal"),
        TranslatedFieldRowPanel("short_description"),
    ]
    description_panels = [
        TranslatedFieldPanel("description"),
    ]
    extra_panels = [
        FieldPanel("body"),
    ]

    #def get_queryset(self, request: HttpRequest) -> NodeConfigQuerySet:
    #    qs = super().get_queryset(request)
    #    qs = qs.filter(instance=self.admin_instance)
    #    return qs

    def get_edit_handler(self):
        edit_handler = TabbedInterface([
            ObjectList(self.basic_panels, heading='Basic'),
            ObjectList(self.description_panels, heading='Description'),
            ObjectList(self.extra_panels, heading='Extra'),
        ])
        return edit_handler.bind_to_model(self.model)


register_snippet(NodeViewSet)


def register_icons(icons: list[str]):
    return icons + [
        'wagtailfontawesomesvg/solid/circle-nodes.svg',
    ]

hooks.register('register_icons', register_icons)
