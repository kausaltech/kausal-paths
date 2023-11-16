from django.db import models
from wagtail import hooks
from wagtail.admin.panels import FieldPanel, ObjectList, TabbedInterface
from wagtail.snippets.models import register_snippet
from wagtail_color_panel.edit_handlers import NativeColorPanel

from admin_site.panels import TranslatedFieldPanel
from admin_site.viewsets import PathsViewSet

from nodes.models import NodeConfig
from paths.types import PathsAdminRequest


class NodeViewSet(PathsViewSet):
    model = NodeConfig
    icon = 'kausal-node'
    add_to_admin_menu = True
    search_fields = ['name', 'identifier']
    menu_order = 10

    basic_panels = [
        FieldPanel("identifier", read_only=True),
        TranslatedFieldPanel("name"),
        NativeColorPanel("color"),
        FieldPanel("indicator_node"),
        FieldPanel("goal"),
        FieldPanel("short_description"),
    ]
    description_panels = [
        FieldPanel("description"),
    ]
    extra_panels = [
        FieldPanel("body"),
    ]

    def get_queryset(self, request: PathsAdminRequest) -> models.QuerySet[NodeConfig]:
        qs = super().get_queryset(request)
        qs = qs.filter(instance=request.admin_instance)
        return qs

    def get_edit_handler(self):
        edit_handler = TabbedInterface([
            ObjectList(self.basic_panels, heading='Basic'),
            ObjectList(self.description_panels, heading='Description'),
            ObjectList(self.extra_panels, heading='Extra'),
        ])
        return edit_handler.bind_to_model(self.model)


register_snippet(NodeViewSet)


@hooks.register("register_icons")
def register_icons(icons: list[str]):
    return icons + [
        'wagtailfontawesomesvg/solid/circle-nodes.svg',
    ]
