from wagtail import hooks
from wagtail.admin.panels import FieldPanel
from wagtail.snippets.models import register_snippet
from wagtail_color_panel.edit_handlers import NativeColorPanel

from admin_site.panels import TranslatedFieldPanel
from admin_site.viewsets import PathsViewSet

from nodes.models import NodeConfig


class NodeViewSet(PathsViewSet):
    model = NodeConfig
    icon = 'circle-nodes'
    add_to_admin_menu = True
    search_fields = ['name', 'identifier']
    menu_order = 10

    panels = [
        FieldPanel("identifier", read_only=True),
        TranslatedFieldPanel("name"),
        NativeColorPanel("color"),
        FieldPanel("short_description"),
        FieldPanel("description"),
    ]

    def get_edit_handler(self):
        return super().get_edit_handler()


register_snippet(NodeViewSet)


@hooks.register("register_icons")
def register_icons(icons: list[str]):
    return icons + [
        'wagtailfontawesomesvg/solid/circle-nodes.svg',
    ]
