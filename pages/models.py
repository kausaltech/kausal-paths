from django.db.models import CharField
from wagtail.core.models import Page
from wagtail.core.fields import RichTextField
from wagtail.admin.edit_handlers import FieldPanel

from pages.loader import loader


class NodePage(Page):
    NODE_CHOICES = [(id, str(node.name)) for id, node in loader.context.nodes.items()]

    description = RichTextField()
    node = CharField(max_length=100, choices=NODE_CHOICES, unique=True)

    content_panels = Page.content_panels + [
        FieldPanel('description', classname="full"),
        FieldPanel('node'),
    ]

    parent_page_types = ['wagtailcore.Page']
    subpage_types = []
