from django import forms
from django.db.models import CharField
from wagtail.core.models import Page
from wagtail.core.fields import RichTextField
from wagtail.admin.edit_handlers import FieldPanel


class NodeSelectWidget(forms.Select):
    def __init__(self, *args, **kwargs):
        from pages.loader import loader
        instance = loader.instance
        kwargs['choices'] = [(id, str(node.name)) for id, node in instance.context.nodes.items()]
        super().__init__(*args, **kwargs)


class NodePage(Page):
    description = RichTextField()
    node = CharField(max_length=100, unique=True)

    content_panels = Page.content_panels + [
        FieldPanel('description', classname="full"),
        FieldPanel('node'),
    ]

    parent_page_types = ['wagtailcore.Page']
    subpage_types = []
