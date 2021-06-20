from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from wagtail.core.models import Page
from wagtail.core.fields import RichTextField
from wagtail.admin.edit_handlers import FieldPanel


class NodePage(Page):
    description = RichTextField()
    node = models.CharField(max_length=100, unique=True)

    content_panels = Page.content_panels + [
        FieldPanel('description', classname="full"),
        FieldPanel('node'),
    ]

    parent_page_types = ['wagtailcore.Page']
    subpage_types = []


class InstanceContent(models.Model):
    identifier = models.CharField(
        max_length=100, unique=True, verbose_name=_('Instance identifier')
    )
    modified_at = models.DateTimeField(editable=False, auto_now=True)

    class Meta:
        verbose_name = _('Instance')
        verbose_name_plural = _('Instances')

    def __str__(self):
        return self.identifier


class NodeContent(models.Model):
    instance = models.ForeignKey(
        InstanceContent, editable=False, on_delete=models.CASCADE, related_name='nodes'
    )
    node_id = models.CharField(
        max_length=100, unique=True, verbose_name=_('Node identifier')
    )
    short_description = RichTextField(
        null=True, blank=True, verbose_name=_('Short description')
    )
    body = RichTextField(
        null=True, blank=True, verbose_name=_('Body')
    )

    class Meta:
        verbose_name = _('Node')
        verbose_name_plural = _('Nodes')

    def get_node_object(self):
        from .global_instance import instance
        context = instance.context
        return context.get_node(self.node_id)

    @property
    def name(self):
        return str(self.get_node_object().name)

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        iobj: InstanceContent = self.instance
        iobj.modified_at = timezone.now()
        iobj.save(update_fields=['modified_at'])
