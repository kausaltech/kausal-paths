from __future__ import annotations

from django.utils.translation import gettext_lazy as _
from wagtail.admin.panels import FieldPanel
from wagtail.snippets.models import register_snippet
from wagtail.snippets.views.snippets import SnippetViewSet

from dal_select2.widgets import ModelSelect2

from .models import Person


class PersonSnippetViewSet(SnippetViewSet):
    model = Person
    menu_label = _('People')
    menu_icon = 'user'
    menu_order = 200
    add_to_admin_menu = False

    panels = [
        FieldPanel('first_name'),
        FieldPanel('last_name'),
        FieldPanel('email'),
        FieldPanel('title'),
        FieldPanel(
            'organization',
            widget=ModelSelect2(url='organization-autocomplete'),
        ),
    ]

    list_display = ['first_name', 'last_name', 'email', 'organization', 'title']

    # list_filter = ['organization', 'created_by']
    search_fields = ['first_name', 'last_name', 'email', 'title']
    list_per_page = 50


    class Media:
        css = {
            'all': ('dal_select2/css/select2.css',)
        }
        js = ('dal_select2/js/select2.js',)

    def get_queryset(self, request):
        qs = Person.objects.qs.available_for_instance(request.user.get_active_instance())
        return qs


register_snippet(PersonSnippetViewSet)
