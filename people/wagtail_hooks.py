from wagtail.snippets.models import register_snippet
from wagtail.snippets.views.snippets import SnippetViewSet
from wagtail.admin.panels import FieldPanel, MultiFieldPanel
from wagtail.admin.filters import DateRangePickerWidget
from django.utils.translation import gettext_lazy as _

from .models import Person

from dal_select2.widgets import ModelSelect2


class PersonSnippetViewSet(SnippetViewSet):
    model = Person
    menu_label = _('People')
    menu_icon = 'user'
    menu_order = 200
    add_to_admin_menu = True

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
    list_filter = ['organization', 'created_by']
    search_fields = ['first_name', 'last_name', 'email', 'title']
    list_per_page = 50


register_snippet(PersonSnippetViewSet)