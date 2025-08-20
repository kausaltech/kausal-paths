from __future__ import annotations

from django.contrib.admin.widgets import AdminFileWidget
from django.utils.translation import gettext_lazy as _
from wagtail.admin.panels import FieldPanel
from wagtail.snippets.models import register_snippet

from dal_select2.widgets import ModelSelect2

from paths.context import realm_context

from admin_site.utils import SuperAdminOnlyMenuItem
from admin_site.viewsets import PathsViewSet

from . import chooser  # noqa: F401
from .forms import PersonForm
from .models import Person


class AvatarWidget(AdminFileWidget):
    template_name = 'kausal_common/people/avatar_widget.html'


class PersonSnippetViewSet(PathsViewSet):
    model = Person
    menu_label = _('People')
    menu_icon = 'user'
    menu_order = 300
    add_to_admin_menu = True
    menu_item_class = SuperAdminOnlyMenuItem
    form_class: PersonForm

    panels = [
        FieldPanel('first_name'),
        FieldPanel('last_name'),
        FieldPanel('email'),
        FieldPanel('title'),
        FieldPanel('image', widget=AvatarWidget),
        FieldPanel(
            'organization',
            widget=ModelSelect2(url='organization-autocomplete'),
        ),
    ]

    search_fields = ['first_name', 'last_name', 'email', 'title']
    list_per_page = 50

    list_display =  ['avatar', 'first_name', 'last_name', 'email', 'organization', 'title']
    class Media:
        css = {
            'all': ('dal_select2/css/select2.css',)
        }
        js = ('dal_select2/js/select2.js',)

    def get_queryset(self, request):
        active_instance = realm_context.get().realm
        qs = Person.objects.qs.available_for_instance(active_instance)
        return qs

    def get_form_class(self, for_update=False):
        return PersonForm


register_snippet(PersonSnippetViewSet)
