from __future__ import annotations

from django.utils.translation import gettext_lazy as _
from wagtail.admin.panels import FieldPanel
from wagtail.snippets.models import register_snippet

from dal.autocomplete import ModelSelect2

from paths.context import realm_context

from admin_site.utils import SuperAdminOnlyMenuItem
from admin_site.viewsets import PathsIndexView, PathsViewSet
from people.chooser import PersonChooserViewSet

from . import (
    chooser,  # noqa: F401
    person_group_admin,  # noqa: F401  # pyright: ignore
)
from .forms import AvatarWidget, PersonForm
from .models import Person


class PersonIndexView(PathsIndexView[Person]):
    def search_queryset(self, queryset):
       # Workaround to prevent Wagtail from looking for `path` (from Organization) in the search fields for Person
        queryset = Person.objects.filter(id__in=queryset)

        return super().search_queryset(queryset)


class PersonSnippetViewSet(PathsViewSet):
    model = Person
    icon = 'user'
    menu_label = _('People')
    menu_icon = 'user'
    menu_order = 300
    add_to_admin_menu = True
    menu_item_class = SuperAdminOnlyMenuItem
    form_class: PersonForm
    chooser_viewset_class = PersonChooserViewSet  # FIXME: needs snippet chooser view set

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
    index_view_class = PersonIndexView

    def get_queryset(self, request):
        active_instance = realm_context.get().realm
        qs = Person.objects.qs.available_for_instance(active_instance)
        return qs

    def get_form_class(self, for_update=False):
        return PersonForm


register_snippet(PersonSnippetViewSet)
