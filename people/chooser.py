from __future__ import annotations

from wagtail import hooks
from wagtail.search.backends import get_search_backend

from kausal_common.people.chooser import (
    PersonChooserMixin as BasePersonChooserMixin,
    PersonChooserViewSet as BasePersonChooserViewSet,
    PersonModelChooserCreateTabMixin as BasePersonModelChooserCreateTabMixin,
)

from paths.context import realm_context


class PersonChooserMixin(BasePersonChooserMixin):

    def get_object_list(self, search_term=None, **kwargs):
        admin_instance = realm_context.get().realm
        object_list = self.get_unfiltered_object_list().available_for_instance(admin_instance)

        # Workaround to prevent Wagtail from looking for `path` (from Organization) in the search fields for Person
        object_list = self.model.objects.filter(id__in=object_list)  # type: ignore[attr-defined]

        if search_term:
            search_backend = get_search_backend()
            object_list = search_backend.autocomplete(search_term, object_list)

        return object_list

class PersonModelChooserCreateTabMixin(BasePersonModelChooserCreateTabMixin):
    def get_initial(self):
        admin_instance = realm_context.get().realm
        return {'instance': admin_instance}


class PersonChooserViewSet(BasePersonChooserViewSet):
    chooser_mixin_class = PersonChooserMixin
    create_tab_mixin_class = PersonModelChooserCreateTabMixin


@hooks.register('register_admin_viewset')
def register_watch_person_chooser_viewset():
    return PersonChooserViewSet('person_chooser', url_prefix='person-chooser')
