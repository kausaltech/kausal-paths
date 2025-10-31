from __future__ import annotations

import typing

from wagtail.admin.views.generic.chooser import ChooseResultsView
from wagtail.snippets.views.chooser import ChooseView

from kausal_common.people.chooser import PersonChooserViewSet as BasePersonChooserViewSet

from paths.context import realm_context

if typing.TYPE_CHECKING:
    from django.db.models import QuerySet

    from .models import Person


def filter_person_queryset_by_admin_instance(view, queryset) -> QuerySet[Person]:
    admin_instance = realm_context.get().realm
    object_list = queryset.available_for_instance(admin_instance)
    # Workaround to prevent Wagtail from looking for `path` (from Organization) in the search fields for Person
    return queryset.filter(id__in=object_list)  # type: ignore[attr-defined]


class PersonChooseView(ChooseView):
    def get_creation_form_kwargs(self) -> dict[str, typing.Any]:
        kwargs = super().get_creation_form_kwargs()
        kwargs['admin_instance'] = realm_context.get().realm
        return kwargs

    def get_object_list(self) -> QuerySet[Person]:
        return filter_person_queryset_by_admin_instance(self, super().get_object_list())


class PersonChooseResultsView(ChooseResultsView):
    def get_object_list(self, search_term=None, **kwargs) -> QuerySet[Person]:
        return filter_person_queryset_by_admin_instance(self, super().get_object_list())


class PersonChooserViewSet(BasePersonChooserViewSet):
    choose_view_class = PersonChooseView
    choose_results_view_class = PersonChooseResultsView
    # TODO: uncomment this once the PersonForm has been merged from the other branch
    # creation_form_class = PersonForm
