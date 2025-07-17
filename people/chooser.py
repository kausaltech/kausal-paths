from __future__ import annotations

from django.utils.translation import gettext_lazy as _
from wagtail.admin.views.generic.chooser import ChooseResultsView, ChooseView
from wagtail.admin.viewsets.chooser import ChooserViewSet

from kausal_common.people.chooser import BasePersonChooseViewMixin

from admin_site.viewsets import AdminInstanceMixin
from people.models import Person


class PersonChooseViewMixin(AdminInstanceMixin, BasePersonChooseViewMixin):
    def get_object_list(self):
        object_list = Person.objects.qs.available_for_instance(self.admin_instance)
        # Workaround to prevent Wagtail from looking for `path` (from Organization) in the search fields for Person
        object_list = Person.objects.filter(id__in=object_list)
        return object_list


class PersonChooseView(PersonChooseViewMixin, ChooseView):
    pass


class PersonChooseResultsView(PersonChooseViewMixin, ChooseResultsView):
    pass


class PersonChooserViewSet(ChooserViewSet):
    model = Person
    choose_view_class = PersonChooseView
    choose_results_view_class = PersonChooseResultsView
    choose_one_text = _('Choose a person')
    choose_another_text = _('Choose another person')
