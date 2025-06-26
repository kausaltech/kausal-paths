from django.contrib.gis.db.models.functions import Distance
from django.contrib.gis.geos import Point
from django.db.models import Q
from django.utils.translation import gettext_lazy as _

from dal_select2.views import Select2QuerySetView

from .models import Organization

class OrganizationAutocomplete(Select2QuerySetView):

    def get_queryset(self):
        if not self.request.user.is_authenticated:
            return Organization.objects.none()
        qs = Organization.objects.available_for_instance(
            self.request.user.get_active_instance())

        if self.q:
            qs = qs.filter(
                Q(distinct_name__icontains=self.q) |
                Q(name__icontains=self.q) |
                Q(internal_abbreviation__icontains=self.q) |
                Q(abbreviation__icontains=self.q),
            )
        return qs

    def get_result_label(self, result):
        return str(result)

    def get_result_value(self, result):
        return result.id