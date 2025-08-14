from __future__ import annotations

from django.db.models import Q

from dal_select2.views import Select2QuerySetView

from kausal_common.users import user_or_none

from paths.context import realm_context

from .models import Organization


class OrganizationAutocomplete(Select2QuerySetView):
    def get_queryset(self):
        user = user_or_none(self.request.user)
        ic = realm_context.get().realm
        if not user:
            return Organization.objects.none()
        qs = Organization.objects.qs.available_for_instance(ic)

        if self.q:
            qs = qs.filter(
                Q(distinct_name__icontains=self.q)
                | Q(name__icontains=self.q)
                | Q(internal_abbreviation__icontains=self.q)
                | Q(abbreviation__icontains=self.q),
            )
        return qs

    def get_result_label(self, result):
        return str(result)

    def get_result_value(self, result):
        return result.id
