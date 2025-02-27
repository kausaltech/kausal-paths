from __future__ import annotations

from wagtail.users.views.users import UserViewSet as WagtailUserViewSet

from .forms import UserCreationForm, UserEditForm


class UserViewSet(WagtailUserViewSet):
    template_prefix = "users/"

    def get_form_class(self, for_update=False):
        if for_update:
            return UserEditForm
        return UserCreationForm
