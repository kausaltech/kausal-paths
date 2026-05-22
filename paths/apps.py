from __future__ import annotations

from wagtail.users.apps import WagtailUsersAppConfig


class CustomUsersAppConfig(WagtailUsersAppConfig):
    user_viewset = 'users.viewsets.UserViewSet'
