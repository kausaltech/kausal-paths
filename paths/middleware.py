from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

from django.conf import settings
from django.db import transaction
from django.utils import translation
from django.utils.deprecation import MiddlewareMixin
from wagtail.users.models import UserProfile

from kausal_common.logging.http import start_request

from paths.admin_context import set_admin_instance
from paths.context import PathsObjectCache
from paths.types import PathsAdminRequest, PathsRequest

from nodes.models import InstanceConfig
from users.models import User

if TYPE_CHECKING:
    from collections.abc import Callable

    from django.http import HttpRequest


class RequestMiddleware(MiddlewareMixin):
    def __init__(self, get_response) -> None:
        super().__init__(get_response)

    def __call__(self, request: HttpRequest):
        request = cast(PathsRequest, request)
        request.cache = PathsObjectCache(request.user)
        with start_request(request):
            return self.get_response(request)  # pyright: ignore


class AdminMiddleware:
    def __init__(self, get_response: Callable) -> None:
        self.get_response = get_response

    def get_admin_instance(self, request: PathsAdminRequest) -> None | InstanceConfig:
        if not re.match(r'^/admin/', request.path):
            return None

        user = request.user
        if not isinstance(user, User):
            return None
        if not user.is_active or not user.is_authenticated or not user.is_staff:
            return None

        instance_config: InstanceConfig | None = None
        admin_instance_id = request.session.get('admin_instance')
        if admin_instance_id:
            instance_config = InstanceConfig.objects.filter(id=admin_instance_id).first()

        adminable_instances = (
            InstanceConfig.permission_policy().instances_user_has_any_permission_for(user, ['change'])
            .filter(site__isnull=False)
        )
        if instance_config is not None and instance_config not in adminable_instances:
            instance_config = None

        if instance_config is None:
            instance_config = adminable_instances.last()
            if instance_config is not None:
                request.session['admin_instance'] = instance_config.pk

        return instance_config

    def activate_language(self, ic: InstanceConfig, user: User):
        profile = UserProfile.get_for_user(user)
        lang = profile.preferred_language
        if (not lang or lang not in (x[0] for x in settings.LANGUAGES)):
            if ic is not None:
                lang = ic.primary_language
                profile.preferred_language = lang
                profile.save(update_fields=['preferred_language'])
            else:
                # Fallback to default language
                lang = settings.LANGUAGES[0][0]
        translation.activate(lang)

    def __call__(self, request: PathsAdminRequest):
        from paths.context import RealmContext, realm_context

        ic = self.get_admin_instance(request)
        if ic is None:
            return self.get_response(request)

        user = request.user
        assert isinstance(user, User)
        self.activate_language(ic, user)

        request._wagtail_site = ic.site

        # If it's an admin method that changes something, invalidate GraphQL cache.
        if request.method in ('POST', 'PUT', 'DELETE'):
            def invalidate_cache() -> None:
                ic.invalidate_cache()
            transaction.on_commit(invalidate_cache)

        set_admin_instance(ic, request=request)
        ctx = RealmContext(realm=ic, user=request.user)
        with realm_context.activate(ctx):
            return self.get_response(request)
