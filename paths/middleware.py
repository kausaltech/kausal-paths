import re
from typing import Callable, Optional, cast

from django.conf import settings
from django.db import transaction
from django.http import HttpRequest
from django.utils import translation
from django.utils.deprecation import MiddlewareMixin
from wagtail.users.models import UserProfile

from nodes.models import InstanceConfig
from paths.admin_context import set_admin_instance
from paths.cache import PathsObjectCache
from paths.types import PathsAdminRequest, PathsRequest
from users.models import User


class RequestMiddleware(MiddlewareMixin):
    def __init__(self, get_response) -> None:
        super().__init__(get_response)

    def __call__(self, request: HttpRequest):
        request = cast(PathsRequest, request)
        request.cache = PathsObjectCache()
        return self.get_response(request)  # pyright: ignore


class AdminMiddleware:
    def __init__(self, get_response: Callable) -> None:
        self.get_response = get_response

    def get_admin_instance(self, request: PathsAdminRequest) -> None | InstanceConfig:
        if not re.match(r'^/admin/', request.path):
            return None

        user = request.user
        if not isinstance(user, User):
            return
        if not user.is_active or not user.is_authenticated or not user.is_staff:
            return None

        instance_config: Optional[InstanceConfig] = None
        admin_instance_id = request.session.get('admin_instance')
        if admin_instance_id:
            instance_config = InstanceConfig.objects.filter(id=admin_instance_id).first()

        # FIXME
        adminable_instances = InstanceConfig.permission_policy.instances_user_has_any_permission_for(user, ['change'])

        if instance_config is not None:
            if instance_config not in adminable_instances:
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
        ic = self.get_admin_instance(request)
        if ic is None:
            return self.get_response(request)

        user = request.user
        assert isinstance(user, User)
        self.activate_language(ic, user)
        assert ic is not None
        set_admin_instance(ic, request=request)

        assert ic.site is not None
        request._wagtail_site = ic.site

        # If it's an admin method that changes something, invalidate GraphQL cache.
        if request.method in ('POST', 'PUT', 'DELETE'):
            def invalidate_cache():
                ic.invalidate_cache()
            transaction.on_commit(invalidate_cache)

        return self.get_response(request)
