from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

from django.conf import settings
from django.db import transaction
from django.utils import translation
from django.utils.deprecation import MiddlewareMixin
from wagtail.users.models import UserProfile

from asgiref.sync import iscoroutinefunction, markcoroutinefunction, sync_to_async

from kausal_common.users import user_or_none

from paths.admin_context import set_admin_instance
from paths.context import PathsObjectCache

from nodes.models import InstanceConfig
from users.models import User

if TYPE_CHECKING:
    from django.http import HttpRequest

    from paths.types import PathsAdminRequest, PathsRequest


class RequestMiddleware(MiddlewareMixin):
    def __init__(self, get_response) -> None:
        super().__init__(get_response)

    def create_object_cache(self, request: HttpRequest):
        req = cast('PathsRequest', request)
        req.cache = PathsObjectCache(req.user)

    async def __acall__(self, request: HttpRequest):
        self.create_object_cache(request)
        return self.get_response(request)

    def __call__(self, request: HttpRequest):
        self.create_object_cache(request)
        return self.get_response(request)


class AdminMiddleware:
    async_capable = True

    def __init__(self, get_response) -> None:
        self.get_response = get_response
        if iscoroutinefunction(get_response):
            markcoroutinefunction(self)

    async def get_admin_instance(self, request: PathsAdminRequest) -> None | InstanceConfig:
        if not re.match(r'^/admin/', request.path):
            return None

        user = await sync_to_async(user_or_none)(request.user)
        if user is None:
            return None
        if not user.is_active or not user.is_authenticated or not user.is_staff:
            return None

        instance_config: InstanceConfig | None = None
        admin_instance_id = request.session.get('admin_instance')
        if admin_instance_id:
            instance_config = await InstanceConfig.objects.filter(id=admin_instance_id).afirst()

        adminable_instances = [
            ic async for ic in InstanceConfig.permission_policy().instances_user_has_any_permission_for(user, ['change'])
            .filter(site__isnull=False)
        ]
        if instance_config is not None and instance_config not in adminable_instances:
            instance_config = None

        if instance_config is None:
            instance_config = adminable_instances[-1]
            if instance_config is not None:
                request.session['admin_instance'] = instance_config.pk

        return instance_config

    async def activate_language(self, ic: InstanceConfig, user: User):
        profile = await sync_to_async(UserProfile.get_for_user)(user)
        lang = profile.preferred_language
        if (not lang or lang not in (x[0] for x in settings.LANGUAGES)):
            if ic is not None:
                lang = ic.primary_language
                profile.preferred_language = lang
                await profile.asave(update_fields=['preferred_language'])
            else:
                # Fallback to default language
                lang = settings.LANGUAGES[0][0]
        translation.activate(lang)

    async def __call__(self, request: HttpRequest):
        from paths.context import RealmContext, realm_context

        request = cast('PathsAdminRequest', request)
        ic = await self.get_admin_instance(request)
        if ic is None:
            return await self.get_response(request)

        user = request.user
        assert isinstance(user, User)
        await self.activate_language(ic, user)

        def update_wagtail_site() -> None:
            request._wagtail_site = ic.site
        await sync_to_async(update_wagtail_site)()

        # If it's an admin method that changes something, invalidate GraphQL cache.
        if request.method in ('POST', 'PUT', 'DELETE'):
            def invalidate_cache() -> None:
                ic.invalidate_cache()
            await sync_to_async(transaction.on_commit)(invalidate_cache)

        set_admin_instance(ic, request=request)
        ctx = RealmContext(realm=ic, user=request.user)
        with realm_context.activate(ctx):
            return await self.get_response(request)
