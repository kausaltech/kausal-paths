import re
from typing import Callable, Optional
from django.conf import settings
from django.db import transaction
from django.utils import translation
from wagtail.users.models import UserProfile
from nodes.models import InstanceConfig
from paths.admin_context import set_admin_instance
from paths.types import PathsRequest
from users.models import User


class AdminMiddleware:
    def __init__(self, get_response: Callable) -> None:
        self.get_response = get_response

    def get_admin_instance(self, request: PathsRequest):
        if not re.match(r'^/admin/', request.path):
            return

        user = request.user
        if not isinstance(user, User):
            return
        if not user.is_active or not user.is_authenticated or not user.is_staff:
            return

        instance_config: Optional[InstanceConfig] = None
        admin_instance_id = request.session.get('admin_instance')
        if admin_instance_id:
            instance_config = InstanceConfig.objects.filter(id=admin_instance_id).first()

        if instance_config is not None:
            # FIXME: Check perms
            pass

        if instance_config is None:
            # FIXME: Find the most recent instance the user has admission permissions to
            instance_config = InstanceConfig.objects.first()
        return instance_config

    def activate_language(self, ic: InstanceConfig, user: User):
        profile = UserProfile.get_for_user(user)
        lang = profile.preferred_language
        if (not lang or lang not in (x[0] for x in settings.LANGUAGES)):
            if ic is not None:
                lang = ic.default_language
                profile.preferred_language = lang
                profile.save(update_fields=['preferred_language'])
            else:
                # Fallback to default language
                lang = settings.LANGUAGES[0][0]
        translation.activate(lang)

    def __call__(self, request: PathsRequest):
        ic = self.get_admin_instance(request)
        if ic is None:
            return self.get_response(request)

        user = request.user
        self.activate_language(ic, user)
        assert ic is not None
        instance = ic.get_instance()
        context = instance.context
        context.activate_scenario(context.get_default_scenario())
        request.admin_instance = ic
        set_admin_instance(ic)

        # FIXME: Create instance-specific Site objects
        if True or not ic.site_id:
            return self.get_response(request)


        request._wagtail_site = instance.site

        # If it's an admin method that changes something, invalidate Plan-related
        # GraphQL cache.
        if request.method in ('POST', 'PUT', 'DELETE'):
            def invalidate_cache():
                instance.invalidate_cache()
            transaction.on_commit(invalidate_cache)
