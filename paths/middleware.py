import re
from typing import Optional
from django.conf import settings
from django.db import transaction
from django.utils import translation
from django.utils.deprecation import MiddlewareMixin
from wagtail.users.models import UserProfile
from nodes.models import InstanceConfig
from paths.types import PathsRequest


class AdminMiddleware(MiddlewareMixin):
    def process_view(self, request: PathsRequest, view, *args, **kwargs):
        if not re.match(r'^/admin/', request.path):
            return

        user = request.user
        if not user or not user.is_authenticated or not user.is_staff:
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

        profile = UserProfile.get_for_user(user)
        lang = profile.preferred_language
        if (not lang or lang not in (x[0] for x in settings.LANGUAGES)):
            if instance_config is not None:
                lang = instance_config.default_language
                profile.preferred_language = lang
                profile.save(update_fields=['preferred_language'])
            else:
                # Fallback to default language
                lang = settings.LANGUAGES[0][0]
        translation.activate(lang)

        assert instance_config is not None
        instance = instance_config.get_instance(generate_baseline=True)
        context = instance.context
        context.activate_scenario(context.get_default_scenario())
        request.admin_instance = instance_config

        # FIXME: Create instance-specific Site objects
        if True or not instance_config.site_id:
            return

        request._wagtail_site = instance.site

        # If it's an admin method that changes something, invalidate Plan-related
        # GraphQL cache.
        if request.method in ('POST', 'PUT', 'DELETE'):
            def invalidate_cache():
                instance.invalidate_cache()
            transaction.on_commit(invalidate_cache)
