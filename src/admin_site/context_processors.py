import json

from django.conf import settings

from paths.context import realm_context


def sentry(request):
    return dict(sentry_dsn=settings.SENTRY_DSN, deployment_type=settings.DEPLOYMENT_TYPE)


def i18n(request):
    return dict(
        language_fallbacks_json=json.dumps(settings.MODELTRANS_FALLBACK),
        supported_languages_json=json.dumps([x[0] for x in settings.LANGUAGES]),
    )


def active_instance(request):
    """Expose the instance the admin is editing, for the sidebar header."""
    if not realm_context.is_set():
        return {}
    return dict(active_instance=realm_context.get().realm)
