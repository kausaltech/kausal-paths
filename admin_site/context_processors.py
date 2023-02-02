import json
from django.conf import settings


def sentry(request):
    return dict(sentry_dsn=settings.SENTRY_DSN, deployment_type=settings.DEPLOYMENT_TYPE)


def i18n(request):
    return dict(
        language_fallbacks_json=json.dumps(settings.MODELTRANS_FALLBACK),
        supported_languages_json=json.dumps([x[0] for x in settings.LANGUAGES]),
    )

