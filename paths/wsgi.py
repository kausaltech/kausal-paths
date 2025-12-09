"""
WSGI config for paths project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/wsgi/
"""
from __future__ import annotations

import os
from datetime import UTC, datetime

from django.core.wsgi import get_wsgi_application

from kausal_common.deployment import run_deployment_checks

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'paths.settings')

django_application = get_wsgi_application()

# We execute all the checks when running under uWSGI, so that we:
#   - load more of the code to save memory after uWSGI forks workers
#   - keep the state of the system closer to how it is under runserver
try:
    import uwsgi  # type: ignore
    run_deployment_checks()
    HAS_UWSGI = True
except ImportError:
    HAS_UWSGI = False


def set_log_vars(resp):
    from kausal_common.logging.handler import ISO_FORMAT
    now = datetime.now(UTC)
    uwsgi.set_logvar('isotime', now.strftime(ISO_FORMAT).replace('+00:00', 'Z'))
    if hasattr(resp, '_response'):
        # Sentry injects a ScopedResponse class
        resp = resp._response
    status = getattr(resp, 'status_code', None)
    level = 'INFO'
    if isinstance(status, int):
        if status >= 400 and status < 500:
            level = 'WARNING'
        elif status >= 500:
            level = 'ERROR'
    uwsgi.set_logvar('level', level)

def application(env, start_response):
    ret = django_application(env, start_response)
    if HAS_UWSGI:
        set_log_vars(ret)
    return ret
