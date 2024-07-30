"""
WSGI config for paths project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/wsgi/
"""

import os
from datetime import datetime, UTC
from types import MethodType
from kausal_common.deployment import run_deployment_checks
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'paths.settings')

django_application = get_wsgi_application()

# We execute all the checks when running under uWSGI, so that we:
#   - load more of the code to save memory after uWSGI forks workers
#   - keep the state of the system closer to how it is under runserver
try:
    import uwsgi  # type: ignore  # noqa
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

import weakref

df_count = 0
df_map = {}

if False:
    import polars as pl
    import weakref

    def print_fin(*args, **kwargs):
        print('DF fin')

    old_new = pl.DataFrame.__new__
    def new_df(cls, *args):
        global df_count
        print(cls)
        print(args)
        ret = old_new(cls)
        df_count += 1
        return ret

    def del_df(self):
        global df_count
        df_count -= 1
        print('del, now %d' % df_count)

    pl.DataFrame.__new__ = classmethod(new_df)
    pl.DataFrame.__del__ = del_df



def application(env, start_response):
    ret = django_application(env, start_response)
    if HAS_UWSGI:
        set_log_vars(ret)
    return ret
