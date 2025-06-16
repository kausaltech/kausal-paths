"""Django settings for Kausal Paths."""

from __future__ import annotations

import sys
import warnings
from importlib.util import find_spec
from pathlib import Path
from threading import ExceptHookArgs
from typing import Any

from django.core.exceptions import ImproperlyConfigured
from django.utils.translation import gettext_lazy as _

import environ
from corsheaders.defaults import default_headers as default_cors_headers

from kausal_common import ENV_SCHEMA as COMMON_ENV_SCHEMA, register_settings as register_common_settings
from kausal_common.deployment import set_secret_file_vars
from kausal_common.deployment.http import get_allowed_cors_headers
from kausal_common.sentry.init import init_sentry

from .const import INSTANCE_HOSTNAME_HEADER, INSTANCE_IDENTIFIER_HEADER

PROJECT_NAME = 'paths'

root = environ.Path(__file__) - 2  # two folders back
env = environ.FileAwareEnv(
    ENV_FILE=(str, ''),
    DEBUG=(bool, False),
    DEPLOYMENT_TYPE=(str, 'development'),
    KUBERNETES_MODE=(bool, False),
    SECRET_KEY=(str, ''),
    ALLOWED_HOSTS=(list, ['*']),
    DATABASE_URL=(str, f'postgresql:///{PROJECT_NAME}'),
    DATABASE_CONN_MAX_AGE=(int, 20),
    REDIS_URL=(str, ''),
    CACHE_URL=(str, 'locmemcache://'),
    MEDIA_ROOT=(environ.Path(), root('media')),
    STATIC_ROOT=(environ.Path(), root('static')),
    ADMIN_BASE_URL=(str, 'http://localhost:8000'),
    MEDIA_URL=(str, '/media/'),
    STATIC_URL=(str, '/static/'),
    SENTRY_DSN=(str, ''),
    COOKIE_PREFIX=(str, PROJECT_NAME),
    INTERNAL_IPS=(list, []),
    HOSTNAME_INSTANCE_DOMAINS=(list, ['localhost']),
    CONFIGURE_LOGGING=(bool, True),
    LOG_SQL_QUERIES=(bool, False),
    LOG_GRAPHQL_QUERIES=(bool, False),
    ENABLE_DEBUG_TOOLBAR=(bool, False),
    ENABLE_PERF_TRACING=(bool, False),
    MEDIA_FILES_S3_ENDPOINT=(str, ''),
    MEDIA_FILES_S3_BUCKET=(str, ''),
    MEDIA_FILES_S3_ACCESS_KEY_ID=(str, ''),
    MEDIA_FILES_S3_SECRET_ACCESS_KEY=(str, ''),
    MEDIA_FILES_S3_CUSTOM_DOMAIN=(str, ''),
    WATCH_DEFAULT_API_BASE_URL=(str, 'https://api.watch.kausal.tech'),
    AZURE_AD_CLIENT_ID=(str, ''),
    AZURE_AD_CLIENT_SECRET=(str, ''),
    NZCPORTAL_CLIENT_ID=(str, ''),
    NZCPORTAL_CLIENT_SECRET=(str, ''),
    GITHUB_APP_ID=(str, ''),
    GITHUB_APP_PRIVATE_KEY=(str, ''),
    MOUNTED_SECRET_PATHS=(list, []),
    REQUEST_LOG_MAX_DAYS=(int, 90),
    REQUEST_LOG_METHODS=(list, ['POST', 'PUT', 'PATCH', 'DELETE']),
    REQUEST_LOG_IGNORE_PATHS=(list, ['/v1/graphql/', '/o/introspect/']),
    **COMMON_ENV_SCHEMA,
)

BASE_DIR = root()

if env.bool('ENV_FILE'):
    environ.Env.read_env(env.str('ENV_FILE'))
else:
    dotenv_path = BASE_DIR / Path('.env')
    if dotenv_path.exists():
        environ.Env.read_env(dotenv_path)

# Read all files in the directories given in MOUNTED_SECRET_PATHS whose names look like environment variables and use
# the contents of the files for the corresponding variables
for directory in env('MOUNTED_SECRET_PATHS'):
    set_secret_file_vars(env, directory)

DEBUG = env('DEBUG')
DEPLOYMENT_TYPE = env('DEPLOYMENT_TYPE')
SENTRY_DSN = env('SENTRY_DSN')
ADMIN_BASE_URL = env('ADMIN_BASE_URL')
ALLOWED_HOSTS = env('ALLOWED_HOSTS') + ['127.0.0.1']  # 127.0.0.1 for, e.g., health check
INTERNAL_IPS = env.list('INTERNAL_IPS', default=(['127.0.0.1'] if DEBUG else []))  # type: ignore
DATABASES = {
    'default': env.db_url(engine='kausal_common.database'),
}
DATABASES['default']['ATOMIC_REQUESTS'] = True

# If Redis is configured, but no CACHE_URL is set in the environment,
# default to using Redis as the cache.
REDIS_URL = env('REDIS_URL')

cache_var = 'CACHE_URL'
if env.get_value('CACHE_URL', default=None) is None and REDIS_URL:  # pyright: ignore
    cache_var = 'REDIS_URL'
CACHES = {
    'default': env.cache_url(var=cache_var),
}
if 'KEY_PREFIX' not in CACHES['default']:
    CACHES['default']['KEY_PREFIX'] = PROJECT_NAME

SECRET_KEY = env('SECRET_KEY')

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

AUTH_USER_MODEL = 'users.User'
LOGIN_URL = '/admin/login/'
LOGIN_REDIRECT_URL = '/admin/'
LOGOUT_REDIRECT_URL = '/admin/'

# Application definition

INSTALLED_APPS = [
    'kausal_common',
    'wagtail.contrib.forms',
    'wagtail.contrib.redirects',
    'wagtail.embeds',
    'wagtail.sites',
    'admin_site',  # must be before wagtail.admin
    'users',  # must be before wagtail.users
    # 'wagtail.users',  # this should be removed in favour of the custom app config
    "paths.apps.CustomUsersAppConfig",  # a custom app config for the wagtail.users app
    'wagtail.snippets',
    'wagtail.documents',
    'wagtail.images',
    'wagtail.search',
    'wagtail.admin',
    'wagtail',
    'wagtail.contrib.styleguide',
    'wagtail_localize',
    'wagtail_localize.locales',  # replaces `wagtail.locales`
    'wagtailfontawesomesvg',
    'wagtail_color_panel',
    'generic_chooser',
    'taggit',
    'modelcluster',
    'grapple',
    'graphene_django',

    'social_django',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'drf_spectacular',
    'corsheaders',
    'rest_framework',
    'rest_framework.authtoken',
    'django_extensions',
    'modeltrans',
    'pages',
    'nodes',
    'kausal_common.datasets',
    'frameworks',
    'request_log',
]

MIDDLEWARE = [
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.locale.LocaleMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'kausal_common.deployment.middleware.RequestStartMiddleware',
    'paths.middleware.RequestMiddleware',
    'admin_site.middleware.AuthExceptionMiddleware',
    'paths.middleware.AdminMiddleware',
    'kausal_common.logging.request_log.middleware.LogUnsafeRequestMiddleware', # use middleware from kausal_common, no additions
]

ROOT_URLCONF = f'{PROJECT_NAME}.urls'

common_template_dir = Path(BASE_DIR) / 'kausal_common/templates'
if common_template_dir.exists():
    common_template_dirs = [str(common_template_dir)]
else:
    common_template_dirs = []

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / Path('templates'), *common_template_dirs],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.template.context_processors.i18n',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'wagtail.contrib.settings.context_processors.settings',
                'admin_site.context_processors.sentry',
                'admin_site.context_processors.i18n',
            ],
        },
    },
]

WSGI_APPLICATION = f'{PROJECT_NAME}.wsgi.application'

# Password validation
# https://docs.djangoproject.com/en/3.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

SOCIAL_AUTH_JSONFIELD_ENABLED = True
SOCIAL_AUTH_RAISE_EXCEPTIONS = False

SOCIAL_AUTH_NZCPORTAL_CLIENT_ID = env.str('NZCPORTAL_CLIENT_ID')
SOCIAL_AUTH_NZCPORTAL_CLIENT_SECRET = env.str('NZCPORTAL_CLIENT_SECRET')

AUTHENTICATION_BACKENDS = (
    'admin_site.auth_backends.AzureADAuth',
    *(['admin_site.auth_backends.NZCPortalOAuth2'] if SOCIAL_AUTH_NZCPORTAL_CLIENT_ID else []),
    'admin_site.auth_backends.PasswordAuth',
    'django.contrib.auth.backends.ModelBackend',
)

SOCIAL_AUTH_AZURE_AD_KEY = env.str('AZURE_AD_CLIENT_ID')
SOCIAL_AUTH_AZURE_AD_SECRET = env.str('AZURE_AD_CLIENT_SECRET')

SOCIAL_AUTH_PASSWORD_FORM_URL = '/admin/login/'  # noqa: S105

SOCIAL_AUTH_PIPELINE = (
    'kausal_common.auth.pipeline.log_login_attempt',
    # Get the information we can about the user and return it in a simple
    # format to create the user instance later. On some cases the details are
    # already part of the auth response from the provider, but sometimes this
    # could hit a provider API.
    'social_core.pipeline.social_auth.social_details',
    # Get the social uid from whichever service we're authing thru. The uid is
    # the unique identifier of the given user in the provider.
    'social_core.pipeline.social_auth.social_uid',
    # Generate username from UUID
    'kausal_common.auth.pipeline.get_username',
    # Checks if the current social-account is already associated in the site.
    'social_core.pipeline.social_auth.social_user',
    # Finds user by email address
    'kausal_common.auth.pipeline.find_user_by_email',
    # Validate password
    'kausal_common.auth.pipeline.validate_user_password',
    # Get or create the user and update user data
    'kausal_common.auth.pipeline.create_or_update_user',
    # Create the record that associated the social account with this user.
    'social_core.pipeline.social_auth.associate_user',
    # Populate the extra_data field in the social record with the values
    # specified by settings (and the default ones like access_token, etc).
    'social_core.pipeline.social_auth.load_extra_data',
    'admin_site.auth_pipeline.assign_roles',
    'admin_site.auth_pipeline.update_role_permissions',
    # Update avatar photo from MS Graph
    # 'kausal_common.auth.pipeline.update_avatar',
)


CORS_ALLOWED_ORIGIN_REGEXES = [
    # Match localhost with optional port
    r'^https?://([a-z0-9-_]+\.)+localhost(:\d+)?$',
    r'^https://([a-z0-9-_]+\.)*kausal\.tech$',
    r'^https://([a-z0-9-_]+\.)*kausal\.dev$',
]
CORS_ALLOW_HEADERS = list(default_cors_headers) + get_allowed_cors_headers() + [
    INSTANCE_IDENTIFIER_HEADER,
    INSTANCE_HOSTNAME_HEADER,
]
CORS_ALLOW_CREDENTIALS = True
CORS_PREFLIGHT_MAX_AGE = 3600
CORS_ALLOW_ALL_ORIGINS = True

CSRF_COOKIE_AGE = 24 * 3600  # one day

SESSION_COOKIE_SAMESITE = 'None'
SESSION_COOKIE_SECURE = True

GRAPHENE = {
    'SCHEMA': f'{PROJECT_NAME}.schema.schema',
}
GRAPPLE = {
    'APPS': ['pages'],
    'PAGE_INTERFACE': 'pages.page_interface.PageInterface',
}

REST_FRAMEWORK = {
    'PAGE_SIZE': 200,
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.TokenAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.DjangoModelPermissions',
    ],
    'DEFAULT_SCHEMA_CLASS': 'paths.openapi.PathsSchema',
}

SPECTACULAR_SETTINGS = {
    'TITLE': 'Kausal Paths REST API',
    'DESCRIPTION': 'REST API for R/W access to Paths',
    'VERSION': '0.1.0',
    'SERVE_INCLUDE_SCHEMA': False,
    'SCHEMA_PATH_PREFIX': '^/v1',
    'SCHEMA_COERCE_PATH_PK_SUFFIX': True,
}

# If we are using Redis, we also use that for the broker and the results.
# Otherwise, we use a Django database backend.
if REDIS_URL:
    CELERY_BROKER_URL = REDIS_URL
    CELERY_RESULT_BACKEND = REDIS_URL
else:
    CELERY_BROKER_URL = 'django-db'
    CELERY_RESULT_BACKEND = 'django-cache'
    INSTALLED_APPS.append('django_celery_results')

CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 2 * 60  # two minutes
CELERY_WORKER_SEND_TASK_EVENTS = True
CELERY_RESULT_EXPIRES = 60 * 60  # one hour
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = False

def _get_channel_layer_config() -> dict[str, Any]:
    if REDIS_URL:
        return {
            "BACKEND": "channels_redis.core.RedisChannelLayer",
            "CONFIG": {
                "hosts": [REDIS_URL],
                "prefix": "paths-%s-asgi" % DEPLOYMENT_TYPE,
            },
        }
    if not DEBUG:
        warnings.warn("Using in-memory channel layer in non-DEBUG mode. This is not recommended for production.", stacklevel=1)
    return {
        "BACKEND": "channels.layers.InMemoryChannelLayer",
    }


CHANNEL_LAYERS = {
    "default": _get_channel_layer_config(),
}


# Internationalization
# https://docs.djangoproject.com/en/3.1/topics/i18n/

# While Django seems to prefer lower-case regions in language codes (e.g., 'en-us' instead of 'en-US'; cf.
# https://github.com/django/django/blob/main/django/conf/global_settings.py), the Accept-Language header is
# case-insensitive, and Django also seems to be able to deal with upper case.
# https://docs.djangoproject.com/en/4.1/topics/i18n/#term-language-code
# On the other hand, i18next strongly suggests regions to be in upper case lest some features break.
# https://www.i18next.com/how-to/faq#how-should-the-language-codes-be-formatted
# Since we send the language code of a plan to the client, let's make sure we use the upper-case format everywhere in
# the backend already so we don't end up with different formats.
LANGUAGES = (
    ('en', _('English')),
    ('fi', _('Finnish')),
    ('sv', _('Swedish')),
    ('de', _('German')),
    ('de-CH', _('German (Switzerland)')),
    ('cs', _('Czech')),
    ('da', _('Danish')),
    ('pl', _('Polish')),
    ('lv', _('Latvian')),
    ('es-US', _('Spanish (United States)')),
    ('el', _('Greek')),
)
# For languages that Django has no translations for, we need to manually specify what the language is called in that
# language. We use this for displaying the list of available languages in the user settings.
LOCAL_LANGUAGE_NAMES = {
    'de-CH': 'Deutsch (Schweiz)',
    'es-US': 'Español (Estados Unidos)',
}
MODELTRANS_AVAILABLE_LANGUAGES = [x[0].lower() for x in LANGUAGES]
MODELTRANS_FALLBACK = {
    'default': (),
    #'en-au': ('en',),
    #'en-gb': ('en',),
    'de-ch': ('de',),
}  # use language in default_language_field instead of a global fallback


WAGTAIL_CONTENT_LANGUAGES = LANGUAGES
LANGUAGE_CODE = 'en'
TIME_ZONE = 'Europe/Helsinki'
USE_I18N = True
WAGTAIL_I18N_ENABLED = True
USE_TZ = True
LOCALE_PATHS = [
    str(BASE_DIR / Path('locale')),
]

WAGTAILSEARCH_BACKENDS = {
    'default': {
        'BACKEND': 'wagtail.search.backends.database',
        'AUTO_UPDATE': True,
    },
}

WAGTAILADMIN_RICH_TEXT_EDITORS = {
    'default': {'WIDGET': 'wagtail.admin.rich_text.DraftailRichTextArea'},
    'limited': {
        'WIDGET': 'wagtail.admin.rich_text.DraftailRichTextArea',
        'OPTIONS': {
            'features': ['bold', 'italic', 'ol', 'ul', 'link'],
        },
    },
    'very-limited': {
        'WIDGET': 'wagtail.admin.rich_text.DraftailRichTextArea',
        'OPTIONS': {
            'features': ['bold', 'italic'],
        },
    },
}

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.1/howto/static-files/
STATICFILES_FINDERS = [
    'django.contrib.staticfiles.finders.FileSystemFinder',
    'django.contrib.staticfiles.finders.AppDirectoriesFinder',
]

STATICFILES_DIRS = [
    str(BASE_DIR / Path('static_overrides')),
]

common_bundle_dir = Path(BASE_DIR, 'kausal_common/client/dist')
if common_bundle_dir.exists():
    STATICFILES_DIRS.append(str(common_bundle_dir))

MEDIA_FILES_S3_ENDPOINT = env('MEDIA_FILES_S3_ENDPOINT')
MEDIA_FILES_S3_BUCKET = env('MEDIA_FILES_S3_BUCKET')
MEDIA_FILES_S3_ACCESS_KEY_ID = env('MEDIA_FILES_S3_ACCESS_KEY_ID')
MEDIA_FILES_S3_SECRET_ACCESS_KEY = env('MEDIA_FILES_S3_SECRET_ACCESS_KEY')
MEDIA_FILES_S3_CUSTOM_DOMAIN = env('MEDIA_FILES_S3_CUSTOM_DOMAIN')

STORAGES: dict[str, Any] = {
    'default': {
        'BACKEND': 'django.core.files.storage.FileSystemStorage',
    },
    'staticfiles': {
        'BACKEND': 'kausal_common.storage.static.ManifestStaticFilesStorage',
    },
}

if MEDIA_FILES_S3_ENDPOINT:
    STORAGES['default']['BACKEND'] = 'paths.storage.MediaFilesS3Storage'


STATIC_URL = env('STATIC_URL')
MEDIA_URL = env('MEDIA_URL')
STATIC_ROOT = env('STATIC_ROOT')
MEDIA_ROOT = env('MEDIA_ROOT')

# Reverse proxy stuff
USE_X_FORWARDED_HOST = True
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

# Wagtail settings

WAGTAIL_SITE_NAME = PROJECT_NAME
WAGTAIL_ENABLE_UPDATE_CHECK = False
WAGTAIL_PASSWORD_MANAGEMENT_ENABLED = True
WAGTAIL_EMAIL_MANAGEMENT_ENABLED = False
WAGTAIL_PASSWORD_RESET_ENABLED = True
WAGTAILADMIN_PERMITTED_LANGUAGES = list(LANGUAGES)
WAGTAILEMBEDS_RESPONSIVE_HTML = True
WAGTAIL_WORKFLOW_ENABLED = False

# Base URL to use when referring to full URLs within the Wagtail admin backend -
# e.g. in notification emails. Don't include '/admin' or a trailing slash
BASE_URL = env('ADMIN_BASE_URL')
WAGTAILADMIN_BASE_URL = BASE_URL

WATCH_DEFAULT_API_BASE_URL = env('WATCH_DEFAULT_API_BASE_URL')

# Information needed to authentiacte as a GitHub App
GITHUB_APP_ID = env('GITHUB_APP_ID')
GITHUB_APP_PRIVATE_KEY = env('GITHUB_APP_PRIVATE_KEY')

register_common_settings(locals())
# Put type hints for stuff registered in register_common_settings here because mypy doesn't figure it out
# ...

ASGI_APPLICATION = 'paths.asgi.application'

if find_spec('daphne') is not None:
    INSTALLED_APPS.insert(INSTALLED_APPS.index('django.contrib.staticfiles'), 'daphne')


if find_spec('kausal_paths_extensions') is not None:
    INSTALLED_APPS.append('kausal_paths_extensions')
    from kausal_paths_extensions import register_settings

    register_settings(locals())


# local_settings.py can be used to override environment-specific settings
# like database and email that differ between development and production.
local_settings = Path(BASE_DIR) / Path('local_settings.py')
if local_settings.exists():
    import types

    module_name = '%s.local_settings' % ROOT_URLCONF.split('.')[0]
    module = types.ModuleType(module_name)
    module.__file__ = str(local_settings)
    sys.modules[module_name] = module
    exec(local_settings.read_bytes())  # noqa: S102

if not locals().get('SECRET_KEY', ''):
    secret_file = Path(BASE_DIR) / Path('.django_secret')
    try:
        with secret_file.open() as f:
            SECRET_KEY = f.read().strip()
    except OSError:
        import random

        system_random = random.SystemRandom()
        try:
            SECRET_KEY = ''.join([system_random.choice('abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*(-_=+)') for i in range(64)])
            with secret_file.open('w') as f:
                secret_file.chmod(0o0600)
                f.write(SECRET_KEY)
        except OSError:
            raise ImproperlyConfigured(
                'Please create a %s file with random characters to generate your secret key!' % secret_file,
            ) from None


if DEBUG:
    from rich import traceback
    from rich.console import Console

    traceback.install(show_locals=True)

    traceback_console = Console(stderr=True)

    def excepthook(args: ExceptHookArgs):
        assert args.exc_value is not None
        traceback_console.print(
            traceback.Traceback.from_exception(
                args.exc_type,
                args.exc_value,
                args.exc_traceback,
                show_locals=True,
            )
        )

    import threading

    threading.excepthook = excepthook

    from paths.watchfiles_reloader import replace_reloader

    replace_reloader()


LOG_GRAPHQL_QUERIES = DEBUG and env('LOG_GRAPHQL_QUERIES')
LOG_SQL_QUERIES = DEBUG and env('LOG_SQL_QUERIES')
ENABLE_DEBUG_TOOLBAR = DEBUG and env('ENABLE_DEBUG_TOOLBAR')
ENABLE_PERF_TRACING: bool = env('ENABLE_PERF_TRACING')

if ENABLE_DEBUG_TOOLBAR:
    INSTALLED_APPS += ['debug_toolbar']
    MIDDLEWARE.insert(0, 'debug_toolbar.middleware.DebugToolbarMiddleware')

if env('CONFIGURE_LOGGING'):
    from kausal_common.logging.init import LogFormat, UserLoggingOptions, init_logging_django

    is_kube = env.bool('KUBERNETES_MODE') or env.bool('KUBERNETES_LOGGING', False)  # type: ignore
    log_format: LogFormat | None
    if not is_kube and DEBUG:
        # If logfmt hasn't been explicitly selected and DEBUG is on, fall back to autodetection.
        log_format = None
    else:
        log_format = 'logfmt'
    LOGGING = init_logging_django(log_format, options=UserLoggingOptions(sql_queries=LOG_SQL_QUERIES))

REQUEST_LOG_MAX_DAYS = env('REQUEST_LOG_MAX_DAYS')
REQUEST_LOG_METHODS = env('REQUEST_LOG_METHODS')
REQUEST_LOG_IGNORE_PATHS = env('REQUEST_LOG_IGNORE_PATHS')
REQUEST_LOG_MAX_BODY_SIZE = 100 * 1024

if True:
    from kausal_common.sentry.init import init_sentry

    init_sentry(SENTRY_DSN, DEPLOYMENT_TYPE)

HOSTNAME_INSTANCE_DOMAINS = env('HOSTNAME_INSTANCE_DOMAINS')
