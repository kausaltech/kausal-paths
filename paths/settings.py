"""
Django settings for paths project.

Generated by 'django-admin startproject' using Django 3.1.8.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/3.1/ref/settings/
"""

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
import os
from importlib.util import find_spec
from typing import Literal

import environ
from corsheaders.defaults import default_headers as default_cors_headers  # noqa
from django.utils.translation import gettext_lazy as _


PROJECT_NAME = 'paths'

root = environ.Path(__file__) - 2  # two folders back
env = environ.FileAwareEnv(
    ENV_FILE=(str, ''),
    DEBUG=(bool, False),
    DEPLOYMENT_TYPE=(str, 'development'),
    SECRET_KEY=(str, ''),
    AZURE_AD_CLIENT_ID=(str, ''),
    AZURE_AD_CLIENT_SECRET=(str, ''),
    ALLOWED_HOSTS=(list, ['*']),
    DATABASE_URL=(str, f'postgresql:///{PROJECT_NAME}'),
    CACHE_URL=(str, 'locmemcache://'),
    MEDIA_ROOT=(environ.Path(), root('media')),
    STATIC_ROOT=(environ.Path(), root('static')),
    ADMIN_BASE_URL=(str, 'http://localhost:8000'),
    MEDIA_URL=(str, '/media/'),
    STATIC_URL=(str, '/static/'),
    SENTRY_DSN=(str, ''),
    COOKIE_PREFIX=(str, PROJECT_NAME),
    SERVER_EMAIL=(str, 'noreply@kausal.tech'),
    DEFAULT_FROM_EMAIL=(str, 'noreply@kausal.tech'),
    INTERNAL_IPS=(list, []),
    HOSTNAME_INSTANCE_DOMAINS=(list, ['localhost']),
    CONFIGURE_LOGGING=(bool, True),
    LOG_GRAPHQL_QUERIES=(bool, True),
    AWS_S3_ENDPOINT_URL=(str, ''),
    AWS_STORAGE_BUCKET_NAME=(str, ''),
    AWS_ACCESS_KEY_ID=(str, ''),
    AWS_SECRET_ACCESS_KEY=(str, ''),
)

BASE_DIR = root()
PROJECT_DIR = os.path.join(BASE_DIR, PROJECT_NAME)

if env('ENV_FILE'):
    environ.Env.read_env(env('ENV_FILE'))
elif os.path.exists(os.path.join(BASE_DIR, '.env')):
    environ.Env.read_env(os.path.join(BASE_DIR, '.env'))

DEBUG = env('DEBUG')
ADMIN_BASE_URL = env('ADMIN_BASE_URL')
ALLOWED_HOSTS = env('ALLOWED_HOSTS')
INTERNAL_IPS = env.list('INTERNAL_IPS', default=(['127.0.0.1'] if DEBUG else []))
DATABASES = {
    'default': env.db()
}
DATABASES['default']['ATOMIC_REQUESTS'] = True

CACHES = {
    'default': env.cache(),
}

SECRET_KEY = env('SECRET_KEY')

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

AUTH_USER_MODEL = 'users.User'

LOGIN_URL = '/admin/login/'

# Application definition

INSTALLED_APPS = [
    'wagtail.contrib.forms',
    'wagtail.contrib.redirects',
    'wagtail.embeds',
    'wagtail.sites',
    'users',  # must be before wagtail.users
    'admin_site',  # must be before wagtail.admin
    'wagtail.users',
    'wagtail.snippets',
    'wagtail.documents',
    'wagtail.images',
    'wagtail.search',
    'wagtail.admin',
    'wagtail.core',
    'wagtail.contrib.modeladmin',
    'wagtail.contrib.styleguide',
    'wagtail_localize',
    'wagtail_localize.locales',  # replaces `wagtail.locales`
    'wagtailfontawesome',

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
    'datasets',
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
    'wagtail.contrib.redirects.middleware.RedirectMiddleware',
    'admin_site.middleware.AuthExceptionMiddleware',
    'paths.middleware.AdminMiddleware',
]

ROOT_URLCONF = f'{PROJECT_NAME}.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
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

AUTHENTICATION_BACKENDS = (
    'admin_site.auth_backends.AzureADAuth',
    'django.contrib.auth.backends.ModelBackend',
)

SOCIAL_AUTH_AZURE_AD_KEY = env.str('AZURE_AD_CLIENT_ID')
SOCIAL_AUTH_AZURE_AD_SECRET = env.str('AZURE_AD_CLIENT_SECRET')

SOCIAL_AUTH_PIPELINE = (
    'admin_site.auth_pipeline.log_login_attempt',

    # Get the information we can about the user and return it in a simple
    # format to create the user instance later. On some cases the details are
    # already part of the auth response from the provider, but sometimes this
    # could hit a provider API.
    'social_core.pipeline.social_auth.social_details',

    # Get the social uid from whichever service we're authing thru. The uid is
    # the unique identifier of the given user in the provider.
    'social_core.pipeline.social_auth.social_uid',

    # Generate username from UUID
    'admin_site.auth_pipeline.get_username',

    # Checks if the current social-account is already associated in the site.
    'social_core.pipeline.social_auth.social_user',

    # Finds user by email address
    'admin_site.auth_pipeline.find_user_by_email',

    # Get or create the user and update user data
    'admin_site.auth_pipeline.create_or_update_user',

    # Create the record that associated the social account with this user.
    'social_core.pipeline.social_auth.associate_user',

    # Populate the extra_data field in the social record with the values
    # specified by settings (and the default ones like access_token, etc).
    'social_core.pipeline.social_auth.load_extra_data',

    # Update avatar photo from MS Graph
    'admin_site.auth_pipeline.update_avatar',
)



INSTANCE_IDENTIFIER_HEADER = 'x-paths-instance-identifier'
INSTANCE_HOSTNAME_HEADER = 'x-paths-instance-hostname'

CORS_ALLOWED_ORIGIN_REGEXES = [
    # Match localhost with optional port
    r'^https?://([a-z0-9-_]+\.)+localhost(:\d+)?$',
    r'^https://([a-z0-9-_]+\.)*kausal\.tech$'
]
CORS_ALLOW_HEADERS = list(default_cors_headers) + [
    'sentry-trace',
]
CORS_ALLOW_CREDENTIALS = True
CORS_PREFLIGHT_MAX_AGE = 3600
CORS_ALLOW_ALL_ORIGINS = True

SESSION_COOKIE_SAMESITE = 'None'
SESSION_COOKIE_SECURE = True

GRAPHENE = {
    'SCHEMA': f'{PROJECT_NAME}.schema.schema',
}
GRAPPLE = {
    'APPS': ['pages'],
}

REST_FRAMEWORK = {
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
    ('fi', _('Finnish')),
    ('en', _('English')),
    ('sv', _('Swedish')),
    ('de', _('German')),
    ('de-CH', _('German (Switzerland)')),
)
# For languages that Django has no translations for, we need to manually specify what the language is called in that
# language. We use this for displaying the list of available languages in the user settings.
LOCAL_LANGUAGE_NAMES = {
    'de-CH': "Deutsch (Schweiz)",
}
MODELTRANS_AVAILABLE_LANGUAGES = [x[0].lower() for x in LANGUAGES]
MODELTRANS_FALLBACK = {
    'default': (),
    #'en-au': ('en',),
    #'en-gb': ('en',),
    'de-ch': ('de',),
}  # use language in default_language_field instead of a global fallback


WAGTAIL_CONTENT_LANGUAGES = LANGUAGES
LANGUAGE_CODE = 'fi'
TIME_ZONE = 'Europe/Helsinki'
USE_I18N = True
WAGTAIL_I18N_ENABLED = True
USE_L10N = True
USE_TZ = True
LOCALE_PATHS = [
    os.path.join(BASE_DIR, 'locale')
]

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.1/howto/static-files/
STATICFILES_FINDERS = [
    'django.contrib.staticfiles.finders.FileSystemFinder',
    'django.contrib.staticfiles.finders.AppDirectoriesFinder',
]

# STATICFILES_DIRS = [
#     os.path.join(PROJECT_DIR, 'static'),
# ]

STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.ManifestStaticFilesStorage'

AWS_S3_ENDPOINT_URL = env('AWS_S3_ENDPOINT_URL')
if AWS_S3_ENDPOINT_URL:
    DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
    AWS_STORAGE_BUCKET_NAME = env('AWS_STORAGE_BUCKET_NAME')
    AWS_ACCESS_KEY_ID=env('AWS_ACCESS_KEY_ID'),
    AWS_SECRET_ACCESS_KEY=env('AWS_SECRET_ACCESS_KEY'),

STATIC_URL = env('STATIC_URL')
MEDIA_URL = env('MEDIA_URL')
STATIC_ROOT = env('STATIC_ROOT')
MEDIA_ROOT = env('MEDIA_ROOT')

# Reverse proxy stuff
USE_X_FORWARDED_HOST = True
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

DEPLOYMENT_TYPE = env('DEPLOYMENT_TYPE')
SENTRY_DSN = env('SENTRY_DSN')

# Wagtail settings

WAGTAIL_SITE_NAME = PROJECT_NAME
WAGTAIL_ENABLE_UPDATE_CHECK = False
WAGTAIL_PASSWORD_MANAGEMENT_ENABLED = True
WAGTAIL_EMAIL_MANAGEMENT_ENABLED = False
WAGTAIL_PASSWORD_RESET_ENABLED = True
WAGTAILADMIN_PERMITTED_LANGUAGES = list(LANGUAGES)

# Base URL to use when referring to full URLs within the Wagtail admin backend -
# e.g. in notification emails. Don't include '/admin' or a trailing slash
BASE_URL = env('ADMIN_BASE_URL')
WAGTAILADMIN_BASE_URL = BASE_URL


INSTANCE_LOADER_CONFIG = 'configs/tampere.yaml'

if find_spec('kausal_paths_extensions') is not None:
    INSTALLED_APPS.append('kausal_paths_extensions')
    from kausal_paths_extensions import register_settings
    register_settings(locals())


# local_settings.py can be used to override environment-specific settings
# like database and email that differ between development and production.
f = os.path.join(BASE_DIR, "local_settings.py")
if os.path.exists(f):
    import sys
    import types
    module_name = "%s.local_settings" % ROOT_URLCONF.split('.')[0]
    module = types.ModuleType(module_name)
    module.__file__ = f
    sys.modules[module_name] = module
    exec(open(f, "rb").read())  # noqa

if not locals().get('SECRET_KEY', ''):
    secret_file = os.path.join(BASE_DIR, '.django_secret')
    try:
        SECRET_KEY = open(secret_file).read().strip()
    except IOError:
        import random
        system_random = random.SystemRandom()
        try:
            SECRET_KEY = ''.join([system_random.choice('abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*(-_=+)') for i in range(64)])  # noqa
            secret = open(secret_file, 'w')
            import os
            os.chmod(secret_file, 0o0600)
            secret.write(SECRET_KEY)
            secret.close()
        except IOError:
            Exception('Please create a %s file with random characters to generate your secret key!' % secret_file)


if DEBUG:
    from rich.traceback import install
    install()

    from paths.watchfiles_reloader import replace_reloader
    replace_reloader()


LOG_GRAPHQL_QUERIES = env('LOG_GRAPHQL_QUERIES')

if env('CONFIGURE_LOGGING') and 'LOGGING' not in locals():
    def level(level: Literal['DEBUG', 'INFO', 'WARNING']):
        return dict(
            handlers=['rich' if DEBUG else 'console'],
            propagate=False,
            level=level,
        )

    LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'verbose': {
                'format': '%(levelname)s %(asctime)s %(module)s %(process)d %(thread)d %(message)s'
            },
            'simple': {
                'format': '%(levelname)s %(name)s %(asctime)s %(message)s'
            },
            'rich': {
                'format': '%(message)s'
            },
        },
        'handlers': {
            'null': {
                'level': 'DEBUG',
                'class': 'logging.NullHandler',
            },
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'simple'
            },
            'rich': {
                'level': 'DEBUG',
                'class': 'paths.log_handler.LogHandler',
                'formatter': 'rich',
                'log_time_format': '%Y-%m-%d %H:%M:%S.%f'
            },
        },
        'loggers': {
            'django.db': level('INFO'),
            'django.template': level('WARNING'),
            'django.utils.autoreload': level('INFO'),
            'django': level('DEBUG'),
            'raven': level('WARNING'),
            'blib2to3': level('INFO'),
            'generic': level('DEBUG'),
            'parso': level('WARNING'),
            'requests': level('WARNING'),
            'urllib3.connectionpool': level('INFO'),
            'elasticsearch': level('WARNING'),
            'PIL': level('INFO'),
            'faker': level('INFO'),
            'factory': level('INFO'),
            'watchfiles': level('INFO'),
            'watchdog': level('INFO'),
            'git': level('INFO'),
            'pint': level('INFO'),
            'matplotlib': level('INFO'),
            '': level('DEBUG'),
        }
    }


if SENTRY_DSN:
    import sentry_sdk
    from sentry_sdk.integrations.django import DjangoIntegration

    sentry_sdk.init(
        dsn=SENTRY_DSN,
        send_default_pii=True,
        traces_sample_rate=1.0,
        integrations=[DjangoIntegration()],
        environment=DEPLOYMENT_TYPE,
    )

if 'DATABASES' in locals():
    if DATABASES['default']['ENGINE'] in ('django.db.backends.postgresql', 'django.contrib.gis.db.backends.postgis'):
        DATABASES['default']['CONN_MAX_AGE'] = 600

CORS_ALLOW_HEADERS.append(INSTANCE_HOSTNAME_HEADER)
CORS_ALLOW_HEADERS.append(INSTANCE_IDENTIFIER_HEADER)

HOSTNAME_INSTANCE_DOMAINS = env('HOSTNAME_INSTANCE_DOMAINS')

if DEBUG:
    try:
        import django_stubs_ext
        django_stubs_ext.monkeypatch()
    except ImportError:
        pass
