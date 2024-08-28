from __future__ import annotations

from django.conf import settings
from django.urls import URLPattern, URLResolver, path
from wagtail.admin import urls as wagtailadmin_urls

from .api import check_login_method

wagtail_urls: list[URLPattern | URLResolver] = list(wagtailadmin_urls.urlpatterns)

if settings.DEBUG:
    fallback = wagtail_urls[-1]
    if fallback.callback and fallback.callback.__name__ == 'default':
        wagtail_urls.remove(fallback)

urlpatterns = [
    path('login/check/', check_login_method, name='admin_check_login_method'),
    *wagtail_urls,
]
