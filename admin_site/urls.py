from __future__ import annotations

from django.conf import settings
from django.urls import URLPattern, URLResolver, path
from django.urls.conf import include, re_path
from rest_framework import routers
from wagtail.admin import urls as wagtailadmin_urls

from orgs.api import all_views as org_views
from orgs.autocomplete import OrganizationAutocomplete
from people.api import all_views as people_views

from .api import check_login_method

wagtail_urls: list[URLPattern | URLResolver] = list(wagtailadmin_urls.urlpatterns)

admin_api_router = routers.DefaultRouter()

if settings.DEBUG:
    fallback = wagtail_urls[-1]
    if fallback.callback and fallback.callback.__name__ == 'default':
        wagtail_urls.remove(fallback)

for view in org_views + people_views:
    basename = view.get('basename') or admin_api_router.get_default_basename(view['class'])
    if admin_api_router.is_already_registered(basename):
        continue
    admin_api_router.register(view['name'], view['class'], basename=view.get('basename'))


urlpatterns = [
    path('login/check/', check_login_method, name='admin_check_login_method'),
    re_path(r'^org-autocomplete/$', OrganizationAutocomplete.as_view(), name='organization-autocomplete'),
    path('v1/', include(admin_api_router.urls)),
    *wagtail_urls,
]
