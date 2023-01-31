from django.urls import path

from wagtail.admin import urls as wagtailadmin_urls

from .api import check_login_method

wagtail_urls = wagtailadmin_urls.urlpatterns

urlpatterns = [
    path('login/check/', check_login_method, name='admin_check_login_method'),
    *wagtail_urls,
]
