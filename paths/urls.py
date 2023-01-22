"""paths URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from types import ModuleType

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include, re_path
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import TemplateView

from wagtail.core import urls as wagtail_urls
from wagtail.admin import urls as wagtailadmin_urls
from wagtail.documents import urls as wagtaildocs_urls
from drf_spectacular.views import SpectacularAPIView, SpectacularRedocView, SpectacularSwaggerView

from .graphql_views import PathsGraphQLView
from users.views import change_admin_instance
from nodes.api import all_routers as nodes_routers
from datasets.api import all_routers as datasets_routers
from .api_router import router as api_router



kpe_urls: ModuleType | None
try:
    from kausal_paths_extensions import urls as kpe_urls
except ImportError:
    kpe_urls = None


api_urls = [
    path(r'', include(api_router.urls)),
    *[path(r'', include(r.urls)) for r in nodes_routers],
    *[path(r'', include(r.urls)) for r in datasets_routers],
]


urlpatterns = [
    path('django-admin/', admin.site.urls),
    re_path(
        r'^admin/change-admin-instance/(?:(?P<instance_id>\d+)/)?$',
        change_admin_instance,
        name='change-admin-instance'
    ),
    path('admin/', include(wagtailadmin_urls)),
    path('documents/', include(wagtaildocs_urls)),
    # For anything not caught by a more specific rule above, hand over to
    # Wagtail's serving mechanism
    path('pages/', include(wagtail_urls)),

    path('v1/graphql/docs/', TemplateView.as_view(
        template_name='graphql-voyager.html',
    ), name='graphql-voyager'),

    path('v1/graphql/', csrf_exempt(PathsGraphQLView.as_view(graphiql=True)), name='graphql'),
    path('v1/', include(api_urls)),
    path('v1/schema/', SpectacularAPIView.as_view(urlconf=api_urls), name='schema'),
    path('v1/schema/swagger-ui/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('v1/schema/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),

]

if kpe_urls is not None:
    urlpatterns.append(path('', include(kpe_urls)))

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
