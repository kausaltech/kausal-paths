"""
paths URL Configuration.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/

Examples
--------
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
from __future__ import annotations

from typing import TYPE_CHECKING

from django.conf import settings
from django.conf.urls.static import static
from django.urls import include, path, re_path
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import TemplateView
from django.views.static import serve as serve_static
from wagtail import urls as wagtail_urls
from wagtail.documents import urls as wagtaildocs_urls

from drf_spectacular.views import SpectacularAPIView, SpectacularRedocView, SpectacularSwaggerView
from social_django import urls as social_urls

#from strawberry.django.views import GraphQLView
from kausal_common.deployment.health_check_view import health_view

from admin_site import urls as admin_urls
from datasets.api import all_routers as datasets_routers
from frameworks.urls import urlpatterns as framework_urls
from nodes.api import all_routers as nodes_routers
from users.views import change_admin_instance

from .api_router import router as api_router
from .graphql_views import PathsGraphQLView

#from .v2_schema import schema as v2_schema

if TYPE_CHECKING:
    from types import ModuleType

#from .schema import federation_schema


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
    # path('django-admin/', admin.site.urls),
    re_path(
        r'^admin/change-admin-instance/(?:(?P<instance_id>\d+)/)?$',
        change_admin_instance,
        name='change-admin-instance',
    ),
    path('admin/', include(admin_urls)),
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
    path('auth/', include(social_urls, namespace='social')),
    path('healthz/', csrf_exempt(health_view), name='healthcheck'),
    path('', include(framework_urls)),
]

if settings.DEBUG:
    #from kausal_common.debugging.memory import memory_trace

    #from debugging.views import memory_tracker

    #urlpatterns.append(path('debug/memory/', csrf_exempt(memory_trace)))
    #urlpatterns.append(path('debug/memory-tracker/', csrf_exempt(memory_tracker)))
    pass

if kpe_urls is not None:
    urlpatterns.append(path('', include(kpe_urls)))

##
#    #path('v1/graphql-subschema/', csrf_exempt(PathsGraphQLView.as_view(schema=federation_schema, graphiql=True)), name='graphql-subschema'),
#    path('v2/graphql/', csrf_exempt(GraphQLView.as_view(schema=v2_schema)), name='graphql_v2'),


if settings.ENABLE_DEBUG_TOOLBAR:
    import debug_toolbar  # type: ignore[import-untyped]

    urlpatterns += [path('__debug__/', include(debug_toolbar.urls))]


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
if settings.DEBUG and settings.STATIC_URL == '/static/':
    urlpatterns += [
        path('static/<path:path>', serve_static, {'document_root': settings.STATIC_ROOT}),
    ]
