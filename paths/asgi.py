"""
ASGI config for paths project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/asgi/
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

from django.core.asgi import get_asgi_application

from channels.routing import ProtocolTypeRouter

from kausal_common.asgi.middleware import HTTPMiddleware, WebSocketMiddleware

if TYPE_CHECKING:
    from collections.abc import Callable

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'paths.settings')

django_asgi_app = get_asgi_application()

class AuthGraphQLProtocolTypeRouter(ProtocolTypeRouter):
    def __init__(self):
        from django.urls import URLPattern, URLResolver, re_path

        from channels.routing import URLRouter

        from .graphql_views import PathsGraphQLHTTPConsumer, PathsGraphQLWSConsumer
        from .schema import schema

        re_path_any = cast('Callable[[str, Any], URLPattern | URLResolver]', re_path)
        gql_url_pattern = r"^v1/graphql/$"
        http_urls: list[URLPattern | URLResolver] = [
            re_path_any(gql_url_pattern, PathsGraphQLHTTPConsumer.as_asgi(schema=schema)),
        ]
        http_urls.append(re_path_any(r"^", django_asgi_app))

        super().__init__(
            {
                "http": HTTPMiddleware(
                        URLRouter(
                            http_urls,
                        ),
                ),
                "websocket": WebSocketMiddleware(
                    URLRouter(
                        [
                            re_path_any(
                                gql_url_pattern,
                                PathsGraphQLWSConsumer.as_asgi(schema=schema),
                            ),
                        ],
                    ),
                ),
            },
        )


application = AuthGraphQLProtocolTypeRouter()
