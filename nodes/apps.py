from __future__ import annotations

from django.apps import AppConfig


class NodesConfig(AppConfig):
    name = 'nodes'

    def ready(self) -> None:
        import nodes.signals  # noqa: F401
