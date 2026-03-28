from __future__ import annotations

from django.apps import AppConfig


class NodesConfig(AppConfig):
    name = 'nodes'

    def ready(self) -> None:
        from kausal_common.i18n.pydantic import on_app_ready

        import nodes.signals  # noqa: F401  # pyright: ignore[reportUnusedImport]

        on_app_ready()
