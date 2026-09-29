from __future__ import annotations

from django.apps import AppConfig


class NodesConfig(AppConfig):
    name = 'nodes'

    def ready(self) -> None:
        from kausal_common.i18n.pydantic import on_app_ready

        import common  # noqa: F401  # pyright: ignore[reportUnusedImport]
        import nodes.signals  # noqa: F401  # pyright: ignore[reportUnusedImport]
        from nodes.units import add_unit_translations

        on_app_ready()
        add_unit_translations()
