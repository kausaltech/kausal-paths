from django.apps import AppConfig


class NodesConfig(AppConfig):
    name = 'nodes'

    def ready(self) -> None:
        from nodes.units import add_unit_translations

        add_unit_translations()
