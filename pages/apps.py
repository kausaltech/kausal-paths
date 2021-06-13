import os
from django.apps import AppConfig
from django.conf import settings

from nodes.instance import InstanceLoader

instance = None


class PagesConfig(AppConfig):
    name = 'pages'

    def ready(self):
        if settings.INSTANCE_LOADER_CONFIG is not None:
            global instance
            loader = InstanceLoader.from_yaml(os.path.join(settings.BASE_DIR, settings.INSTANCE_LOADER_CONFIG))
            instance = loader.instance
            instance.context.print_graph()
            instance.context.generate_baseline_values()
