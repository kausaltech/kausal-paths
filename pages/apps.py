import os
from django.apps import AppConfig
from django.conf import settings

from nodes.instance import InstanceLoader

loader = None


class PagesConfig(AppConfig):
    name = 'pages'

    def ready(self):
        global loader
        loader = InstanceLoader.from_yaml(os.path.join(settings.BASE_DIR, 'configs/tampere.yaml'))
        loader.print_graph()
        loader.context.generate_baseline_values()
