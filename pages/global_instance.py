import os
from django.conf import settings

from nodes.instance import InstanceLoader

instance = None

if settings.INSTANCE_LOADER_CONFIG is not None:
    loader = InstanceLoader.from_yaml(os.path.join(settings.BASE_DIR, settings.INSTANCE_LOADER_CONFIG))
    instance = loader.instance
    instance.context.print_graph()
    instance.context.generate_baseline_values()
