import os
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from nodes.instance import InstanceLoader


if settings.INSTANCE_LOADER_CONFIG is None:
    raise ImproperlyConfigured('Global instance requires the INSTANCE_LOADER_CONFIG setting')

loader = InstanceLoader.from_yaml(os.path.join(settings.BASE_DIR, settings.INSTANCE_LOADER_CONFIG))
instance = loader.instance
instance.context.print_graph()
instance.context.generate_baseline_values()
