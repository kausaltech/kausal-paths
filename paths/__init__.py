import os
from django.conf import settings

from nodes.instance import InstanceLoader

loader = InstanceLoader(os.path.join(settings.BASE_DIR, 'configs/tampere.yaml'))
loader.print_graph()
loader.context.generate_baseline_values()
