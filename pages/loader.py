import os
from nodes.instance import InstanceLoader
from django.conf import settings

loader = InstanceLoader(os.path.join(settings.BASE_DIR, 'configs/tampere.yaml'))
loader.print_graph()
loader.context.generate_baseline_values()
