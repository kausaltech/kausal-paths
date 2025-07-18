from django.apps import AppConfig
from django.utils.translation import gettext_lazy as _

class PeopleConfig(AppConfig):
    name = 'people'
    verbose_name = _('People')

    def ready(self):
        # Register feature detection library
        from willow.registry import registry
        try:
            import rustface.willow
        except ImportError:
            pass
        else:
            registry.register_plugin(rustface.willow)
