from django.apps import AppConfig


class FrameworksConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'frameworks'

    def ready(self) -> None:
        import frameworks.signals  # noqa: F401
