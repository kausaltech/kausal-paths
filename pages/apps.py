from django.apps import AppConfig


class PagesConfig(AppConfig):
    name = 'pages'

    def ready(self):
        from pages.schema import monkeypatch_grapple

        monkeypatch_grapple()
