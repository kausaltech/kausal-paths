from __future__ import annotations

from django.apps import AppConfig


class RequestLogConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'request_log'
