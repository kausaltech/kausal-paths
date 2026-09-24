from __future__ import annotations

import os

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'paths.settings')

app = Celery('paths')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object('django.conf:settings', namespace='CELERY')


@worker_process_init.connect(weak=False)
def init_worker_metrics(*args, **kwargs):
    if not os.getenv('OTEL_EXPORTER_OTLP_METRICS_ENDPOINT'):
        return

    from opentelemetry.instrumentation.celery import CeleryInstrumentor
    from opentelemetry.trace import NoOpTracerProvider

    from kausal_common.telemetry.metrics import init_metrics

    init_metrics()
    CeleryInstrumentor().instrument(tracer_provider=NoOpTracerProvider())


@worker_process_shutdown.connect(weak=False)
def shutdown_worker_metrics(*args, **kwargs):
    if not os.getenv('OTEL_EXPORTER_OTLP_METRICS_ENDPOINT'):
        return

    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider

    provider = metrics.get_meter_provider()
    if isinstance(provider, MeterProvider):
        provider.shutdown(timeout_millis=2_500)


# Load task modules from all registered Django apps.
app.autodiscover_tasks()

app.conf.result_backend_transport_options = {
    'global_keyprefix': 'paths-',
}
