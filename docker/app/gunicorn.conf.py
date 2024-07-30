import os
import multiprocessing
from kausal_common.deployment.gunicorn import get_gunicorn_hooks


bind = "127.0.0.1:8000"
workers = min(multiprocessing.cpu_count() * 2 + 1, 4)
threads = workers
wsgi_app = 'paths.wsgi:application'
forwarded_allow_ips = '*'

KUBE_MODE = os.getenv('KUBERNETES_MODE', '') == '1'

if KUBE_MODE:
    preload_app = True

if KUBE_MODE or os.getenv('KUBERNETES_LOGGING', '') == '1':
    logger_class = 'kausal_common.logging.gunicorn.Logger'
else:
    access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
    accesslog = '-'

locals().update(get_gunicorn_hooks())
