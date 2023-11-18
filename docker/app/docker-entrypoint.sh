#!/bin/bash

set -e

/wait-for-it.sh db:5432

cd /code
python manage.py migrate --no-input
# Clear the caches before uwsgi starts
python manage.py shell -c "from django.core.cache import cache; cache.clear()"

# Log to stdout
exec uwsgi --http11-socket :8000 --socket :8001 --processes 8 \
    --enable-threads \
    --buffer-size=32768 \
    --static-map /static=/srv/static \
    --static-map /media=/srv/media \
    --module paths.wsgi
