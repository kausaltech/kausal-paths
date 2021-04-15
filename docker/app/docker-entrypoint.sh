#!/bin/bash

set -e

/wait-for-it.sh db:5432

cd /code
python manage.py migrate --no-input
# Log to stdout
exec uwsgi --http-socket :8000 --socket :8001 --processes 4 \
    --enable-threads \
    --buffer-size=32768 \
    --static-map /static=/srv/static \
    --static-map /media=/srv/media \
    --module paths.wsgi
