#!/bin/bash
set -eo pipefail

# Keep startup bounded even when a model or remote dataset is unavailable.
# Inherit the server's user, environment and on-disk cache directories.
if timeout --kill-after=5s "${COMPUTE_INSTANCES_TIMEOUT:-120}s" \
    python manage.py compute_instances --in-customer-use; then
    echo 'Instance warm-up completed.'
else
    status=$?
    echo "Instance warm-up exited with status $status; continuing server startup." >&2
fi
