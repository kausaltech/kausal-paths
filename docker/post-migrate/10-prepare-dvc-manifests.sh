#!/bin/bash
set -eo pipefail

# Metadata preparation is an optimization; missing manifests are resolved at runtime.
if timeout --kill-after=5s "${PREPARE_DVC_MANIFESTS_TIMEOUT:-300}s" \
    python manage.py prepare_dvc_manifests --in-customer-use; then
    echo 'DVC source manifests prepared.'
else
    status=$?
    echo "DVC manifest preparation exited with status $status; runtime preparation remains available." >&2
fi
