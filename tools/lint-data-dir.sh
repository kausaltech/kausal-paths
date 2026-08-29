#!/usr/bin/env bash
#
# Lint and type-check the scripts under data/.
#
# data/ is gitignored, which means the ordinary checks never look at it:
#
#   * pre-commit only ever hands hooks git-tracked files, so run-ruff and
#     run-mypy-pre-push pass while ignoring every extractor;
#   * `ruff check data/` prints "No Python files found under the given path(s)"
#     and then "All checks passed!", because ruff respects .gitignore.
#
# The code is real all the same: data/**/*.py is published to the private scripts
# repo by data/copy_documentation_files.py and committed there, so it is
# version-controlled code with readers.
#
# Nothing here is required to exist. On a checkout without a data/ directory, or
# with no Python in it, the hook is a no-op and says so.

set -euo pipefail

if [ ! -d data ]; then
    echo "no data/ directory here; nothing to check"
    exit 0
fi

if [ -z "$(find data -name '*.py' -not -path '*/__pycache__/*' -print -quit)" ]; then
    echo "no Python files under data/; nothing to check"
    exit 0
fi

status=0

# --no-respect-gitignore is the whole point: without it ruff walks past data/.
echo "==> ruff check data/"
ruff check --no-respect-gitignore data/ || status=1

echo "==> ruff format --check data/"
ruff format --check --no-respect-gitignore data/ || status=1

# mypy does not consult .gitignore, so it needs no equivalent flag.
echo "==> mypy data/"
mypy data/ || status=1

exit "$status"
