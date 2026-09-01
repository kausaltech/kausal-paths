# Migrating an existing checkout to the src layout

Commit `382b1a99` ("Build Paths from a src layout") moved every application package under
`src/`, switched the build backend from setuptools to `uv_build` with an explicit module
list, and removed the `PYTHONPATH = "{{ config_root }}"` override from `mise.toml`.

A fresh clone needs nothing from this page. **An existing working tree does**, and it fails
in a way that does not point at the cause.

## The symptom

Anything that starts Django stops working, naming the settings module rather than the
layout:

```
$ python manage.py shell_plus
ModuleNotFoundError: No module named 'paths.settings'
```

`src/paths/settings.py` is right there. `git status` is clean. Nothing looks deleted.

## Why it happens

Git removes the files it tracks, but it does not remove a directory that still has
untracked content in it — and every one of those package directories had a
`__pycache__/`. So after the checkout the old locations survive as directories that are
empty of source but not empty:

```
paths/
└── __pycache__/          # 124 stale .pyc files
```

Three things then combine:

1. A directory with no `__init__.py` is still importable — Python 3 treats it as a
   **namespace package**. So `import paths` succeeds and binds to the empty leftover.
2. `manage.py` lives at the repository root, and Python puts a script's own directory
   first on `sys.path`. The leftover therefore wins over the installed package.
3. The leftover has no `settings` submodule, so the failure surfaces one level down, as a
   missing `paths.settings` rather than a missing `paths`.

`import paths; print(paths.__file__)` printing `None` is the tell: a real package reports a
file, a namespace package reports nothing.

The second half is the editable install. The old `.venv` carries a setuptools-generated
finder whose mapping was written for the flat layout, and after the move it resolves
nothing:

```python
# .venv/.../__editable___kausal_paths_1_0_0_finder.py
MAPPING: dict[str, str] = {}
```

Removing the leftovers alone therefore changes the error from `No module named
'paths.settings'` to `No module named 'paths'`. Both steps are needed.

## The fix

Delete the stale package directories, then rebuild the environment:

```bash
rm -rf admin_site common datasets frameworks gql_client nodes optimizer orgs \
       pages params paths people request_log users
mise prepare
```

The authoritative list is `module-name` under `[tool.uv.build-backend]` in
`pyproject.toml` — read it from there rather than from this page, which will age.

Nothing in those directories is tracked, so a guard is cheap and worth using if you would
rather not `rm -rf` a list of names by hand:

```bash
for d in $(python -c "import tomllib;print(' '.join(tomllib.load(open('pyproject.toml','rb'))['tool']['uv']['build-backend']['module-name']))"); do
  [ -d "$d" ] || continue
  if [ -z "$(git ls-files "$d")" ] && [ -z "$(find "$d" -type f -not -path '*__pycache__*' ! -name '.DS_Store')" ]; then
    rm -rf "$d" && echo "removed $d"
  else
    echo "KEPT $d — it still has real content, look before deleting"
  fi
done
```

Verify with something that exercises both the install and the app:

```bash
python manage.py check
python -m tools.debug_instance -i <some-instance> --node net_emissions
```

## What this does not affect

* **Container builds.** The image is built from a fresh context, `**/__pycache__/` is in
  `.dockerignore`, and `kausal_common/docker/Dockerfile` runs `./manage.py check` during
  the build — so a layout problem fails the build rather than shipping. The trap is
  specific to long-lived working trees.
* **CI.** Same reason: it checks out clean.
* **The database, DVC data and instance configs.** This is a code-layout change only.

## Two things to know afterwards

* **`PYTHONPATH` no longer includes the repository root.** Anything that relied on it —
  an ad-hoc script, an editor run configuration, a cron entry — now depends on the
  package being installed instead. Scripts under `data/` that do
  `sys.path.insert(0, str(Path.cwd()))` before `django.setup()` keep working, but the
  insert is now redundant: it was reaching the flat packages, and the installed
  distribution is what answers now.
* **`tools/`, `configs/`, `data/`, `docs/` and `kausal_common/` did not move.** Only the
  application packages did, so import paths in code are unchanged — `from nodes.node
  import Node` still reads the same.
