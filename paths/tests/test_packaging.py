"""
The translation harness's dependencies must be declared where a development `uv sync` will see them.

`kausal-paths-extensions` declares them itself, in the `l10n` extra of its `setup.cfg`, and that is
the right place: the extra's comment keeps `anthropic` and `lxml` out of the web and worker images,
which never run a translation pass. But in development the package is reached by a symlink and
`PYTHONPATH` rather than installed from the index, so nothing consults that extra and `uv sync`
leaves the harness unimportable — `translations/markup.py` derives its inline-tag whitelist from
`lxml.html.defs` at import time, so the failure is not a missing feature but an ImportError.

Hence the `l10n` dependency *group* here, which `mise`'s `uv sync --all-groups --all-extras` picks up
and `kausal_common/docker/Dockerfile` (which syncs only `prod`) does not. Two declarations of one
fact, so the first test derives one from the other wherever both are on disk.
"""

import configparser
import tomllib
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

# Nothing here touches the database, but the root conftest's autouse `instance_config` fixture does.
pytestmark = pytest.mark.django_db

REPO_ROOT = Path(__file__).resolve().parents[2]

# The symlink resolves into the kausal-extensions checkout, whose package directory sits beside the
# setup.cfg that declares the extra. Absent in any deployment, where the package is installed.
EXTENSIONS_SETUP_CFG = (REPO_ROOT / 'kausal_paths_extensions').resolve().parent / 'setup.cfg'

EXTRA_NAME = 'l10n'


def _requirements(specifiers: list[str]) -> dict[str, str]:
    """Map normalised distribution name to the version constraint declared with it, if any."""
    requirements = {}
    for spec in specifiers:
        text = spec.split(';')[0].split('#')[0].strip()
        if not text:
            continue
        name = text
        for boundary in ('[', '=', '<', '>', '!', '~', ' '):
            name = name.split(boundary)[0]
        name = name.strip()
        if not name:
            continue
        requirements[name.lower().replace('_', '-')] = text[len(name) :].replace(' ', '')
    return requirements


def _l10n_group() -> dict[str, str]:
    pyproject = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text())
    entries = pyproject['dependency-groups'][EXTRA_NAME]
    # Groups may hold `{include-group = ...}` tables as well as requirement strings.
    return _requirements([entry for entry in entries if isinstance(entry, str)])


def test_the_group_mirrors_the_extra_the_extensions_declare():
    """Derived rather than listed: an extension that gains a dependency must not go unnoticed here."""
    if not EXTENSIONS_SETUP_CFG.exists():
        pytest.skip(f'{EXTENSIONS_SETUP_CFG} is only present in a symlinked development checkout')

    cfg = configparser.ConfigParser()
    cfg.read_string(EXTENSIONS_SETUP_CFG.read_text())
    upstream = _requirements(cfg['options.extras_require'][EXTRA_NAME].split('\n'))
    group = _l10n_group()

    assert upstream, f'no [options.extras_require] {EXTRA_NAME} in {EXTENSIONS_SETUP_CFG}'
    for name, constraint in upstream.items():
        assert name in group, f'{name} is in the extra but not the group'
        # The constraint, not just the name: `anthropic` is pinned below 1.x on both sides because
        # 1.x moved the SDK to `httpx2`, and a pin that holds on one side only is no pin at all.
        assert group[name] == constraint, f'{name} declared as {constraint!r} upstream, {group[name]!r} here'


def test_the_group_carries_the_import_time_dependency():
    """Unconditional, since the test above skips wherever the extensions are installed rather than symlinked."""
    assert 'lxml' in _l10n_group()


def test_the_lock_file_resolves_the_group():
    """
    A group `uv.lock` has not caught up with fails the `--locked` sync in the Dockerfile.

    The version is checked, not just the presence of a row: a constraint added after the lock was
    written leaves a resolution that violates it, and the name alone cannot see that.
    """
    lock = tomllib.loads((REPO_ROOT / 'uv.lock').read_text())
    locked = {package['name'].lower().replace('_', '-'): package['version'] for package in lock['package']}

    for name, constraint in _l10n_group().items():
        assert name in locked, f'{name} is declared but absent from uv.lock; re-run `uv lock`'
        specifier = SpecifierSet(constraint)
        assert specifier.contains(Version(locked[name])), (
            f'uv.lock pins {name} {locked[name]}, which does not satisfy {constraint!r}; re-run `uv lock`'
        )
