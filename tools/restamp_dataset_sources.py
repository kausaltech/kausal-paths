r"""
Attach data source provenance to DVC datasets that have none, without touching their data.

    python -m tools.restamp_dataset_sources --sources-csv "$PATHS_DATA/modules/bisko/bisko_sources.csv"
    python -m tools.restamp_dataset_sources --sources-csv ... --apply
    python -m tools.restamp_dataset_sources --sources-csv ... de/emissionsfaktoren_endenergie

Nothing is written without ``--apply``.

**Do one dataset first.** The dry run exercises everything except the push, so the first ``--apply``
of a batch is also the first exercise of the write path. Stamp a small, low-stakes dataset on its
own (``de/biodieselanteil`` is 34 rows), read it back, and only then run the rest:

    python -m tools.restamp_dataset_sources --sources-csv ... de/biodieselanteil --apply

## Why this exists rather than a re-upload

``upload_new_dataset`` is the right tool when a dataset is *produced* from a CSV: it infers
dimensions, units and metrics, validates them against an instance, and writes the result. But
many datasets in DVC have no generator CSV any more -- they were uploaded from a workbook
before the sources mechanism existed -- and for those a re-upload is the wrong shape of
operation. It would have to reconstruct the CSV first, and every step of that reconstruction
is a chance to write different numbers, a different unit, or (because the DVC path is derived
from the CSV's ``Dataset`` column, not its filename) a different dataset entirely, silently,
beside the live one.

This tool does the one thing that is actually wanted: read the dataset, add
``metadata['sources']``, write it back. The frame is passed through untouched, so the data
cannot change -- and the tool verifies that rather than asserting it, by comparing the frame
it read against the frame it is about to write.

## What counts as a source here

Only registry rows with ``Target: dataset`` (see ``load_sources_registry`` in
``upload_new_dataset``). A ``data_point``-targeted source is placed by the ``Source`` cells of
a generator CSV and therefore belongs to an upload, not to a restamp; naming one here is an
error rather than a no-op, because it would otherwise look as though provenance had been
attached when none was.

A row's ``Datasets`` column may name either the leaf name (as the ``Dataset`` column of a
generator CSV spells it, e.g. ``emissionsfaktoren_endenergie``) or the full DVC identifier
(``de/emissionsfaktoren_endenergie``). Leaf names repeat across namespaces -- six of the BISKO
names exist in both ``de/`` and ``kommune/`` -- so **prefer the full identifier**; a leaf name
that matches more than one requested dataset is reported rather than applied.

## HEAD, not the pin

The dataset is read from, and written to, the **current head** of the DVC repository, never
from a config's ``dataset_repo.commit``. Reading at a pin and pushing the result would
resurrect the data as it was at that pin and quietly revert anything uploaded since. Pass
``--config`` to have the pin compared against head and the difference reported, which is worth
doing: a dataset whose head differs from the pin the models read is one whose stamped copy the
models will not see until the pin is bumped.
"""  # noqa: INP001

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'paths.settings')
django.setup()

import dvc_pandas  # noqa: E402
import ruamel.yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from dvc_pandas import Dataset, DatasetMeta, Repository  # noqa: E402

from nodes.constants import SOURCE_TARGET_DATASET  # noqa: E402
from tools.upload_new_dataset import load_sources_registry  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tools.upload_new_dataset import SourceRegistryEntry

REPO_URL = 'https://github.com/kausaltech/dvctest.git'
DVC_REMOTE = 'kausal-s3'


def open_repo(*, writable: bool) -> Repository:
    """Open the DVC repository at its current head, with push credentials when they are needed."""
    creds = None
    if writable:
        creds = dvc_pandas.RepositoryCredentials(
            git_username=os.getenv('DVC_PANDAS_GIT_USERNAME'),
            git_token=os.getenv('DVC_PANDAS_GIT_TOKEN'),
            git_ssh_public_key_file=os.getenv('DVC_SSH_PUBLIC_KEY_FILE'),
            git_ssh_private_key_file=os.getenv('DVC_SSH_PRIVATE_KEY_FILE'),
        )
    return Repository(repo_url=REPO_URL, dvc_remote=DVC_REMOTE, repo_credentials=creds)


def config_pin(path: str) -> str | None:
    """Return the ``dataset_repo.commit`` a config names, for comparison against head."""
    yaml = ruamel.yaml.YAML(typ='safe')
    with Path(path).open(encoding='utf-8') as fh:
        config = yaml.load(fh)
    repo = config.get('dataset_repo') or {}
    commit = repo.get('commit')
    return str(commit) if commit else None


def dataset_level_sources(registry: dict[str, SourceRegistryEntry], identifier: str) -> list[dict[str, Any]]:
    """
    Return the ``metadata['sources']`` entries that apply to one dataset.

    A registry row applies when it is dataset-targeted and its ``Datasets`` column names this
    dataset -- by full identifier or by leaf name -- or is empty, which means every dataset.
    """
    leaf = identifier.rsplit('/', maxsplit=1)[-1]
    out: list[dict[str, Any]] = []
    for name, entry in sorted(registry.items()):
        if entry.target != SOURCE_TARGET_DATASET:
            continue
        if not (entry.applies_to(identifier) or entry.applies_to(leaf)):
            continue
        out.append({'name': name, **entry.fields})
    return out


def requested_identifiers(
    registry: dict[str, SourceRegistryEntry],
    given: list[str],
    skip: list[str] | None = None,
) -> list[str]:
    """
    Work out which datasets to restamp: the ones named on the command line, or the registry's own.

    With no identifiers given, every dataset a dataset-targeted row names is restamped -- which
    is why those rows should name full identifiers. A row with an empty ``Datasets`` column
    names nothing in particular and cannot drive the run; it is only meaningful alongside an
    explicit list, so it is reported and skipped.

    ``skip`` removes datasets from either list. It exists for the **external DVC placeholders** --
    datasets the model reads straight from DVC and never imports into a database, because they
    carry raw index columns no dimension resolves (`district`, `ags`, `we_from`). Stamping one is
    worse than pointless: `metadata['sources']` is read back by `load_dvc_dataset` into
    `DataSource` rows, so on a dataset that is never imported the stamp reaches nothing, while the
    push rewrites and re-uploads the entire parquet. The two BISKO transport tables are 2.35M and
    1.23M rows.

    Their provenance still belongs in the registry -- the registry is documentation as well as an
    instruction -- so they are skipped here rather than deleted from it.

    To find them: they are the rows with a DB row and **zero data points**. `load_dvc_dataset`
    creates the row and schema before it fails to fill them, so "a row with nothing in it" is
    exactly this case and is what `dataset_inventory` reports as `dvc only`.
    """
    skip_set = set(skip or ())
    if given:
        return [i for i in dict.fromkeys(given) if i not in skip_set]
    named: list[str] = []
    unbounded: list[str] = []
    for name, entry in sorted(registry.items()):
        if entry.target != SOURCE_TARGET_DATASET:
            continue
        if entry.datasets is None:
            unbounded.append(name)
            continue
        named.extend(sorted(entry.datasets))
    for name in unbounded:
        print(f"  ! '{name}' is dataset-targeted but names no datasets; skipped. Name it on the command line.")
    chosen = [i for i in dict.fromkeys(named) if '/' in i]
    dropped = [i for i in chosen if i in skip_set]
    for i in dropped:
        print(f'  - {i}: skipped by --skip')
    return [i for i in chosen if i not in skip_set]


def index_column_names(index_columns: list[Any] | None, columns: Sequence[str]) -> list[str]:
    """
    Reduce what a dataset reports as its index to the column names it actually names.

    pandas metadata describes an index either by column name or by a descriptor dict. A
    descriptor for a stored column carries its ``name``; the one for a frame with no index of
    its own (``{'kind': 'range', 'name': None, ...}``) names nothing, and there is then nothing
    to record -- omitting the key leaves the reader to derive the index from the parquet, which
    is what it did before this tool ever ran.

    Names the frame does not carry are dropped as well: an index column that is not a column of
    the written parquet cannot be set on read.
    """
    if not index_columns:
        return []
    names: list[str] = []
    for entry in index_columns:
        name: Any = entry if isinstance(entry, str) else entry.get('name') if isinstance(entry, dict) else None
        if isinstance(name, str) and name in columns and name not in names:
            names.append(name)
    return names


def restamp(
    repo: Repository,
    identifier: str,
    sources: list[dict[str, Any]],
    *,
    replace: bool,
    apply: bool,
) -> bool:
    """Read one dataset, attach the sources and push it back. Return True when it was (or would be) written."""
    ds = repo.load_dataset(identifier)
    df = ds.df
    if df is None:
        print(f'  {identifier}: has no data frame; skipped')
        return False
    metadata = dict(ds.meta.metadata or {})
    existing = metadata.get('sources')

    print(f'  {identifier}')
    print(f'      rows={df.height} cols={len(df.columns)} units={ds.meta.units}')
    if existing:
        names = ', '.join(str(s.get('name')) for s in existing)
        if not replace:
            print(f'      already stamped ({names}); left alone -- pass --replace to overwrite')
            return False
        print(f'      replacing: {names}')
    if not sources:
        print('      no dataset-level source in the registry applies; skipped')
        return False
    for source in sources:
        print(f'      + {source["name"]}  [{source.get("authority") or "no authority"}]')

    if not apply:
        return True

    metadata['sources'] = sources
    # Record index_columns explicitly, as `upload_new_dataset.push_to_dvc` does. Read back from
    # the parquet's pandas metadata here, it is the same list; written into the .dvc metadata it
    # stops the reading path having to re-derive it, which is what fails on datasets whose Year
    # values would become a virtual RangeIndex.
    #
    # Only column *names* may be written. What is read back can also be a pandas index
    # descriptor -- `{'kind': 'range', 'name': None, ...}` for a frame with no index of its own
    # -- and writing that into the .dvc metadata poisons the dataset for every reader: it names
    # no column, and dvc-pandas >= 0.4.0 rejects the manifest outright. That is what happened to
    # `de/biodieselanteil` on 2026-09-19; see docs/operations/dvc-metadata-index-columns.md.
    index_columns = index_column_names(ds.meta.index_columns, df.columns)
    if index_columns:
        metadata['index_columns'] = index_columns

    # index_columns=None on the meta so `to_parquet` does not call set_index(); every column
    # stays physical, exactly as an upload writes it.
    meta = DatasetMeta(identifier=identifier, index_columns=None, units=ds.meta.units, metadata=metadata)
    out = Dataset(df, meta=meta)
    # The frame is the one that was read, not a rebuilt one -- but say so in a way that would
    # fail if that ever stopped being true, because "the data cannot change" is the whole claim
    # this tool makes.
    if out.df is None or not out.df.equals(df):
        raise AssertionError(f'{identifier}: the frame to be written differs from the frame read')
    repo.push_dataset(out)
    print('      pushed')
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('identifiers', nargs='*', help='DVC identifiers, e.g. de/emissionsfaktoren_endenergie')
    parser.add_argument('--sources-csv', required=True, help='Data source registry CSV')
    parser.add_argument('--config', help='Config whose dataset_repo.commit is compared against head')
    parser.add_argument(
        '--skip',
        action='append',
        default=[],
        metavar='IDENTIFIER',
        help='Leave this dataset alone; repeatable. For external DVC placeholders that no instance imports.',
    )
    parser.add_argument('--replace', action='store_true', help='Overwrite sources on a dataset that already has some')
    parser.add_argument('--apply', action='store_true', help='Push; without it nothing is written')
    args = parser.parse_args()

    load_dotenv()
    registry = load_sources_registry(args.sources_csv)
    dataset_level = {n for n, e in registry.items() if e.target == SOURCE_TARGET_DATASET}
    print(f'{len(registry)} source(s) in the registry, {len(dataset_level)} of them dataset-level')

    identifiers = requested_identifiers(registry, args.identifiers, args.skip)
    if not identifiers:
        print('Nothing to do: no identifiers given and no dataset-level source names any.')
        return 1

    repo = open_repo(writable=args.apply)
    print(f'repository head: {repo.commit_id}')
    if args.config:
        pin = config_pin(args.config)
        if pin and pin != repo.commit_id:
            print(f'  ! {args.config} pins {pin[:12]}, head is {str(repo.commit_id)[:12]}.')
            print('    Stamping happens on head. Bump the pin afterwards or the models will not see it.')

    print(f'\n{len(identifiers)} dataset(s):')
    written = 0
    for identifier in identifiers:
        sources = dataset_level_sources(registry, identifier)
        try:
            if restamp(repo, identifier, sources, replace=args.replace, apply=args.apply):
                written += 1
        except Exception as exc:
            print(f'  {identifier}: FAILED -- {type(exc).__name__}: {exc}')

    verb = 'stamped' if args.apply else 'would be stamped'
    print(f'\n{written} of {len(identifiers)} dataset(s) {verb}.')
    if written and not args.apply:
        print('Re-run with --apply to push.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
