#!/usr/bin/env python3
"""
Fetch one stored parquet out of the DVC bucket by hash, and write it as CSV.

The tool for "what is *actually* in the store", when all you have is a hash -- from a
bucket listing, a `.dvc` file, or an error message -- and not the dataset it belongs to.
It needs no Django, no database and no repository checkout, which is the point: it cannot
be misled by a stale mirror or a pin, because it reads the object and nothing else.

When you *do* know the instance and the dataset identifier, prefer:

    python manage.py export_dataset <instance> <identifier> --from dvc --out DIR

which resolves the identifier against the instance's pinned commit, so there is no hash to
find first, and writes the CSV upload format with its sources registry rather than a flat
dump of the stored columns. See `docs/dataset-round-trip.md`.

Usage:

    python -m tools.fetch_dataset <ekey> <output.csv>

`<ekey>` is an S3 object key, an ETag, or an MD5 hash. Run it as a module rather than as a
script path, per `docs/trailhead/tools.md`.
"""

import argparse
import xml.etree.ElementTree as ET

import polars as pl
import requests

BUCKET_URL = 'https://s3.kausal.tech/datasets/'

# How long to wait for the bucket listing and the parquet. Without a timeout a hung
# connection hangs the tool forever, which is the failure mode `S113` warns about.
TIMEOUT_SECONDS = 60


def bucket_entries(listing: str) -> list[tuple[str, str]]:
    """Return (key, etag) for every object in the bucket listing, skipping malformed entries."""
    # Our own bucket's listing, not user input; the S314 concern does not apply.
    root = ET.fromstring(listing)  # noqa: S314
    entries: list[tuple[str, str]] = []
    for content in root.findall('.//{*}Contents'):
        key_element = content.find('.//{*}Key')
        etag_element = content.find('.//{*}ETag')
        if key_element is None or key_element.text is None:
            continue
        etag = (etag_element.text or '').strip('"') if etag_element is not None else ''
        entries.append((key_element.text, etag))
    return entries


def resolve_key(ekey: str, entries: list[tuple[str, str]]) -> str:
    """
    Work out which object `ekey` names, trying the three things it can be.

    In order: an exact object key; an ETag, which the listing carries; or a bare MD5, which
    the store lays out as `files/md5/<first two>/<the rest>`. The third is a convention
    rather than a lookup, so it is last and is returned unverified -- reading it is what
    tells you whether it was right.

    The previous version tried the MD5 guess *inside* the ETag loop, on the first entry
    whose ETag did not match, so it effectively skipped the ETag search on any bucket whose
    first object was not the one wanted.
    """
    for key, _ in entries:
        if key == ekey:
            print(f'Found direct key match: {key}')
            return key
    for key, etag in entries:
        if etag and ekey in etag:
            print(f'Found matching ETag on key: {key}')
            return key
    guess = f'files/md5/{ekey[:2]}/{ekey[2:]}'
    print(f'No key or ETag matched; trying the conventional MD5 path {guess}')
    return guess


def fetch_and_process_dataset(ekey: str, output_filename: str) -> bool:
    """Fetch the object `ekey` names and write it to `output_filename` as CSV."""
    try:
        response = requests.get(BUCKET_URL, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        entries = bucket_entries(response.content.decode('utf-8', errors='replace'))
    except requests.exceptions.RequestException as exc:
        print(f'Error fetching the bucket listing: {exc}')
        return False
    except ET.ParseError as exc:
        print(f'Error parsing the bucket listing: {exc}')
        return False

    key = resolve_key(ekey, entries)
    try:
        print(f'Reading parquet file from: {BUCKET_URL + key}')
        df = pl.read_parquet(BUCKET_URL + key)
    except Exception as exc:
        print(f'Error reading the parquet file: {exc}')
        return False

    df.write_csv(output_filename)
    print(f'Wrote {df.height} rows x {df.width} columns to {output_filename}')
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='python -m tools.fetch_dataset',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('ekey', help='An S3 object key, an ETag, or an MD5 hash.')
    parser.add_argument('output', help='The filename for the output CSV file.')
    args = parser.parse_args()

    if not fetch_and_process_dataset(args.ekey, args.output):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
