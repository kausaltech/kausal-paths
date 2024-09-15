from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from django.db.models import CharField
from django.db.models.expressions import F
from django.db.models.functions import Cast

import polars as pl
from dvc_pandas.repository import Repository as DVCRepository

PLACEHOLDER_DATASET_IDENTIFIER = 'nzc/placeholders'


@dataclass
class NZCPlaceholderInput:
    population: int
    renewmix: Literal['low', 'high']
    temperature: Literal['low', 'high']


def _calculate_placeholders(
    df: pl.DataFrame, data: NZCPlaceholderInput,
) -> pl.DataFrame:

    clookup = {'low-low': 0, 'low-high': 1, 'high-low': 2, 'high-high': 3}
    cluster = '%i' % clookup['%s-%s' % (data.renewmix, data.temperature)]

    df = df.with_columns(
        pl.when(pl.col('PerCapita').eq(other=True))
            .then(pl.col(cluster) * data.population)
            .otherwise(pl.col(cluster)).alias('Value'),
    )

    return df.select(['Value', 'UUID'])


def get_nzc_default_values(repo: DVCRepository, data: NZCPlaceholderInput) -> dict[str, float]:
    ds_id = PLACEHOLDER_DATASET_IDENTIFIER
    if not repo.has_dataset(ds_id):
        raise Exception("Dataset '%s' not found in DVC repository" % ds_id)
    df = repo.load_dataframe(PLACEHOLDER_DATASET_IDENTIFIER)
    df = _calculate_placeholders(df, data)
    return {row[0]: row[1] for row in df.select(['UUID', 'Value']).iter_rows()}


if __name__ == '__main__':
    from rich import print

    repo = DVCRepository('https://github.com/kausaltech/dvctest.git')
    repo.pull_datasets()
    data = NZCPlaceholderInput(population=150000, renewmix='high', temperature='low')
    defaults = get_nzc_default_values(repo, data)
    try:
        import os

        import django

        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "paths.settings")
        django.setup()
    except Exception as e:
        print(defaults)
        print("Unable to initialize Django: %s" % e)
        exit(0)

    from frameworks.models import MeasureTemplate
    mts = list(
        MeasureTemplate.objects
            .filter(section__framework__identifier='nzc')
            .values(
                'name',
                uuid_str=Cast('uuid', output_field=CharField()),
                section_name=F('section__name'),
            ),
    )
    df = pl.DataFrame(dict(UUID=list(defaults.keys()), Value=list(defaults.values())))
    mt_df = pl.DataFrame(data=mts, schema=dict(
        name=pl.String, uuid_str=pl.String, section_name=pl.String,
    ), orient='row')
    df = df.join(mt_df, left_on='UUID', right_on='uuid_str', how='outer').select([
        pl.col('UUID').fill_null(pl.col('uuid_str')),
        pl.col('Value'),
        pl.col('name').alias('Measure'),
        pl.col('section_name').alias('Section'),
    ])
    pl.Config.set_tbl_rows(-1)
    print(df)
