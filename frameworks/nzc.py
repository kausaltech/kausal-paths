from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from django.db.models import CharField
from django.db.models.expressions import F
from django.db.models.functions import Cast

import polars as pl
from dvc_pandas.repository import Repository as DVCRepository

PLACEHOLDER_DATASET_IDENTIFIER = 'nzc/placeholders'
PLACEHOLDER_YEARLY_DATASET_IDENTIFIER = 'nzc/placeholders_yearly'


@dataclass
class NZCPlaceholderInput:
    population: int
    renewmix: Literal['low', 'high']
    temperature: Literal['low', 'high']


def _calculate_placeholders(
    df: pl.DataFrame,
    data: NZCPlaceholderInput,
) -> pl.DataFrame:

    clookup = {'low-low': 0, 'low-high': 1, 'high-low': 2, 'high-high': 3}
    cluster = '%i' % clookup['%s-%s' % (data.renewmix, data.temperature)]

    df = df.with_columns(
        pl
        .when(pl.col('PerCapita').eq(other=True))
        .then(pl.col(cluster) * data.population)
        .otherwise(pl.col(cluster))
        .alias('Value'),
    )

    return df.select(['Value', 'UUID'])


def _calculate_yearly_placeholders(df: pl.DataFrame, data: NZCPlaceholderInput) -> pl.DataFrame:
    clookup = {'low-low': 0, 'low-high': 1, 'high-low': 2, 'high-high': 3}
    c = str(clookup['%s-%s' % (data.renewmix, data.temperature)])

    def scale(col: str) -> pl.Expr:
        return pl.when(pl.col('PerCapita').eq(other=True)).then(pl.col(col) * data.population).otherwise(pl.col(col))

    return df.select([
        pl.col('UUID'),
        pl.col('Year'),
        scale(f'{c}_ccv').alias('Value'),
        scale(f'{c}_min').alias('LowerBound'),
        scale(f'{c}_max').alias('UpperBound'),
    ])


def get_nzc_yearly_default_values(
    repo: DVCRepository,
    data: NZCPlaceholderInput,
) -> dict[str, dict[int, tuple[float, float | None, float | None]]]:
    """
    Return per-year default values with confidence bounds for one city cluster.

    Returns ``{uuid: {year: (value, lower_bound, upper_bound)}}``.
    """
    ds_id = PLACEHOLDER_YEARLY_DATASET_IDENTIFIER
    if not repo.has_dataset(ds_id):
        return {}
    df = repo.load_dataframe(ds_id)
    df = _calculate_yearly_placeholders(df, data)
    result: dict[str, dict[int, tuple[float, float | None, float | None]]] = {}
    for uuid_str, year, value, lower, upper in df.iter_rows():
        result.setdefault(uuid_str, {})[year] = (value, lower, upper)
    return result


def get_nzc_default_values(repo: DVCRepository, data: NZCPlaceholderInput) -> dict[str, float]:
    ds_id = PLACEHOLDER_DATASET_IDENTIFIER
    if not repo.has_dataset(ds_id):
        raise Exception("Dataset '%s' not found in DVC repository" % ds_id)
    df = repo.load_dataframe(PLACEHOLDER_DATASET_IDENTIFIER)
    df = _calculate_placeholders(df, data)
    return dict(df.select(['UUID', 'Value']).iter_rows())


if __name__ == '__main__':
    from rich import print

    repo = DVCRepository('https://github.com/kausaltech/dvctest.git')
    repo.pull_datasets()
    data = NZCPlaceholderInput(population=150000, renewmix='high', temperature='low')
    defaults = get_nzc_default_values(repo, data)
    try:
        import os

        import django

        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'paths.settings')
        django.setup()
    except Exception as e:
        print(defaults)
        print('Unable to initialize Django: %s' % e)
        exit(0)

    from frameworks.models import MeasureTemplate

    mts = list(
        MeasureTemplate.objects.filter(section__framework__identifier='nzc').values(
            'name',
            uuid_str=Cast('uuid', output_field=CharField()),
            section_name=F('section__name'),
        ),
    )
    df = pl.DataFrame(dict(UUID=list(defaults.keys()), Value=list(defaults.values())))
    mt_df = pl.DataFrame(
        data=mts,
        schema=dict(
            name=pl.String,
            uuid_str=pl.String,
            section_name=pl.String,
        ),
        orient='row',
    )
    df = df.join(mt_df, left_on='UUID', right_on='uuid_str', how='outer').select([
        pl.col('UUID').fill_null(pl.col('uuid_str')),
        pl.col('Value'),
        pl.col('name').alias('Measure'),
        pl.col('section_name').alias('Section'),
    ])
    pl.Config.set_tbl_rows(-1)
    print(df)
