"""Datasets whose values are carried inline: fixed values in the config, or a serialized table."""

import io
import json
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast, override

import polars as pl

from common import polars as ppl
from datasets.runtime.base import Dataset
from nodes.constants import (
    FORECAST_COLUMN,
    VALUE_COLUMN,
    YEAR_COLUMN,
)
from nodes.defs.transform_def import (
    InterpolateOp,
)

if TYPE_CHECKING:
    from pandas import DataFrame as PandasDataFrame

    from nodes.units import Unit


@dataclass
class FixedDataset(Dataset):
    """Dataset from fixed values."""

    # Use `dimensionless` for `unit` if the quantities should be dimensionless.
    unit: Unit
    historical: list[tuple[int, float]] | None
    forecast: list[tuple[int, float]] | None
    use_interpolation: bool = False

    def _fixed_multi_values_to_df(self, data) -> PandasDataFrame:
        series = []
        for d in data:
            vals = d['values']
            s = self.pd.Series(data=[x[1] for x in vals], index=[x[0] for x in vals], name=d['id'])
            series.append(s)
        df = self.pd.concat(series, axis=1)
        df.index.name = YEAR_COLUMN
        df = df.reset_index()
        return df

    def __post_init__(self):
        super().__post_init__()
        import pandas as pd

        self.pd = pd

        if self.use_interpolation and not any(isinstance(op, InterpolateOp) for op in self.transformations):
            self.transformations.append(InterpolateOp())

        if self.historical:
            hdf = pl.DataFrame(self.historical, orient='row', schema=[YEAR_COLUMN, VALUE_COLUMN])
            hdf = hdf.with_columns(pl.lit(value=False).alias(FORECAST_COLUMN))
        else:
            hdf = None
        if self.forecast:
            fdf = pl.DataFrame(self.forecast, orient='row', schema=[YEAR_COLUMN, VALUE_COLUMN])
            fdf = fdf.with_columns(pl.lit(value=True).alias(FORECAST_COLUMN))
        else:
            fdf = None

        if hdf is not None and fdf is not None:
            df = pl.concat([hdf, fdf])
        elif hdf is not None:
            df = hdf
        else:
            assert fdf is not None
            df = fdf

        assert df is not None, 'Both historical and forecast data are None'

        # Ensure value column has right units
        pdf = ppl.to_ppdf(df)
        pdf = pdf.set_unit(VALUE_COLUMN, self.unit)
        pdf = pdf.add_to_index(YEAR_COLUMN)
        pdf = self.apply_transformations(pdf)

        self.df = pdf

    @override
    def load_internal(self) -> ppl.PathsDataFrame:
        df = self.df
        assert df is not None
        if self.context.sample_size > 0:
            df = self._sample(df)
        self.df = df
        return self.df

    def hash_data(self) -> dict[str, Any]:
        assert self.df is not None
        df = self.df.to_pandas()
        return dict(hash=int(self.pd.util.hash_pandas_object(df).sum()), sample_size=self.context.sample_size)

    def get_unit(self) -> Unit:
        assert self.unit is not None
        return self.unit


@dataclass
class JSONDataset(Dataset):
    data: dict[str, Any]
    unit: Unit | None
    df: ppl.PathsDataFrame = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.df = JSONDataset.deserialize_df(self.data)
        meta = self.df.get_meta()
        if len(meta.units) == 1:
            self.unit = next(iter(meta.units.values()))

    @override
    def load_internal(self) -> ppl.PathsDataFrame:
        assert self.df is not None
        return self.post_process(self.df)

    def hash_data(self) -> dict[str, Any]:
        import pandas as pd

        df = self.df.to_pandas()
        return dict(hash=int(pd.util.hash_pandas_object(df).sum()))

    def get_unit(self) -> Unit:
        return cast('Unit', self.unit)

    @classmethod
    def deserialize_df(cls, value: dict[str, Any]) -> ppl.PathsDataFrame:
        import pandas as pd
        from pint_pandas import PintType

        sio = io.StringIO(json.dumps(value))
        df = pd.read_json(sio, orient='table')
        for f in value['schema']['fields']:
            unit = f.get('unit')
            col = f['name']
            if unit is not None:
                pt = PintType(unit)
                df[col] = df[col].astype(float).astype(pt)
        return ppl.from_pandas(df)

    @classmethod
    def serialize_df(cls, pdf: ppl.PathsDataFrame, add_uuids: bool = False) -> dict[str, Any]:
        units = {}
        df = pdf.to_pandas()
        df = df.copy()
        for col in df.columns:
            if hasattr(df[col], 'pint'):
                units[col] = str(df[col].pint.units)
                df[col] = df[col].pint.m

        d = json.loads(df.to_json(orient='table'))
        fields = d['schema']['fields']
        for f in fields:
            if f['name'] in units:
                f['unit'] = units[f['name']]

        if add_uuids:
            for row in d['data']:
                uv = row.get('uuid')
                if not uv:
                    row['uuid'] = str(uuid.uuid4())
            for f in fields:
                if f['name'] == 'uuid':
                    break
            else:
                f = dict(name='uuid', type='string')
                fields.append(f)
            f['format'] = 'uuid'

        return d
