from abc import ABC, abstractmethod
from typing import Any, ClassVar, List, Dict, Literal, overload
import pandas as pd
import polars as pl
import numpy as np
from dataclasses import asdict

from params.param import NumberParameter, Parameter
from common import polars as ppl
from common.i18n import gettext_lazy as _

from .node import Node, NodeError
from .context import Context


class Processor(ABC):
    context: Context
    node: Node
    # Parameters with their values
    parameters: Dict[str, Parameter]
    allowed_parameters: ClassVar[List[Parameter]] = []

    def __init__(self, context: Context, node: Node, params: dict[str, Any] = {}):
        self.context = context
        self.node = node
        self.parameters = {}
        for key, val in params.items():
            for param in self.allowed_parameters:
                if param.local_id == key:
                    break
            else:
                raise NodeError(node, "Unknown parameter: %s" % key)

            fields = asdict(param)
            cloned_param = type(param)(**fields)
            cloned_param.set(val)
            self.parameters[key] = cloned_param

    @overload
    def get_parameter(self, local_id: str, required: Literal[True]) -> Parameter: ...

    @overload
    def get_parameter(self, local_id: str) -> Parameter: ...

    def get_parameter(self, local_id: str, required: bool = True) -> Parameter | None:
        """Get the parameter with the given local id from this node's parameters."""
        if local_id in self.parameters:
            return self.parameters[local_id]
        if required:
            raise NodeError(self.node, f"Processor parameter {local_id} not found")
        return None

    @abstractmethod
    def process_input_dataset(self, df: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        pass


class LinearInterpolation(Processor):  # FIXME Probably the df needs first to be converted from long to wide format.
    def process_input_dataset(self, pldf: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        df = pldf.to_pandas()
        index_name = df.index.name
        df = df.reindex(pd.RangeIndex(df.index.min(), df.index.max() + 1))  # FIXME Not meaningful for multiindex.
        for col_name in df.columns:
            col = df[col_name]
            if isinstance(col.iloc[0], (bool,)):
                df[col_name] = col.fillna(method='bfill')
            elif hasattr(col, 'pint'):
                pt = col.dtype
                df[col_name] = col.pint.m.interpolate().astype(pt)
            else:
                df[col_name] = col.interpolate()
        df.index.name = index_name
        return ppl.from_pandas(df)


class FixedMultiplier(Processor):
    allowed_parameters: ClassVar[List[Parameter]] = [
        NumberParameter(
            local_id='multiplier',
            label=_('Multiplier')
        ),
    ]

    def process_input_dataset(self, pldf: ppl.PathsDataFrame) -> ppl.PathsDataFrame:
        df = pldf.to_pandas()
        for col_name in df.columns:
            col = df[col_name]
            if isinstance(col.iloc[0], (bool, np.bool_)):
                continue
            df[col_name] *= self.get_parameter('multiplier').value
        return ppl.from_pandas(df)

