from __future__ import annotations
import base64

from dataclasses import dataclass, field
import logging
import hashlib
import os
import io
import inspect
import json
import typing
from time import perf_counter_ns
from typing import Any, Callable, ClassVar, Dict, List, Literal, Optional, Sequence, Union, Set, overload

import numpy as np
import pandas as pd
import pint_pandas
from rich import print as pprint
import networkx as nx

from common.i18n import I18nString, TranslatedString, get_modeltrans_attrs_from_str
from common.perf import PerfCounter
from common.types import Identifier, MixedCaseIdentifier
from common.utils import hash_unit
from nodes.constants import (
    EMISSION_FACTOR_QUANTITY, EMISSION_QUANTITY, ENERGY_QUANTITY, FORECAST_COLUMN,
    MILEAGE_QUANTITY, VALUE_COLUMN, YEAR_COLUMN,
    ensure_known_quantity, DEFAULT_METRIC, get_quantity_icon
)
from params import Parameter
from params.param import ParameterWithUnit

from .context import Context
from .datasets import Dataset, JSONDataset
from .exceptions import NodeError
from .dimensions import Dimension, DimensionCategory
from .units import Unit, Quantity

if typing.TYPE_CHECKING:
    from .processors import Processor


class NodeMetric:
    id: MixedCaseIdentifier  # FIXME: Convert to Identifier
    column_id: str
    unit: Unit
    quantity: str
    node: Node
    label: I18nString | None
    default_unit: str | Unit

    __slots__ = ('id', 'unit', 'quantity', 'default_unit', 'label', 'column_id', 'node')

    def __init__(
        self, unit: Union[str, Unit], quantity: str, id: str | None = None,
        label: I18nString | None = None, column_id: str | None = None,
    ):
        if id is not None:
            self.id = MixedCaseIdentifier.validate(id)
        if isinstance(unit, Unit):
            self.unit = unit
        self.default_unit = unit
        ensure_known_quantity(quantity)
        self.quantity = quantity
        self.label = label
        if column_id is not None:
            self.column_id = MixedCaseIdentifier.validate(column_id)
        else:
            self.column_id = None  # type: ignore

    def populate_unit(self, context: Context):
        unit = self.default_unit
        if isinstance(unit, Unit):
            self.unit = unit
        else:
            self.unit = context.unit_registry.parse_units(unit)

    def calculate_hash(self) -> bytes:
        s = '%s:%s:%s' % (self.id, self.quantity, self.column_id)
        return s.encode('utf-8') + hash_unit(self.unit)

    def ensure_output_unit(self, s: pd.Series, input_node: Node | None = None):
        if hasattr(s, 'pint'):
            s_u: Unit = s.pint.u
            if self.unit.dimensionality != s_u.dimensionality:
                if input_node is not None:
                    node_str = ' from node %s' % input_node.id
                else:
                    node_str = ''
                raise NodeError(self.node, 'Series with type %s%s is not compatible with %s' % (
                    s_u, node_str, self.unit
                ))
            # Units match exactly
            if s_u == self.unit:
                return s
            s_pt = pint_pandas.PintType(s_u)
            values_type = s.pint.m.dtype
        else:
            s_pt = None
            values_type = s.dtype

        if values_type not in (np.float64, np.float32):
            s = s.astype(np.float64)
            if s_pt is not None:
                s = s.astype(s_pt)

        node_pt = pint_pandas.PintType(self.unit)
        s = s.astype(node_pt)
        return s


@dataclass
class Edge:
    input_node: Node
    output_node: Node
    tags: list[str] = field(default_factory=list)
    dimensions: dict[str, list[DimensionCategory]] = field(default_factory=dict)

    def __post_init__(self):
        self.tags = self.tags.copy()
        self.dimensions = self.dimensions.copy()

    @classmethod
    def from_config(cls, config: dict | str, node: Node, is_output: bool, context: Context) -> Edge:
        if isinstance(config, str):
            other_id = config
        else:
            other_id = config.get('id')
            if other_id is None:
                raise KeyError("node id not given in edge definition")
        other = context.nodes.get(other_id)
        if other is None:
            raise KeyError("node %s not found" % other_id)

        args = {}
        args['output_node'], args['input_node'] = (other, node) if is_output else (node, other)
        if isinstance(config, dict):
            args['tags'] = config.get('tags', [])
            dcs = config.get('dimensions', [])
            dims = {}
            for dc in dcs:
                dim_id = dc.get('id')
                if dim_id not in context.dimensions:
                    raise KeyError("dimension %s not found" % dim_id)
                dim = context.dimensions[dim_id]
                cat_ids = dc['categories']
                cats = [dim.get(cat_id) for cat_id in cat_ids]
                dims[dim.id] = cats
            args['dimensions'] = dims
        return Edge(**args)


class Node:
    id: Identifier
    "Identifier of the Node instance."

    database_id: Optional[int]
    "The database row that corresponds to this Node instance."

    name: I18nString
    "Human-readable label for the Node instance."

    # Description for the Node instance
    #
    # This gets mapped to NodeType.short_description in the GraphQL schema and
    # wrapped in a <p> tag.
    description: I18nString | None

    # if the node has an established visualisation color
    color: Optional[str]
    # order comes from NodeConfig
    order: Optional[int] = None
    # if this node should have its own outcome page
    is_outcome: bool = False

    # output unit (from pint)
    unit: Optional[Unit]
    # default unit for a node class (defined as a class variable)
    default_unit: ClassVar[str]
    # output quantity (like 'energy' or 'emissions')
    quantity: Optional[str]

    # optional tags to differentiate between multiple input/output nodes
    tags: Set[str]

    # output units and quantities (for multi-metric nodes)
    output_metrics: dict[str, NodeMetric] = {}

    output_dimensions: dict[str, Dimension]
    "The dimensions that this node's output will contain."

    output_dimension_ids: list[str] = []
    "References to the dimensions that this node's output will contain (typically set in a class)."

    # set if this node has a specific goal for the simulation target year
    target_year_goal: Optional[float]

    input_datasets: List[str]

    input_dataset_instances: List[Dataset]
    input_dataset_processors: List[Processor]

    edges: List[Edge]

    # Global input parameters the node needs
    global_parameters: List[str]

    # Parameters with their values
    parameters: Dict[str, Parameter]

    # All allowed parameters for this class
    allowed_parameters: ClassVar[list[Parameter]]

    # Output for the node in the baseline scenario
    baseline_values: Optional[pd.DataFrame]

    # When was this node last changed
    modified_at: int | None = None
    last_hash: bytes | None = None
    last_hash_time: int | None = None
    param_hash: bytes | None = None
    mtime_hash: bytes | None = None
    dim_hash: bytes | None = None
    metrics_hash: bytes | None = None

    # Cache last historical year
    _last_historical_year: Optional[int]
    context: Context

    logger: logging.Logger
    debug: bool = False


    def __post_init__(self): ...

    def _init_metrics(self, unit: Unit | None, quantity: str | None):
        self.output_metrics = self.output_metrics.copy()
        if self.output_metrics:
            for met_id, met in self.output_metrics.items():
                met.populate_unit(self.context)
                met.id = MixedCaseIdentifier.validate(met_id)

            if len(self.output_metrics) == 1:
                # Single-metric node
                metric = list(self.output_metrics.values())[0]
                self.unit = metric.unit
                self.quantity = metric.quantity
                if not metric.column_id:
                    metric.column_id = VALUE_COLUMN
            else:
                self.unit = None
                self.quantity = None
                for met_id, met in self.output_metrics.items():
                    if not met.column_id:
                        met.column_id = VALUE_COLUMN
        else:
            quantity = quantity or getattr(self, 'quantity', None)
            if quantity is None:
                raise NodeError(self, "Must provide quantity")
            ensure_known_quantity(quantity)
            self.quantity = quantity
            self.unit = unit
            if unit is None:
                raise NodeError(self, "Attempting to initialize node without a unit")
            # Create the default metric automatically for now
            self.output_metrics[DEFAULT_METRIC] = NodeMetric(
                unit, self.quantity, id=DEFAULT_METRIC, column_id=VALUE_COLUMN
            )

        for m in self.output_metrics.values():
            m.node = self

    def __init__(
        self, id: str, context: Context, name: I18nString, unit: Unit | None = None,
        quantity: str | None = None, description: I18nString | None = None,
        color: str | None = None, order: int | None = None, is_outcome: bool = False,
        target_year_goal: float | None = None, input_datasets: List[Dataset] | None = None,
        output_dimension_ids: list[str] | None = None,
    ):
        self.id = Identifier.validate(id)
        self.context = context
        self._init_metrics(unit, quantity)
        if input_datasets is None:
            input_datasets = []

        self.database_id = None
        self.name = name
        self.description = description
        self.color = color
        self.order = order
        self.is_outcome = is_outcome
        self.target_year_goal = target_year_goal
        self.input_dataset_instances = input_datasets
        self.edges = []
        self.baseline_values = None
        self.parameters = {}
        self.tags = set()
        self.input_dataset_processors = []

        kls = type(self)
        self.logger = logging.getLogger('%s.%s' % (kls.__module__, kls.__name__))

        if not hasattr(self, 'global_parameters'):
            self.global_parameters = []

        self.output_dimensions = {}
        if output_dimension_ids and self.output_dimension_ids:
            if set(output_dimension_ids) != set(self.output_dimension_ids):
                raise NodeError(self, "Invalid dimensions supplied: %s" % ', '.join(output_dimension_ids))
        elif self.output_dimension_ids:
            output_dimension_ids = self.output_dimension_ids

        if output_dimension_ids:
            for dim_id in self.output_dimension_ids:
                dim = self.context.dimensions.get(dim_id)
                if not dim:
                    raise NodeError(self, "Dimension %s not found" % dim_id)
                self.output_dimensions[dim_id] = dim

        # Call the subclass post-init method if it is defined
        if hasattr(self, '__post_init__'):
            self.__post_init__()

    def add_parameter(self, param: Parameter):
        if param.local_id in self.parameters:
            raise Exception(f"Local parameter {param.local_id} already defined for node {self.id}")
        self.parameters[param.local_id] = param
        param.set_node(self)

    def _mark_modified(self):
        self.modified_at = perf_counter_ns()
        for node in self.output_nodes:
            node._mark_modified()

    def notify_parameter_change(self, param: Parameter):
        """
        Notify the node that an input parameter changed.
        """
        self.param_hash = None
        # Propagate change notification to downstream nodes
        self._mark_modified()

    def get_parameters(self):
        for param in self.parameters.values():
            yield param

    @overload
    def get_parameter(self, local_id: str, *, required: Literal[True] = True) -> Parameter: ...

    @overload
    def get_parameter(self, local_id: str, *, required: Literal[False]) -> Parameter | None: ...

    @overload
    def get_parameter(self, local_id: str, *, required: bool) -> Parameter | None: ...

    def get_parameter(self, local_id: str, required: bool = True):
        """Get the parameter with the given local id from this node's parameters."""
        if local_id in self.parameters:
            return self.parameters[local_id]
        if required:
            raise NodeError(self, f"Local parameter {local_id} not found for node {self.id}")
        return None


    @overload
    def get_parameter_value(self, id: str, *, required: Literal[True] = True, units: Literal[True]) -> Quantity: ...

    @overload
    def get_parameter_value(self, id: str, *, required: Literal[False], units: Literal[True]) -> Quantity | None: ...

    @overload
    def get_parameter_value(self, id: str, *, required: Literal[True] = True, units: Literal[False] = False) -> Any: ...

    @overload
    def get_parameter_value(self, id: str, *, required: Literal[False], units: Literal[False] = ...) -> Any | None: ...

    def get_parameter_value(self, id: str, *, required: bool = True, units: bool = False) -> Any:
        param = self.get_parameter(id, required=required)
        if param is None:
            return None
        if units:
            unit = param.get_unit()
            return param.value * unit
        return param.value

    @overload
    def get_global_parameter_value(self, id: str, *, required: Literal[True] = True, units: Literal[True]) -> Quantity: ...

    @overload
    def get_global_parameter_value(self, id: str, *, required: Literal[False], units: Literal[True]) -> Quantity | None: ...

    @overload
    def get_global_parameter_value(self, id: str, *, required: Literal[True] = True, units: Literal[False] = False) -> Any: ...

    @overload
    def get_global_parameter_value(self, id: str, *, required: Literal[False], units: Literal[False] = ...) -> Any | None: ...

    def get_global_parameter_value(self, id: str, *, required: bool = True, units: bool = False) -> Any:
        if id not in self.global_parameters:
            raise NodeError(self, f"Attempting to access global parameter {id} which is not declared")
        if units:
            param = self.context.get_parameter(id, required=required)
            if param is None:
                return None
            if not hasattr(param, 'unit'):
                raise NodeError(self, f"Parameter {id} does not support units")
            assert isinstance(param, ParameterWithUnit)
            assert param.unit is not None
            return param.value * param.unit
        return self.context.get_parameter_value(id, required=required)

    def set_parameter_value(self, local_id: str, value: Any, force: bool = False):
        # if force:  # FIXME if force, create this parameter
        #     self.parameters[local_id] = Parameter()
        if local_id not in self.parameters:
            raise NodeError(self, f"Local parameter {local_id} not found for node {self.id}")
        self.parameters[local_id].set(value)

    def get_input_datasets(self) -> List[Union[pd.DataFrame, pd.Series]]:
        dfs: List[Union[pd.DataFrame, pd.Series]] = []
        for ds in self.input_dataset_instances:
            df = ds.get_copy(self.context)
            if df.index.duplicated().any():
                raise NodeError(self, "Input dataset has duplicate index rows")
            assert isinstance(df, pd.DataFrame)
            dfs.append(df)

        if self.input_dataset_processors:
            for proc in self.input_dataset_processors:
                for idx, df in enumerate(list(dfs)):
                    assert isinstance(df, pd.DataFrame)
                    df = proc.process_input_dataset(df)
                    dfs[idx] = df
        return dfs

    @overload
    def get_input_dataset(self, required: Literal[True] = True) -> pd.DataFrame: ...

    @overload
    def get_input_dataset(self, required: Literal[False]) -> Optional[pd.DataFrame]: ...

    def get_input_dataset(self, required: bool = True) -> pd.DataFrame | None:
        """Gets the first (and only) dataset if it exists."""
        datasets = self.get_input_datasets()
        if not datasets:
            if required:
                raise NodeError(self, 'No input datasets, but node requires one')
            return None
        if len(datasets) != 1:
            raise NodeError(self, 'Expected only 1 input dataset, got %d' % len(datasets))
        df = datasets[0]
        assert isinstance(df, pd.DataFrame)
        return df

    def get_input_node(self, tag: Optional[str] = None, quantity: str | None = None) -> Node:
        matching_nodes = []
        for edge in self.edges:
            if edge.output_node != self:
                continue
            node = edge.input_node
            if tag is not None:
                if tag not in edge.tags and tag not in node.tags:
                    continue
            if quantity is not None:
                if node.quantity != quantity:
                    # FIXME: Multi-metric support
                    continue
            matching_nodes.append(node)
        if len(matching_nodes) != 1:
            tag_str = (' with tag %s' % tag) if tag is not None else ''
            raise NodeError(self, 'Found %d input nodes%s' % (len(matching_nodes), tag_str))
        return matching_nodes[0]

    @property
    def input_nodes(self) -> list[Node]:
        return [edge.input_node for edge in self.edges if edge.output_node == self]

    @property
    def output_nodes(self) -> list[Node]:
        return [edge.output_node for edge in self.edges if edge.input_node == self]

    def get_last_historical_year(self) -> Optional[int]:
        year = getattr(self, '_last_historical_year', None)
        if year is not None:
            return year

        if len(self.input_dataset_instances) != 1:
            return None
        ds = self.input_dataset_instances[0]
        df = ds.get_copy(self.context)
        if FORECAST_COLUMN not in df:
            return None

        year = df[~df[FORECAST_COLUMN]].index.max()
        if np.isnan(year):
            year = None
        else:
            year = int(year)

        self._last_historical_year = year
        return year

    def _get_cached_hash(self) -> bytes | None:
        if self.modified_at is None or self.last_hash is None or self.last_hash_time is None:
            return None
        if self.modified_at <= self.last_hash_time:
            return self.last_hash
        return None

    def calculate_hash(self, cache: dict | None = None) -> bytes:
        cached_hash = self._get_cached_hash()
        if cached_hash is not None:
            return cached_hash

        h = hashlib.md5()
        debug = self.debug
        if cache is None:
            cache = {}
        if debug:
            print('hashing %s' % self.id)

        def hash_part(part: str, val: str | bytes):
            if isinstance(val, str):
                if debug:
                    print('\t%s: "%s"' % (part, val))
                val = val.encode('utf8')
            else:
                if debug:
                    print('\t%s: "%s"' % (part, base64.b64encode(val).decode('ascii')))
            h.update(val)


        hash_part('id', self.id.encode('ascii'))
        if self.metrics_hash is None:
            metrics_hash = bytes()
            for m in self.output_metrics.values():
                metrics_hash += m.calculate_hash()
            self.metrics_hash = metrics_hash
        hash_part('metrics', self.metrics_hash)

        if self.output_dimensions:
            dim_hash = bytes()
            for dim_id, dim in self.output_dimensions.items():
                dim_hash += dim.calculate_hash()
            self.dim_hash = dim_hash
            hash_part('dimensions', self.dim_hash)

        for node in self.input_nodes:
            hash_part('input %s' % node.id, node.calculate_hash(cache=cache))

        if self.param_hash is None:
            param_hash = bytes()
            for param in self.parameters.values():
                param_hash += param.calculate_hash()
                # hash_part('param %s' % param.local_id, param.calculate_hash())
            for param_id in self.global_parameters:
                gp = self.context.get_parameter(param_id, required=False)
                if gp is not None:
                    param_hash += gp.calculate_hash()
                    # hash_part('global param %s' % gp.global_id, gp.calculate_hash())
            self.param_hash = param_hash
        hash_part('params', self.param_hash)

        for ds in self.input_dataset_instances:
            hash_part('dataset %s' % ds.id, ds.calculate_hash(self.context))

        if self.mtime_hash is None:
            mtime_hash = bytes()

            for klass in type(self).mro():
                if klass == object:
                    continue
                if klass in cache:
                    mod_mtime = cache[klass]
                else:
                    fn = getattr(klass, '_paths_fname', None)
                    if fn is None:
                        fn = inspect.getfile(klass)
                        setattr(klass, '_paths_fname', fn)

                    try:
                        mod_mtime = os.path.getmtime(fn)
                    except TypeError:
                        continue
                    cache[klass] = mod_mtime
                mtime_hash += str(mod_mtime).encode('ascii')
            self.mtime_hash = mtime_hash
        hash_part('mtime', self.mtime_hash)

        ret = h.digest()
        self.last_hash = ret
        self.last_hash_time = perf_counter_ns()
        return ret

    def _get_output_for_target(self, df: pd.DataFrame, target_node: Node) -> pd.DataFrame:
        col_name: Optional[str] = None

        for edge in self.edges:
            if edge.output_node == target_node:
                break
        else:
            raise NodeError(self, "No connection to target node %s" % target_node.id)

        # If target node is given, check first if there is a level in the df's
        # multi-index called 'node'.
        if isinstance(df.index, pd.MultiIndex) and 'node' in df.index.names:
            df = df.xs(target_node.id, level='node')  # type: ignore
            assert isinstance(df, pd.DataFrame)
            return df

        # Other possibility is that the node id is one of the columns.
        if target_node.id in df.columns:
            col_name = target_node.id
        elif self.unit is None:
            # multi-metric node
            if target_node.quantity in self.output_metrics:
                col_name = target_node.quantity
            else:
                raise NodeError(self, "Quantity '%s' for node %s not found metrics" % (target_node.quantity, target_node))

        if col_name:
            cols = [col_name]
            if FORECAST_COLUMN in df.columns:
                cols.append(FORECAST_COLUMN)
            df = df[cols]
            df = df.rename(columns={col_name: VALUE_COLUMN})
        return df

    def validate_output(self, df: pd.DataFrame):
        if df.index.duplicated().any():
            raise NodeError(self, "Node output has duplicate index rows")
        if FORECAST_COLUMN in df.columns:
            if df.dtypes[FORECAST_COLUMN] != bool:
                raise NodeError(self, "Forecast column is not a boolean")

        if self.output_dimensions:
            assert isinstance(df.index, pd.MultiIndex)
            names = set(df.index.names)
            dim_ids = set(df.output_dimensions.keys())
            if names != dim_ids:
                raise NodeError(self, "Invalid dimensions")
            for idx, name in enumerate(names):
                pass

    def get_output(self, target_node: Node | None = None, metric: str | None = None) -> pd.DataFrame:
        pc = PerfCounter(
            '%s [%s.%s]' % (self.id, type(self).__module__, type(self).__name__),
            level=PerfCounter.Level.INFO if self.debug else PerfCounter.Level.DEBUG
        )
        self.context.perf_context.node_start(self)

        node_hash = self.calculate_hash().hex()
        out = self.context.cache.get(node_hash)
        pc.display("Cache %s" % ('hit' if out is not None else 'miss'))
        if out is None or self.context.skip_cache:
            try:
                out = self.compute()
                pc.display('Computation done')
            except Exception as e:
                print('Exception when computing node %s' % self.id)
                raise e
            if out is None:
                raise NodeError(self, "Node returned no output")

            if self.context.check_mode:
                self.validate_output(out)

            cache_hit = False
        else:
            cache_hit = True
        self.context.perf_context.record_cache(self, is_hit=cache_hit)

        assert isinstance(out, pd.DataFrame)

        if not cache_hit:
            self.context.cache.set(node_hash, out)

        # If a node has multiple outputs, we can specify only one series
        # to include.
        if metric is not None:
            assert metric in out.columns
            cols = [metric]  # FIXME This assumes that the column name and metric are identical
            if FORECAST_COLUMN in out.columns:
                cols.append(FORECAST_COLUMN)
            out = out[cols]
            out = out.rename(columns={metric: VALUE_COLUMN})
            self.context.perf_context.node_end(self)
            pc.display('Done (with dimensions)')
            return out.copy()

        if target_node is not None:
            out = self._get_output_for_target(out, target_node)
            pc.display('Done (with target node)')
        else:
            pc.display('Done (normal exit)')
        self.context.perf_context.node_end(self)
        return out.copy()

    def print_output(self):
        df = self.get_output()
        if self.baseline_values is not None and VALUE_COLUMN in df.columns:
            df['Baseline'] = self.baseline_values[VALUE_COLUMN]
        self.print_pint_df(df)

    def print_pint_df(self, df: Union[pd.DataFrame, pd.Series]):
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df)

        pint_cols = [col for col in df.columns if hasattr(df[col], 'pint')]
        if not pint_cols:
            pprint(df)
            return

        out = df[pint_cols].pint.dequantify()
        for col in df.columns:
            if col in pint_cols:
                continue
            out[col] = df[col]
        pprint(out)

    def print(self, obj: Any):
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            self.print_pint_df(obj)
            return
        pprint(obj)

    def print_outline(self, df):
        if YEAR_COLUMN in df.index.names:
            pick_rows = df.index.get_level_values(YEAR_COLUMN).isin([2017, 2021, 2022])
            self.print_pint_df(df.iloc[pick_rows])
        elif len(df) > 6:
            self.print_pint_df(df.iloc[[0, 1, 2, -3, -2, -1]])
        else:
            self.print_pint_df(df)

    def get_target_year(self) -> int:
        return self.context.target_year

    def get_end_year(self) -> int:
        return self.context.model_end_year

    def compute(self) -> pd.DataFrame:
        raise Exception('Implement in subclass')

    def is_compatible_unit(self, unit_a: Union[str, Unit], unit_b: Union[str, Unit]):
        if isinstance(unit_a, str):
            unit_a = self.context.unit_registry.parse_units(unit_a)
        if isinstance(unit_b, str):
            unit_b = self.context.unit_registry.parse_units(unit_b)
        if unit_a.dimensionality != unit_b.dimensionality:
            return False
        return True

    def strip_units(self, s: pd.Series) -> pd.Series:
        if not hasattr(s, 'pint'):
            return s
        return s.pint.m

    def convert_to_unit(self, s: pd.Series, unit: Unit) -> pd.Series:
        if not s.pint.units.is_compatible_with(unit):
            raise NodeError(self, 'Series with type %s is not compatible with %s' % (
                s.pint.units, unit
            ))
        return s.astype(pint_pandas.PintType(unit))

    def ensure_output_unit(self, s: pd.Series, input_node: Node | None = None):
        if self.unit is None:
            raise NodeError(self, 'Does currently not work on multi-metric nodes')

        metric = self.output_metrics.get(DEFAULT_METRIC)
        if metric is None:
            assert len(self.output_metrics) == 1
            metric = list(self.output_metrics.values())[0]
        return metric.ensure_output_unit(s, input_node)

    def get_downstream_nodes(self) -> List[Node]:
        # Depth-first traversal
        result = []
        closed = set()
        open = self.output_nodes.copy()
        while open:
            current = open.pop()
            if current not in closed:
                closed.add(current)
                result.append(current)
                open += current.output_nodes
        return result

    def get_upstream_nodes(self, filter: Callable[[Node], bool] | None = None) -> List[Node]:
        result = []
        closed = set()
        open = self.input_nodes.copy()
        while open:
            current = open.pop()
            if current not in closed:
                closed.add(current)
                if filter is None or filter(current):
                    result.append(current)
                open += current.input_nodes
        return result

    def on_scenario_created(self, scenario):
        """Called when a scenario is created with this node among the nodes to be notified."""
        pass

    def get_icon(self) -> Optional[str]:
        from nodes.actions import ActionNode
        if isinstance(self, ActionNode):
            return '⚒'

        icons = []
        for metric in self.output_metrics.values():
            icon = get_quantity_icon(metric.quantity)
            if icon:
                icons.append(icon)
        if not icons:
            return None
        return '+'.join(icons)

    def __str__(self):
        return '%s [%s]' % (self.id, str(type(self)))

    def add_input_node(self, node: Node, tags: list[str] = []):
        if node in self.input_nodes:
            raise Exception(f"Node {node} already added to input nodes for {self.id}")
        self.edges.append(Edge(input_node=node, output_node=self, tags=tags))

    def add_edge(self, edge: Edge):
        if edge.input_node == self:
            if edge.output_node in self.output_nodes:
                raise NodeError(self, f"Node {edge.output_node} already added to output nodes")
        elif edge.output_node == self:
            if edge.input_node in self.input_nodes:
                raise NodeError(self, f"Node {edge.input_node} already added to input nodes")
        else:
            raise NodeError(self, f"Attempting to add edge: {edge.input_node} -> {edge.output_node}")
        self.edges.append(edge)

    def is_connected_to(self, other: Node):
        return nx.has_path(self.context.node_graph, self.id, other.id)

    def generate_baseline_values(self):
        assert self.baseline_values is None
        assert self.context.active_scenario.id == 'baseline'
        self.baseline_values = self.get_output()

    def serialize_input_data(self):
        if len(self.input_dataset_instances) != 1:
            datasets = [x.id for x in self.input_dataset_instances]
            raise NodeError(self, 'Too many input datasets: %s' % ', '.join(datasets))
        ds = self.input_dataset_instances[0]
        df = ds.get_copy(self.context)
        if not isinstance(df, pd.DataFrame) or FORECAST_COLUMN not in df.columns:
            raise Exception('Dataset %s is not suitable for serialization')

        out = JSONDataset.serialize_df(df)
        return out

    def validate_input_data(self, data: dict):
        return self._make_input_dataset(data)

    def _make_input_dataset(self, data: dict):
        sio = io.StringIO(json.dumps(data))
        old = self.serialize_input_data()
        old['data'] = data['data']
        _ = pd.read_json(sio, orient='table')
        return old

    def replace_input_data(self, data: dict):
        if len(self.input_dataset_instances) != 1:
            raise NodeError(
                self, "Can't replace data for node with %d input datasets" % len(self.input_dataset_instances))

        d = self._make_input_dataset(data)
        old_ds = self.input_dataset_instances[0]
        try:
            unit = old_ds.get_unit(self.context)
        except Exception:
            # FIXME: Make this more robust
            unit = self.unit
        self.input_dataset_instances[0] = JSONDataset(
            id=old_ds.id, data=d, unit=unit
        )

        self._last_historical_year = None

    def as_node_config_attributes(self):
        """Return a dict that can be used to set corresponding fields of NodeConfig."""
        attributes: dict[str, str] = {
            'identifier': self.id,
        }

        i18n = {}
        default_lang = self.context.instance.default_language

        def set_from_translated_str(s: str | TranslatedString | None, field_name: str):
            if s is None:
                return

            val, tr = get_modeltrans_attrs_from_str(s, field_name, default_lang)
            i18n.update(tr)

        set_from_translated_str(self.name, 'name')
        # Node's description is called `short_description` in NodeConfig and there is no equivalent for
        # NodeConfig's `long_description` in Node
        set_from_translated_str(self.description, 'short_description')

        if self.color:
            attributes['color'] = self.color

        return attributes
