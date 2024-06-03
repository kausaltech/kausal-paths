from __future__ import annotations

import base64
from functools import wraps
import hashlib
import inspect
import io
import json
import logging
import os
import typing
from time import perf_counter_ns
from typing import Any, Callable, ClassVar, Dict, List, Literal, Optional, Set, Tuple, Union, overload

import sentry_sdk
import numpy as np
import pandas as pd
import pint_pandas
import polars as pl
from rich import print as pprint
import networkx as nx

from common import polars as ppl
from common.i18n import I18nString, TranslatedString, get_modeltrans_attrs_from_str
from common.types import Identifier, MixedCaseIdentifier, validate_identifier
from common.utils import hash_unit
from nodes.constants import (
    DEFAULT_METRIC,
    FORECAST_COLUMN,
    NODE_COLUMN,
    VALUE_COLUMN,
    YEAR_COLUMN,
    ensure_known_quantity,
    get_quantity_icon,
)
from nodes.goals import NodeGoals
from nodes.calc import extend_last_historical_value_pl
from params import Parameter
from params.param import ParameterWithUnit

from .context import Context
from .datasets import Dataset, JSONDataset
from .dimensions import Dimension
from .edges import Edge
from .exceptions import NodeComputationError, NodeError, NodeHashingError
from .units import Quantity, Unit, unit_registry

if typing.TYPE_CHECKING:
    from .processors import Processor
    from .scenario import Scenario
    from .models import NodeConfig
    from common.cache import CacheResult


class_fname_cache: dict[type, str] = {}


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
            self.id = validate_identifier(id, mixed=True)
        if isinstance(unit, Unit):
            self.unit = unit
        self.default_unit = unit
        ensure_known_quantity(quantity)
        self.quantity = quantity
        self.label = label
        if column_id is not None:
            self.column_id = validate_identifier(column_id, mixed=True)
        else:
            self.column_id = None  # type: ignore

    @classmethod
    def from_config(cls, config: dict):
        return cls(
            unit=config['unit'], quantity=config['quantity'], id=config['id'],
            label=None, column_id=None
        )

    def copy(self) -> NodeMetric:
        return NodeMetric(unit=self.unit, quantity=self.quantity, id=self.id, label=self.label, column_id=self.column_id)

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


class Node:
    id: Identifier
    "Identifier of the Node instance."

    database_id: Optional[int]
    "The database row that corresponds to this Node instance."

    db_obj: NodeConfig | None
    "The Django object for this Node Instance"

    name: I18nString
    "Human-readable label for the Node instance."

    short_name: I18nString | None
    "Shortened label for the node"

    # Description for the Node instance
    #
    # This gets mapped to NodeType.short_description in the GraphQL schema and
    # wrapped in a <p> tag.
    description: I18nString | None
    "Long description for the node"

    # if the node has an established visualisation color
    color: Optional[str]
    # order comes from NodeConfig
    order: Optional[int] = None
    is_visible: bool = True
    # if this node should have its own outcome page
    is_outcome: bool = False

    # output unit (from pint)
    unit: Optional[Unit]
    # default unit for a node class (defined as a class variable)
    default_unit: ClassVar[str]
    # output quantity (like 'energy' or 'emissions')
    quantity: Optional[str]
    # minimum year for node -- all output before this year is filtered out
    minimum_year: Optional[int]

    # optional tags to differentiate between multiple input/output nodes
    tags: Set[str]

    # output units and quantities (for multi-metric nodes)
    output_metrics: dict[str, NodeMetric] = {}

    output_dimensions: dict[str, Dimension]
    "The dimensions that this node's output will contain."

    output_dimension_ids: list[str] = []
    "References to the dimensions that this node's output will contain (typically set in a class)."

    input_dimensions: dict[str, Dimension]
    "The dimensions that this node's input must contain."

    input_dimension_ids: list[str] = []
    "References to the dimensions that this node's input must contain (typically set in a class)."

    # set if this node has a specific goal for the simulation target year
    goals: NodeGoals | None

    input_datasets: List[str]

    input_dataset_instances: List[Dataset]
    input_dataset_processors: List[Processor]

    edges: List[Edge]

    # Global input parameters the node needs
    global_parameters: List[str] = []

    # Parameters with their values
    parameters: Dict[str, Parameter]

    # All allowed parameters for this class
    allowed_parameters: ClassVar[list[Parameter]]

    # Output for the node in the baseline scenario
    baseline_values: Optional[ppl.PathsDataFrame]

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
    disable_cache: bool = False
    yaml_fn: str | None
    """YAML filename"""
    yaml_lc: Tuple[int, int] | None
    """YAML line and column information"""

    def __post_init__(self): ...

    def finalize_init(self):
        """Customization and validation that is run after the node graph is fully configured."""
        pass

    def _init_metrics(
        self, unit: Unit | None, quantity: str | None, output_metrics: dict[str, NodeMetric] | None = None
    ):
        if output_metrics is not None:
            self.output_metrics = output_metrics.copy()
        else:
            self.output_metrics = self.output_metrics.copy()
        if self.output_metrics:
            for met_id, met in self.output_metrics.items():
                met.populate_unit(self.context)
                met.id = validate_identifier(met_id, mixed=True)

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
                        met.column_id = met_id
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

        metric_cols = set()
        for metric in self.output_metrics.values():
            if metric.column_id in metric_cols:
                raise NodeError(self, "Duplicate metric column IDs: %s" % metric.column_id)
            metric_cols.add(metric.column_id)

        for m in self.output_metrics.values():
            m.node = self

    def _init_dimensions(
        self,
        class_dims: dict[str, Dimension],
        arg_dims: list[str] | None,
        class_dim_ids: list[str]
    ) -> dict[str, Dimension]:
        dims = class_dims.copy()

        for dim_id, dim in dims.items():
            if not dim.is_internal:
                raise NodeError(self, "Dimensions defined in class can only be internal ones")
            if dim_id in self.context.dimensions:
                raise NodeError(self, "Internal dimension is also a global one")

        if arg_dims and class_dim_ids:
            if set(arg_dims) != set(class_dim_ids):
                raise NodeError(
                    self,
                    "Invalid dimensions supplied: %s; expecting: %s" %
                     (', '.join(arg_dims), ', '.join(class_dim_ids)))
        elif class_dim_ids:
            arg_dims = class_dim_ids

        if not arg_dims:
            return dims

        for dim_id in arg_dims:
            d = self.context.dimensions.get(dim_id)
            if not d:
                raise NodeError(self, "Dimension %s not found" % dim_id)
            dims[dim_id] = d
        return dims

    def __init__(
        self, id: str, context: Context, name: I18nString, short_name: I18nString | None = None,
        unit: Unit | None = None, quantity: str | None = None, minimum_year: int | None = None,
        description: I18nString | None = None, color: str | None = None, order: int | None = None,
        is_visible: bool = True, is_outcome: bool = False, target_year_goal: float | None = None, goals: dict | None = None,
        input_datasets: List[Dataset] | None = None,
        output_dimension_ids: list[str] | None = None, input_dimension_ids: list[str] | None = None,
        output_metrics: dict[str, NodeMetric] | None = None,
        yaml_fn: str | None = None, yaml_lc: Tuple[int, int] | None = None,
    ):
        self.id = validate_identifier(id)
        self.context = context

        self._init_metrics(unit, quantity, output_metrics)
        if input_datasets is None:
            input_datasets = []

        self.database_id = None
        self.db_obj = None
        self.name = name
        self.yaml_fn = yaml_fn
        self.yaml_lc = yaml_lc
        if self.name is None:
            raise NodeError(self, "Node has no name")
        self.short_name = short_name
        self.description = description
        self.color = color
        self.order = order
        self.is_visible = is_visible
        self.is_outcome = is_outcome
        self.minimum_year = minimum_year
        if goals is not None:
            self.goals = NodeGoals.model_validate(goals)
        else:
            if target_year_goal is not None:
                is_main_goal = self.is_outcome
                self.goals = NodeGoals.model_validate(
                    [dict(values=[dict(year=context.target_year, value=target_year_goal)], is_main_goal=is_main_goal)]
                )
            else:
                self.goals = None
        if self.goals is not None:
            self.goals.set_node(self)

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
        else:
            # Copy the parameters so that the list can be mutated later
            self.global_parameters = list(self.global_parameters)

        if not hasattr(self, 'output_dimensions'):
            self.output_dimensions = {}
        self.output_dimensions = self._init_dimensions(
            self.output_dimensions, output_dimension_ids, self.output_dimension_ids
        )

        if not hasattr(self, 'input_dimensions'):
            self.input_dimensions = {}
        self.input_dimensions = self._init_dimensions(
            self.input_dimensions, input_dimension_ids, self.input_dimension_ids
        )

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
            if not required:
                return None
            raise NodeError(self, f"Attempting to access global parameter {id} which is not declared in the node definition")
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

    def get_input_datasets_pl(self, tag: str | None = None, exclude_tags: list[str] | None = None) -> List[ppl.PathsDataFrame]:
        dfs: List[ppl.PathsDataFrame] = []
        for ds in self.input_dataset_instances:
            if tag is not None:
                if tag not in ds.tags:
                    continue
            if exclude_tags is not None:
                exclude = False
                for etag in exclude_tags:
                    if etag in ds.tags:
                        exclude = True
                        break
                if exclude:
                    continue
            df = ds.get_copy(self.context)
            if df.paths.index_has_duplicates():
                raise NodeError(self, "Input dataset has duplicate index rows")
            assert isinstance(df, ppl.PathsDataFrame)
            dfs.append(df)

        if self.input_dataset_processors:
            for proc in self.input_dataset_processors:
                for idx, df in enumerate(list(dfs)):
                    df = proc.process_input_dataset(df)
                    dfs[idx] = df
        return dfs

    def get_input_datasets(self) -> List[pd.DataFrame]:
        dfs = self.get_input_datasets_pl()
        return [df.to_pandas() for df in dfs]

    @overload
    def get_input_dataset_pl(self, tag: str | None = None, *, required: Literal[False]) -> Optional[ppl.PathsDataFrame]: ...

    @overload
    def get_input_dataset_pl(self, tag: str | None = None, required: Literal[True] = True) -> ppl.PathsDataFrame: ...

    @overload
    def get_input_dataset_pl(self, tag: str | None = None, required: bool = ...) -> ppl.PathsDataFrame | None: ...

    def get_input_dataset_pl(self, tag: str | None = None, required: bool = True) -> ppl.PathsDataFrame | None:
        """Gets the first (and only) dataset if it exists."""
        datasets = self.get_input_datasets_pl(tag=tag)
        if not datasets:
            if required:
                raise NodeError(self, 'No input datasets, but node requires one')
            return None
        if len(datasets) != 1:
            raise NodeError(self, 'Expected only 1 input dataset, got %d' % len(datasets))
        df = datasets[0]
        assert isinstance(df, ppl.PathsDataFrame)
        return df


    @overload
    def get_input_dataset(self, required: Literal[True] = True) -> pd.DataFrame: ...

    @overload
    def get_input_dataset(self, required: Literal[False]) -> Optional[pd.DataFrame]: ...

    def get_input_dataset(self, required: bool = True) -> pd.DataFrame | None:
        df = self.get_input_dataset_pl(required=required)
        if df is None:
            return None
        return df.to_pandas()

    def get_input_nodes(self, tag: Optional[str] = None, quantity: str | None = None) -> list[Node]:
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
        return matching_nodes

    @overload
    def get_input_node(self, tag: Optional[str] = None, quantity: str | None = None, *, required: Literal[False]) -> Node | None: ...

    @overload
    def get_input_node(self, tag: Optional[str] = None, quantity: str | None = None, required: Literal[True] = True) -> Node: ...

    def get_input_node(self, tag: Optional[str] = None, quantity: str | None = None, required: bool = True) -> Node | None:
        matching_nodes = self.get_input_nodes(tag=tag, quantity=quantity)
        if len(matching_nodes) == 0 and not required:
            return None
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

        year = df.filter(~df[FORECAST_COLUMN])[YEAR_COLUMN].max()
        if year is None:
            return None
        year = int(year)  # type: ignore

        self._last_historical_year = year
        return year

    def get_default_output_metric(self) -> NodeMetric:
        if DEFAULT_METRIC in self.output_metrics:
            return self.output_metrics[DEFAULT_METRIC]
        if len(self.output_metrics) > 1:
            raise NodeError(self, "Node outputs multiple output metrics, but none of them is set as default")
        return list(self.output_metrics.values())[0]

    def _get_cached_hash(self) -> bytes | None:
        if self.modified_at is None or self.last_hash is None or self.last_hash_time is None:
            return None
        if self.modified_at <= self.last_hash_time:
            return self.last_hash
        return None

    @staticmethod
    def _wrap_hashing_error(func):
        @wraps(func)
        def report_error(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                node = args[0]
                if isinstance(e, NodeHashingError):
                    e.add_node(node)
                    raise
                raise NodeHashingError(args[0], "Unable to hash node") from e
        return report_error

    @_wrap_hashing_error
    def calculate_hash(self, cache: dict | None = None) -> bytes:
        cached_hash = self._get_cached_hash()
        if cached_hash is not None:
            return cached_hash

        h = hashlib.md5()
        if cache is None:
            cache = {}

        cache_parts = []

        def hash_part(typ: str, part: str, val: str | bytes):
            try:
                if isinstance(val, str):
                    cache_parts.append((typ, part, val))
                    val = val.encode('ascii', 'backslashreplace')
                else:
                    cache_parts.append((typ, part, base64.b64encode(val).decode('ascii')))
            except Exception:
                self.logger.error("Unable to hash node: %s %s (value %s)" % (typ, part, val))
                raise
            h.update(val)

        hash_part('id', '', self.id)
        if self.metrics_hash is None:
            metrics_hash = bytes()
            for m in self.output_metrics.values():
                metrics_hash += m.calculate_hash()
            self.metrics_hash = metrics_hash
        hash_part('metrics', '', self.metrics_hash)

        if self.output_dimensions:
            dim_hash = bytes()
            for dim_id, dim in self.output_dimensions.items():
                dim_hash += dim.calculate_hash()
            self.dim_hash = dim_hash
            hash_part('dimensions', '', self.dim_hash)

        for node in self.input_nodes:
            hash_part('input node', node.id, node.calculate_hash(cache=cache))

        if True or self.param_hash is None:
            param_hash = ''
            for _, param in sorted(self.parameters.items(), key=lambda x: x[0]):
                ph = param.calculate_hash()
                param_hash += ph
                hash_part('param', param.local_id, ph)
            for param_id in self.global_parameters:
                gp = self.context.get_parameter(param_id, required=False)
                if gp is not None:
                    ph = gp.calculate_hash()
                    param_hash += ph
                    hash_part('global param', gp.global_id, ph)
            self.param_hash = hashlib.md5(param_hash.encode('ascii')).digest()

        for ds in self.input_dataset_instances:
            hash_part('dataset', ds.id, ds.calculate_hash(self.context))

        if self.mtime_hash is None:
            mtime_hash = bytes()

            for klass in type(self).mro():
                if klass == object:
                    continue
                if klass in cache:
                    mod_mtime = cache[klass]
                else:
                    fn = class_fname_cache.get(klass)
                    if fn is None:
                        fn = inspect.getfile(klass)
                        class_fname_cache[klass] = fn

                    try:
                        mod_mtime = os.path.getmtime(fn)
                    except TypeError:
                        continue
                    cache[klass] = mod_mtime
                mtime_hash += str(mod_mtime).encode('ascii')
            self.mtime_hash = mtime_hash
        hash_part('mtime', '', self.mtime_hash)

        ret = h.digest()
        self.last_hash = ret
        self.last_hash_time = perf_counter_ns()
        self.prev_hash_parts = getattr(self, 'last_hash_parts', None)
        self.last_hash_parts = cache_parts
        return ret

    def _get_output_for_node(self, df: ppl.PathsDataFrame, edge: Edge) -> ppl.PathsDataFrame:
        '''
        Filter and select the dataframe in the following ways:
        * Choose rows that are specific to the target node.
        * Choose the column that is defined in edge.metrics.
        * Choose the only metric column.
        * Remove rows where there are only nulls.
        * Choose the column that has the name of the target node.
        * Choose the column that has the name of the target node quantity.
        '''
        target_node = edge.output_node
        # If target node is given, check first if there is a level in the df's
        # multi-index called 'node'.
        if NODE_COLUMN in df.columns:
            return df.filter(pl.col(NODE_COLUMN) == target_node.id).drop(NODE_COLUMN)

        if edge.metrics:
            drop_cols = list(df.metric_cols)
            remain_cols = []
            for m_id in edge.metrics:
                m = self.output_metrics.get(m_id)
                if m is None:
                    # FIXME: Remove this fallback
                    for m in self.output_metrics.values():
                        if m.column_id == m_id:
                            break
                    else:
                        raise NodeError(self, "Metric '%s' defined at the edge but not present in output_metrics" % m_id)
                if m.column_id not in drop_cols:
                    raise NodeError(self, "Metric column '%s' defined at the edge but not present in DF" % m)
                drop_cols.remove(m.column_id)
                remain_cols.append(m.column_id)
            if drop_cols:
                df = df.drop(drop_cols)
            # FIXME: Should we look at target node's input metrics? Maybe define the
            # edge's metric selection as a mapping instead of a flat list?
            if len(edge.metrics) == 1:
                if edge.metrics[0] != VALUE_COLUMN:
                    df = df.rename({edge.metrics[0]: VALUE_COLUMN})
                    remain_cols = [VALUE_COLUMN]
            # Drop rows where all metric cols are null
            df = df.filter(~pl.all_horizontal(pl.col(col).is_null() for col in remain_cols))
            return df

        col_name: Optional[str] = None

        # Other possibility is that the node id is one of the columns.
        if target_node.id in df.columns:
            col_name = target_node.id
        elif self.unit is None:
            # multi-metric node
            if target_node.quantity in self.output_metrics:
                col_name = target_node.quantity
            # FIXME Add functionality that the target node is a simple one-metric node (to be further adjusted by to_dimensions)
#            elif len(df.get_meta().metric_cols) == 1:
#                print('pitäisi toimia', self.id)
#                return df
            else:
                raise NodeError(self, "Quantity '%s' for node %s not found metrics" % (target_node.quantity, target_node))

        if col_name:
            cols = [YEAR_COLUMN, col_name]
            if FORECAST_COLUMN in df.columns:
                cols.append(FORECAST_COLUMN)
            df = df.select(cols)
            df = df.rename({col_name: VALUE_COLUMN})

        return df

    def _get_output_for_target(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        for edge in self.edges:
            if edge.output_node == target_node:
                break
        else:
            raise NodeError(self, "No connection to target node %s" % target_node.id)

        df = self._get_output_for_node(df, edge)
        if edge.from_dimensions:
            meta = df.get_meta()
            if not meta.dim_ids:
                raise NodeError(self, "No dimensions in node output")
            for dim_id, edge_dim in edge.from_dimensions.items():
                cat_ids = [cat.id for cat in edge_dim.categories]
                if dim_id not in meta.dim_ids:
                    raise NodeError(self, "Dimension %s not in output df" % dim_id)
                filter_expr = pl.col(dim_id).is_in(cat_ids)
                if edge_dim.exclude:
                    filter_expr = ~filter_expr
                df = df.filter(filter_expr)
                if len(df) == 0:
                    raise NodeError(self, "No rows left after filtering by %s" % dim_id)
                if edge_dim.flatten:
                    meta = df.get_meta()
                    df = df.paths.sum_over_dims([dim_id])

        if edge.to_dimensions is not None:
            new_cols = []
            meta = df.get_meta()
            output_dimensions = set(edge.to_dimensions.keys())
            for dim_id, edge_dim in edge.to_dimensions.items():
                nr_cats = len(edge_dim.categories)
                if not nr_cats:
                    continue
                if nr_cats > 1:
                    raise NodeError(self, "to_dimensions can have only one category for now")
                if dim_id in df.columns:
                    raise NodeError(self, "attempting to assign a category to an existing dimension")
                cat = edge_dim.categories[0]
                new_cols.append((dim_id, cat.id))

            if new_cols:
                exprs = [pl.lit(cat).alias(dim_id) for dim_id, cat in new_cols]
                df = df.with_columns(exprs)
                for dim_id, _ in new_cols:
                    meta.primary_keys.append(dim_id)
                df = ppl.to_ppdf(df=df, meta=meta)

            if set(df.dim_ids) != output_dimensions:
                raise NodeError(
                    self, "Dimensions (%s) do not match in output for %s (expecting %s)" % (
                        set(df.dim_ids), target_node, output_dimensions
                    )
                )
        else:
            output_dimensions = set(target_node.input_dimensions.keys())

        # Validate output
        meta = df.get_meta()
        # Drop dim columns that only have nulls
        for dim_col in meta.dim_ids:
            if df[dim_col].null_count() == df[dim_col].len():
                df = df.drop(dim_col)
        meta = df.get_meta()
        if set(meta.dim_ids) != output_dimensions:
            print(df)
            out_dims = ', '.join(meta.dim_ids)
            target_in_dims = ', '.join(output_dimensions)
            raise NodeError(
                self, "Dimensions of output [%s] do not match the input dimensions of %s [%s]" % (
                    out_dims, target_node.id, target_in_dims
                )
            )

        return df

    def validate_output(self, df: ppl.PathsDataFrame):
        meta = df.get_meta()

        if YEAR_COLUMN not in meta.primary_keys:
            raise NodeError(self, "'%s' column missing" % YEAR_COLUMN)

        if df.schema[YEAR_COLUMN] not in pl.INTEGER_DTYPES:
            raise NodeError(self, "Invalid dtype for 'Year': %s" % df.schema[YEAR_COLUMN])

        ldf = df.lazy()
        dupe_rows = (
            ldf.groupby(meta.primary_keys)
            .agg(pl.count())
            .filter(pl.col('count') > 1)
            .sort(YEAR_COLUMN)
        )
        has_dupes = bool(len(dupe_rows.limit(1).collect()))
        if has_dupes:
            print(dupe_rows.collect())
            print(df)
            raise NodeError(self, "Node output has duplicate index rows")

        if FORECAST_COLUMN in df.columns:
            if df[FORECAST_COLUMN].dtype != pl.Boolean:
                raise NodeError(self, "Forecast column is not a boolean")
        else:
            raise NodeError(self, "Forecast column missing")

        for metric in self.output_metrics.values():
            if metric.column_id not in df.columns:
                raise NodeError(self, "Output does not have a column '%s'" % metric.column_id)
            if metric.column_id not in df.columns:
                continue
            if not df.has_unit(metric.column_id):
                raise NodeError(self, "Output column '%s' does not have units" % metric.column_id)
            unit = df.get_unit(metric.column_id)
            if unit != metric.unit:
                raise NodeError(self, "Expecting unit '%s' in column '%s'; got '%s'" % (metric.unit, metric.column_id, unit))

            continue
            # all_hsy_emissions still has nulls
            col = df[metric.column_id]
            if (col.is_nan() | col.is_null()).sum() > 0:
                self.print(df)
                raise NodeError(self, "Output column '%s' has NaN or null values" % metric.column_id)

        if NODE_COLUMN in meta.primary_keys:
            # FIXME
            return

        for dim_id, dim in self.output_dimensions.items():
            if dim_id not in meta.primary_keys:
                raise NodeError(self, "Dimension column '%s' not included in index" % dim_id)
            dt = df[dim_id].dtype
            if dt not in (pl.Utf8, pl.Categorical):
                raise NodeError(self, "Dimension column '%s' is of wrong type (%s)" % (dim_id, dt))

            if dim.is_internal:
                # Skip validation for internal dimensions
                continue

            cats = set(df[dim_id].unique())
            dim_cats = dim.get_cat_ids()
            diff = cats - dim_cats
            if diff:
                raise NodeError(self, "Unknown categories in dimension column '%s': %s" % (dim_id, ', '.join(diff)))

        dim_ids = set(meta.dim_ids)
        node_dims = set(self.output_dimensions.keys())
        if dim_ids != node_dims and not getattr(self, 'allow_unknown_dimensions', None):
            raise NodeError(self, "Output has unknown dimensions: %s (expecting %s)" % (', '.join(dim_ids), ', '.join(node_dims)))

    def get_output(self, target_node: Node | None = None, metric: str | None = None) -> pd.DataFrame:
        df = self.get_output_pl(target_node, metric)
        return df.to_pandas()

    def get_output_pl(self, target_node: Node | None = None, metric: str | None = None) -> ppl.PathsDataFrame:
        perf_cm = self.context.perf_context
        with sentry_sdk.start_span(op='node', description=self.id), perf_cm.exec_node(self) as node_run:
            try:
                res, cache_res = self._get_output_pl(target_node=target_node, metric=metric)
            except Exception as e:
                if isinstance(e, NodeComputationError):
                    e.add_node(self)
                    raise
                raise NodeComputationError(self, "Error getting output") from e
            if node_run is not None and cache_res is not None:
                node_run.mark_cache(cache_res)

        if target_node is not None:
            for edge in self.edges:
                if edge.output_node == target_node:
                    break
            else:
                raise NodeError(self, "No connection to target node %s" % target_node.id)
            if 'extend_values' in edge.tags:
                res = extend_last_historical_value_pl(res, self.get_end_year())
            if 'arithmetic_inverse' in edge.tags:
                res = res.multiply_quantity(VALUE_COLUMN, unit_registry('-1 * dimensionless'))
            if 'geometric_inverse' in edge.tags:
                res = res.divide_quantity(VALUE_COLUMN, unit_registry('1 * dimensionless'))
            if 'complement' in edge.tags:
                if not res.get_unit(VALUE_COLUMN).is_compatible_with('dimensionless'):
                    raise NodeError(self, 'The unit of node %s must be compatible with dimensionless for taking complement' % self.id)
                if not self.quantity in ['fraction', 'probability']:
                    raise NodeError(self, 'The quantity of node %s must be fraction or probability for taking complement' % self.id)
                res = res.ensure_unit(VALUE_COLUMN, unit='dimensionless')  # TODO CHECK
                res = res.with_columns((pl.lit(1.0) - pl.col(VALUE_COLUMN)).alias(VALUE_COLUMN))
            if 'difference' in edge.tags:
                res = res.diff(VALUE_COLUMN)
            if 'cumulative' in edge.tags:
                res = res.cumulate(VALUE_COLUMN)

        return res

    def _get_output_pl(
        self, target_node: Node | None = None, metric: str | None = None
    ) -> Tuple[ppl.PathsDataFrame, CacheResult[ppl.PathsDataFrame] | None]:
        use_cache = not (self.disable_cache or self.context.skip_cache)
        cache_res = None
        out = None
        if use_cache:
            node_hash = '%s:%s' % (self.id, self.calculate_hash().hex())
            cache_res = self.context.cache.get(node_hash)
            out = cache_res.obj
        else:
            node_hash = ''

        if False and out is None and self.prev_hash_parts:
            print(self.id)
            if len(self.prev_hash_parts) != len(self.last_hash_parts):
                pprint('Length mismatch!!')
                pprint(self.prev_hash_parts)
                pprint(self.last_hash_parts)
                pprint('\n\n')
            for old, new in zip(self.prev_hash_parts, self.last_hash_parts):
                if old[0] == 'input node' and new[0] == 'input node' and old[1] == new[1]:
                    continue
                if old != new:
                    pprint('\told: %s\n\tnew: %s' % (old, new))

        if out is None:
            try:
                out = self.compute()
            except Exception as e:
                self.context.log.exception('Exception when computing node %s' % str(self))
                raise e
            if out is None:
                raise NodeError(self, "Node returned no output")

            if isinstance(out, pd.DataFrame):
                out = ppl.from_pandas(out)

            self.validate_output(out)

        assert isinstance(out, ppl.PathsDataFrame)

        if cache_res and not cache_res.is_hit:
            self.context.cache.set(node_hash, out.copy())

        meta = out.get_meta()

        if self.minimum_year is not None:
            out = out.filter(pl.col(YEAR_COLUMN) >= self.minimum_year)

        # If a node has multiple outputs, we can specify only one series
        # to include.
        if metric is not None:
            assert metric in out.columns  # If target_node has metric, self MUST have that also.
            cols = meta.primary_keys.copy()
            cols.append(metric)
            if FORECAST_COLUMN in out.columns:
                cols.append(FORECAST_COLUMN)
            out = out.select(cols)
            out = out.rename({metric: VALUE_COLUMN})
            return (out, cache_res)  # FIXME I'd like to comment this out because if metric, to_dimensions is ignored.

        if target_node is not None:
            out = self._get_output_for_target(out, target_node)

        return (out, cache_res)

    def print_output(self, only_years: list[int] | None = None, filters: list[str] | None = None):
        from .debug import print_node_output
        print_node_output(self, only_years=only_years, filters=filters)

    def plot_output(self, filters: list[str] | None = None):
        from .debug import plot_node_output
        plot_node_output(self, filters=filters)

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
        #if isinstance(obj, ppl.PathsDataFrame):
        #    obj.print()
        #    return
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

    def compute(self) -> pd.DataFrame | ppl.PathsDataFrame:
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

    def get_downstream_nodes(self, to_node: Node | None = None, max_depth: int | None = None) -> List[Node]:
        res = nx.bfs_successors(self.context.node_graph, self.id, depth_limit=max_depth)
        nodes = []
        for _, children in res:
            for node_id in children:
                nodes.append(self.context.get_node(node_id))
        return nodes

    def get_upstream_nodes(self, filter: Callable[[Node], bool] | None = None) -> List[Node]:
        from nodes.actions.parent import ParentActionNode
        from nodes.actions import ActionNode

        result = []
        closed = set()
        open = self.input_nodes.copy()
        while open:
            current = open.pop()
            if current in closed:
                continue
            closed.add(current)
            if filter is None or filter(current):
                result.append(current)
            if isinstance(current, ActionNode) and current.parent_action is not None:
                open.append(current.parent_action)
            if isinstance(current, ParentActionNode):
                open += current.subactions
            open += current.input_nodes
        return result

    def on_scenario_created(self, scenario: Scenario):
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
        cls = type(self)
        return '%s <%s.%s>' % (self.id, cls.__module__, cls.__name__)

    def __repr__(self):
        return self.__str__()

    def add_input_node(self, node: Node, tags: list[str] = []):
        if node in self.input_nodes:
            raise Exception(f"Node {node} already added to input nodes for {self.id}")
        self.edges.append(Edge(input_node=node, output_node=self, tags=tags))

    def add_output_node(self, node: Node, tags: list[str] = []):
        if node in self.output_nodes:
            raise Exception(f"Node {node} already added to input nodes for {self.id}")
        self.edges.append(Edge(output_node=node, input_node=self, tags=tags))

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
        if self.baseline_values is not None:
            self.logger.error('Baseline values already calculated')
        assert self.context.active_scenario.id == 'baseline'
        self.baseline_values = self.get_output_pl()

    def serialize_input_data(self):
        if len(self.input_dataset_instances) != 1:
            datasets = [x.id for x in self.input_dataset_instances]
            raise NodeError(self, 'Too many input datasets: %s' % ', '.join(datasets))
        ds = self.input_dataset_instances[0]
        df = ds.get_copy(self.context)
        if not isinstance(df, ppl.PathsDataFrame) or FORECAST_COLUMN not in df.columns:
            raise Exception('Dataset %s is not suitable for serialization')

        out = JSONDataset.serialize_df(df.to_pandas())
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
        except:
            # FIXME: Make this more robust
            unit = self.unit  # type: ignore
        self.input_dataset_instances[0] = JSONDataset(
            id=old_ds.id, data=d, unit=unit, tags=[],
        )

        self._last_historical_year = None

    def as_node_config_attributes(self):
        """Return a dict that can be used to set corresponding fields of NodeConfig."""
        attributes: dict[str, str | dict[str, str]] = {
            'identifier': self.id,
        }

        i18n = {}
        default_lang = self.context.instance.default_language

        def set_from_translated_str(s: str | TranslatedString | None, field_name: str):
            if s is None:
                return

            val, tr = get_modeltrans_attrs_from_str(s, field_name, default_lang)
            i18n.update(tr)
            attributes[field_name] = val

        set_from_translated_str(self.name, 'name')
        # Node's description is called `short_description` in NodeConfig and there is no equivalent for
        # NodeConfig's `long_description` in Node
        set_from_translated_str(self.description, 'short_description')

        attributes['i18n'] = i18n

        if self.color:
            attributes['color'] = self.color

        return attributes

    def warning(self, msg: str):
        self.context.warning('%s %s' % (str(self), msg))

    def add_nodes_pl(
        self, df: ppl.PathsDataFrame | None, nodes: List[Node], metric: str | None = None, keep_nodes: bool = False,
        node_multipliers: List[float] | None = None, unit: Unit | None = None,
    ) -> ppl.PathsDataFrame:
        if len(nodes) == 0:
            if df is None:
                raise NodeError(self, "No input dataset and no input nodes")
            return df
        if self.debug:
            print('%s: input dataset:' % self.id)
            if df is not None:
                print(self.print(df))
            else:
                print('\tNo input dataset')

        node_outputs: List[Tuple[Node, ppl.PathsDataFrame]] = []
        for node in nodes:
            node_df = node.get_output_pl(self, metric=metric)
            if node_df.paths.index_has_duplicates():
                raise NodeError(self, "Input from node '%s' has duplicate index rows" % node.id)
            if keep_nodes:
                node_df = node_df.with_columns(pl.lit(node.id).alias(NODE_COLUMN)).add_to_index(NODE_COLUMN)
            node_outputs.append((node, node_df))

        if df is None:
            node, df = node_outputs.pop(0)
            if self.debug:
                print('%s: adding output from node %s' % (self.id, node.id))
                self.print(df)
            if node_multipliers:
                mult = node_multipliers.pop(0)
                df = df.with_columns(pl.col(VALUE_COLUMN) * mult)
        else:
            if keep_nodes:
                df = df.with_columns(pl.lit('').alias(NODE_COLUMN)).add_to_index(NODE_COLUMN)

        cols = df.columns
        if VALUE_COLUMN not in cols:
            raise NodeError(self, "Value column missing in data")
        if FORECAST_COLUMN not in cols:
            raise NodeError(self, "Forecast column missing in data")

        if unit is None:
            unit = self.unit
            assert unit is not None

        df = df.ensure_unit(VALUE_COLUMN, unit)
        meta = df.get_meta()
        for node, node_df in node_outputs:
            if self.debug:
                print('%s: adding output from node %s' % (self.id, node.id))
                self.print(node_df)

            if VALUE_COLUMN not in node_df.columns:
                raise NodeError(self, "Value column missing in output of %s" % node.id)

            if node_multipliers:
                mult = node_multipliers.pop(0)  # FIXME Should this be the i-th multiplier, not the first?
                node_df = node_df.with_columns(pl.col(VALUE_COLUMN) * mult)

            ndf_meta = node_df.get_meta()
            if set(ndf_meta.dim_ids) != set(meta.dim_ids):
                raise NodeError(self, "Dimensions do not match with %s (%s vs. %s)" % (node.id,
                    ndf_meta.dim_ids, meta.dim_ids))
            df = df.paths.add_with_dims(node_df, how='outer')

        df = df.select([YEAR_COLUMN, *meta.dim_ids, VALUE_COLUMN, FORECAST_COLUMN])
        return df

    def check(self):
        from nodes.metric import Metric
        df = self.get_output_pl()
        for m in self.output_metrics.values():
            nulls = df.filter(pl.col(m.column_id).is_null() | pl.col(m.column_id).is_nan())
            if len(nulls):
                raise NodeError(self, 'Output has nulls or NaNs in column %s' % m.column_id)

        if self.baseline_values is not None:
            bdf = self.baseline_values
            for m in self.output_metrics.values():
                nulls = bdf.filter(pl.col(m.column_id).is_null() | pl.col(m.column_id).is_nan())
                if len(nulls):
                    raise NodeError(self, 'Baseline output has nulls or NaNs in column %s' % m.column_id)

        m = Metric.from_node(self)  # FIXME Should this be done for DimensionslMetric as well?
        if m is None:
            raise NodeError(self, "Output did not result in a Metric")
        else:
            fail = False
            for vals in (m.get_forecast_values(), m.get_historical_values()):
                for v in vals:
                    if v.value is None or v.value is float('nan'):
                        fail = True
                        break
                if fail:
                    break
            if fail:
                raise NodeError(self, 'Metric had nan or null values')
