from __future__ import annotations

import math
import typing

# import warnings
from contextlib import AbstractContextManager, nullcontext
from typing import Any, ClassVar, Literal, Self, cast, overload

from django.utils.translation import gettext_lazy as _

import numpy as np
import pandas as pd
import pint_pandas
import polars as pl
from polars.datatypes.group import INTEGER_DTYPES, NUMERIC_DTYPES
from rich import print as pprint
from sentry_sdk.consts import SPANSTATUS

from paths.const import NODE_CALC_OP

from common import polars as ppl
from common.i18n import I18nString, I18nStringInstance, TranslatedString, get_modeltrans_attrs_from_str
from common.types import Identifier, MixedCaseIdentifier, validate_identifier
from common.utils import hash_unit
from nodes.calc import extend_last_forecast_value_pl, extend_last_historical_value_pl
from nodes.constants import (
    DEFAULT_METRIC,
    FORECAST_COLUMN,
    NODE_COLUMN,
    UNCERTAINTY_COLUMN,
    VALUE_COLUMN,
    YEAR_COLUMN,
    ensure_known_quantity,
    get_quantity_icon,
)
from nodes.goals import NodeGoals
from params.param import ParameterWithUnit

from .datasets import Dataset, JSONDataset
from .edges import Edge
from .exceptions import NodeComputationError, NodeError, NodeMissingDefaultUnitError
from .units import Quantity, Unit, unit_registry

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import loguru
    from sentry_sdk.tracing import Span

    from common.cache import CacheResult
    from nodes.instance_loader import ConfigLocation
    from nodes.visualizations import NodeVisualizations
    from params import Parameter

    from .context import Context
    from .dimensions import Dimension
    from .models import NodeConfig
    from .node_cache import NodeHasher
    from .scenario import Scenario


class NodeMetric:
    """
    Represents a metric for a node in the calculation graph.

    NodeMetric defines the measurement characteristics of a node's output, including
    its unit of measurement, physical quantity type, and identification information.

    Attributes:
        id (MixedCaseIdentifier): Unique identifier for the metric
        column_id (str): Column name in the node's output DataFrame
        unit (Unit): Unit of measurement for the metric (e.g. 'MWh/a')
        quantity (str): Type of quantity being measured (e.g. 'energy')
        label (I18nString | None): Optional internationalized label for the metric
        default_unit (str | Unit): Unit of measurement passed as a string.
            If it is a string, it is parsed at a later point in initialization
            using the context's unit registry.

    Methods:
        calculate_hash: Generates a unique hash for the metric configuration

    """

    id: MixedCaseIdentifier  # FIXME: Convert to Identifier
    column_id: str
    unit: Unit
    quantity: str
    label: I18nString | None
    default_unit: str | Unit

    # __slots__ = ('id', 'unit', 'quantity', 'default_unit', 'label', 'column_id')

    def __init__(
        self,
        unit: str | Unit,
        quantity: str,
        id: str | None = None,
        label: I18nString | None = None,
        column_id: str | None = None,
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
    def from_config(cls, config: dict) -> Self:
        """Create a NodeMetric instance from a configuration dictionary."""
        return cls(
            unit=config['unit'],
            quantity=config['quantity'],
            id=config['id'],
            label=None,
            column_id=None,
        )

    def copy(self) -> NodeMetric:
        """Create a deep copy of the NodeMetric instance."""
        unit = getattr(self, 'unit', self.default_unit)
        return NodeMetric(unit=unit, quantity=self.quantity, id=self.id, label=self.label, column_id=self.column_id)

    def populate_unit(self, context: Context):
        """Initialize the unit attribute using the context's unit registry."""

        unit = self.default_unit
        if isinstance(unit, Unit):
            self.unit = unit
        else:
            self.unit = context.unit_registry.parse_units(unit)

    def calculate_hash(self) -> bytes:
        s = '%s:%s:%s' % (self.id, self.quantity, self.column_id)
        return s.encode('utf-8') + hash_unit(self.unit)

    def ensure_output_unit(self, node: Node, s: pd.Series, input_node: Node | None = None):
        if hasattr(s, 'pint'):
            s_u: Unit = s.pint.u
            if self.unit.dimensionality != s_u.dimensionality:
                if input_node is not None:
                    node_str = ' from node %s' % input_node.id
                else:
                    node_str = ''
                raise NodeError(
                    node,
                    'Series with type %s%s is not compatible with %s'
                    % (
                        s_u,
                        node_str,
                        self.unit,
                    ),
                )
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
    """
    Represents a node in the calculation graph.

    Nodes are the fundamental building blocks of the calculation model, representing
    data transformations, calculations, or data sources. Each node can have multiple
    inputs and outputs, and maintains its own metrics and parameters.

    The inputs can be datasets, other nodes, or parameters.
    """

    id: Identifier
    'Identifier of the Node instance.'

    database_id: int | None
    'The database row that corresponds to this Node instance.'

    db_obj: NodeConfig | None
    'The Django object for this Node Instance'

    name: I18nStringInstance
    'Human-readable label for the Node instance.'

    short_name: I18nStringInstance | None
    'Shortened label for the node'

    # Description for the Node instance
    #
    # This gets mapped to NodeType.short_description in the GraphQL schema and
    # wrapped in a <p> tag.
    description: I18nStringInstance | None
    'Long description for the node'

    # if the node has an established visualisation color
    color: str | None
    'Color for the node in visualizations'

    # order comes from NodeConfig
    order: int | None = None
    'Order of the node in visualizations'

    is_visible: bool = True
    'If the node should be visible in visualizations'

    is_outcome: bool = False
    'If the node is classified as a key outcome for the model'

    # optional grouping information for nodes
    node_group: str | None = None
    'Grouping information for the node (e.g. "transport" or "buildings")'

    unit: Unit | None
    """Unit of measurement for the node's output. Applies only to single-metric nodes."""

    default_unit: ClassVar[str]
    'Default unit for the node class (defined as a class variable)'

    quantity: str | None
    """Physical quantity of the node's output (e.g. "energy" or "emissions")"""

    minimum_year: int | None
    'Minimum allowed year for the node. All output before this year is filtered out.'

    allow_nulls: bool
    'If the node allows null values in the output'

    tags: set[str]
    'Tags to differentiate between multiple input/output nodes'

    output_metrics: dict[str, NodeMetric] = {}
    """Units and quantities for the node's output (for single-metric and multi-metric nodes)"""

    output_dimensions: dict[str, Dimension]
    "The dimensions that this node's output will contain."

    output_dimension_ids: list[str] = []
    "References to the dimensions that this node's output will contain (typically set in a class)."

    input_dimensions: dict[str, Dimension]
    "The dimensions that this node's input must contain."

    input_dimension_ids: list[str] = []
    "References to the dimensions that this node's input must contain (typically set in a class)."

    explanation: str | I18nString = "Text about the node class missing."
    'Textual explanation about what the node computes (typicallly set in a class).'

    # set if this node has a specific goal for the simulation target year
    goals: NodeGoals | None
    "Set if there are official, future goals for the node's output."

    visualizations: NodeVisualizations | None = None
    """
    Specific visualizations for the node.

    These are considered to illustrate especially well the upstream causalities
    affecting the node's output.
    """

    input_datasets: list[str]
    "List of input dataset identifiers for the node."
    input_dataset_instances: list[Dataset]
    "List of input dataset instances for the node."

    edges: list[Edge]
    "List of edges that connect this node to other nodes, both input and output."

    global_parameters: list[str] = []
    "List of identifiers for global parameters that affect the node's output."

    parameters: dict[str, Parameter]
    "Parameters with their values."

    allowed_parameters: ClassVar[Sequence[Parameter]]
    "All allowed parameters for this node class."

    _baseline_values: ppl.PathsDataFrame | None
    "Cached output for the node in the baseline scenario."

    _last_historical_year: int | None
    "Cached last historical year for in the node's output."

    context: Context
    "Computation context."

    hasher: NodeHasher
    "Cache helper for the node."

    logger: loguru.Logger
    "Logger for the node."

    debug: bool = False
    "If debug mode is enabled for the node. Will print extra debug information."

    disable_cache: bool = False
    "If caching should be disabled for this node. Used for debugging."

    config_location: ConfigLocation | None = None
    """Location of the node configuration in a YAML file"""

    def __post_init__(self): ...

    def finalize_init(self):
        """Customization and validation that is run after the node graph is fully configured."""  # noqa: D401
        pass

    @property
    def single_metric_unit(self) -> Unit:
        if self.unit is None:
            raise NodeMissingDefaultUnitError(self)
        return self.unit

    def _init_metrics(  # noqa: C901, PLR0912
        self,
        unit: Unit | None,
        quantity: str | None,
        output_metrics: dict[str, NodeMetric] | None = None,
    ) -> None:
        if output_metrics is not None:
            self.output_metrics = {metric_id: metric.copy() for metric_id, metric in output_metrics.items()}
        else:
            self.output_metrics = self.__class__.output_metrics.copy()
        if self.output_metrics:
            for met_id, met in self.output_metrics.items():
                met.populate_unit(self.context)
                met.id = validate_identifier(met_id, mixed=True)

            if len(self.output_metrics) == 1:
                # Single-metric node
                metric = next(iter(self.output_metrics.values()))
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
                raise NodeError(self, 'Must provide quantity')
            ensure_known_quantity(quantity)
            self.quantity = quantity
            self.unit = unit
            if unit is None:
                raise NodeError(self, 'Attempting to initialize node without a unit')
            # Create the default metric automatically for now
            self.output_metrics[DEFAULT_METRIC] = NodeMetric(
                unit,
                self.quantity,
                id=DEFAULT_METRIC,
                column_id=VALUE_COLUMN,
            )

        metric_cols = set()
        for metric in self.output_metrics.values():
            if metric.column_id in metric_cols:
                raise NodeError(self, 'Duplicate metric column IDs: %s' % metric.column_id)
            metric_cols.add(metric.column_id)

    def _init_dimensions(
        self,
        class_dims: dict[str, Dimension],
        arg_dims: list[str] | None,
        class_dim_ids: list[str],
    ) -> dict[str, Dimension]:
        dims = class_dims.copy()

        for dim_id, dim in dims.items():
            if not dim.is_internal and dim_id is not UNCERTAINTY_COLUMN:
                raise NodeError(self, 'Dimensions defined in class can only be internal ones')
            if dim_id in self.context.dimensions and dim_id is not UNCERTAINTY_COLUMN:
                raise NodeError(self, 'Internal dimension is also a global one')

        if arg_dims and class_dim_ids:
            if set(arg_dims) != set(class_dim_ids):
                raise NodeError(
                    self,
                    'Invalid dimensions supplied: %s; expecting: %s' % (', '.join(arg_dims), ', '.join(class_dim_ids)),
                )
        elif class_dim_ids:
            arg_dims = class_dim_ids

        if not arg_dims:
            return dims

        for dim_id in arg_dims:
            d = self.context.dimensions.get(dim_id)
            if not d:
                raise NodeError(self, 'Dimension %s not found' % dim_id)
            dims[dim_id] = d
        return dims

    def __init__(  # noqa: PLR0913, PLR0915
        self,
        id: str,
        context: Context,
        name: I18nStringInstance,
        short_name: I18nStringInstance | None = None,
        unit: Unit | None = None,
        quantity: str | None = None,
        minimum_year: int | None = None,
        description: I18nStringInstance | None = None,
        color: str | None = None,
        order: int | None = None,
        node_group: str | None = None,
        is_visible: bool = True,
        is_outcome: bool = False,
        target_year_goal: float | None = None,
        goals: dict | None = None,
        allow_nulls: bool = False,
        input_datasets: list[Dataset] | None = None,
        output_dimension_ids: list[str] | None = None,
        input_dimension_ids: list[str] | None = None,
        output_metrics: dict[str, NodeMetric] | None = None,
        config_location: ConfigLocation | None = None,
    ):
        from .node_cache import NodeHasher

        self.id = validate_identifier(id)
        self.context = context

        self._init_metrics(unit, quantity, output_metrics)
        if input_datasets is None:
            input_datasets = []

        self.database_id = None
        self.db_obj = None
        self.name = name
        self.config_location = config_location
        self.node_group = node_group
        if self.name is None:
            raise NodeError(self, 'Node has no name')
        self.short_name = short_name
        self.description = description
        self.color = color
        self.order = order
        self.is_visible = is_visible
        self.is_outcome = is_outcome
        self.minimum_year = minimum_year
        if goals is not None:
            self.goals = NodeGoals.model_validate(goals)
        elif target_year_goal is not None:
            is_main_goal = self.is_outcome
            self.goals = NodeGoals.model_validate(
                [dict(values=[dict(year=context.target_year, value=target_year_goal)], is_main_goal=is_main_goal)],
            )
        else:
            self.goals = None
        if self.goals is not None:
            self.goals.set_node(self)
        self.allow_nulls = allow_nulls

        self.input_dataset_instances = input_datasets
        self.edges = []
        self._baseline_values = None
        self.parameters = {}
        self.tags = set()

        kls = type(self)
        self.logger = context.log.bind(node=self.id, node_class='%s.%s' % (kls.__module__, kls.__qualname__))

        if not hasattr(self, 'global_parameters'):
            self.global_parameters = []
        else:
            # Copy the parameters so that the list can be mutated later
            self.global_parameters = list(self.global_parameters)

        if not hasattr(self, 'output_dimensions'):
            self.output_dimensions = {}
        self.output_dimensions = self._init_dimensions(
            self.output_dimensions,
            output_dimension_ids,
            self.output_dimension_ids,
        )

        if not hasattr(self, 'input_dimensions'):
            self.input_dimensions = {}
        self.input_dimensions = self._init_dimensions(
            self.input_dimensions,
            input_dimension_ids,
            self.input_dimension_ids,
        )

        self.hasher = NodeHasher(self)

        # Call the subclass post-init method if it is defined
        if hasattr(self, '__post_init__'):
            self.__post_init__()

        super().__init__()

    def add_parameter(self, param: Parameter):
        if param.local_id in self.parameters:
            msg = f'Local parameter {param.local_id} already defined for node {self.id}'
            raise Exception(msg)
        self.parameters[param.local_id] = param
        param.set_node(self)

    def notify_parameter_change(self, param: Parameter):
        """Notify the node that an input parameter changed."""
        # Propagate change notification to downstream nodes
        self.hasher.mark_modified()

    def get_parameters(self):
        yield from self.parameters.values()

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
            raise NodeError(self, f'Local parameter {local_id} not found for node {self.id}')
        return None

    @overload
    def get_parameter_value(self, id: str, *, required: Literal[True] = True, units: Literal[True]) -> Quantity: ...

    @overload
    def get_parameter_value(self, id: str, *, required: Literal[False], units: Literal[True]) -> Quantity | None: ...

    @overload
    def get_parameter_value(self, id: str, *, required: Literal[False], units: Literal[False] = ...) -> object | None: ...

    @overload
    def get_parameter_value(self, id: str, *, required: bool = True, units: Literal[False] = False) -> object: ...

    @overload
    def get_parameter_value(self, id: str, *, required: bool, units: bool) -> object | None: ...

    def get_parameter_value(self, id: str, *, required: bool = True, units: bool = False) -> object | None:
        param = self.get_parameter(id, required=required)
        if param is None:
            return None
        if units:
            unit = param.get_unit()
            return param.value * unit
        return param.value

    @overload
    def get_typed_parameter_value[T](self, param_id: str, param_type: type[T], *, required: Literal[True] = True) -> T: ...

    @overload
    def get_typed_parameter_value[T](
        self,
        param_id: str,
        param_type: type[T],
        *,
        required: Literal[False] = False,
    ) -> T | None: ...

    @overload
    def get_typed_parameter_value[T](
        self,
        param_id: str,
        param_type: type[T],
        *,
        required: bool = True,
    ) -> T | None: ...

    def get_typed_parameter_value[T](self, param_id: str, param_type: type[T], *, required: bool = True) -> T | None:
        val = self.get_parameter_value(param_id, required=required)
        if val is not None:
            if type(val) is not param_type:
                raise NodeError(
                    self,
                    "Parameter '%s' is of invalid type (required %s, got %s)" % (param_id, param_type, type(val)),
                )
            assert isinstance(val, param_type)
        return val

    @overload
    def get_parameter_value_str(self, param_id: str, *, required: Literal[True] = True) -> str: ...

    @overload
    def get_parameter_value_str(self, param_id: str, *, required: Literal[False]) -> str | None: ...

    def get_parameter_value_str(self, param_id: str, *, required: bool = True) -> str | None:
        return self.get_typed_parameter_value(param_id, str, required=required)


    @overload
    def get_parameter_value_int(self, param_id: str, *, required: Literal[True] = True) -> int: ...
    @overload
    def get_parameter_value_int(self, param_id: str, *, required: Literal[False]) -> int | None: ...

    def get_parameter_value_int(self, param_id: str, *, required: bool = True) -> int | None:
        ret = self.get_parameter_value(param_id, required=required)
        if ret is None:
            return None
        if isinstance(ret, float):
            return int(ret)
        assert isinstance(ret, int)
        return ret


    @overload
    def get_parameter_value_float(self, param_id: str, *, required: bool = True, units: Literal[True]) -> Quantity: ...
    @overload
    def get_parameter_value_float(self, param_id: str, *, required: bool = True, units: Literal[False] = False) -> float: ...
    @overload
    def get_parameter_value_float(self, param_id: str, *, required: bool, units: bool) -> Quantity | float | None: ...

    def get_parameter_value_float(self, param_id: str, *, required: bool = True, units: bool = False) -> float | Quantity | None:
        ret = self.get_parameter_value(param_id, required=required, units=units)
        if ret is None:
            return None
        if isinstance(ret, Quantity):
            return ret
        if isinstance(ret, int):
            return float(ret)
        assert isinstance(ret, float)
        return ret

    @overload
    def get_global_parameter_value(self, id: str, *, required: Literal[True] = True, units: Literal[True]) -> Quantity: ...

    @overload
    def get_global_parameter_value(self, id: str, *, required: Literal[False], units: Literal[True]) -> Quantity | None: ...

    @overload
    def get_global_parameter_value(self, id: str, *, required: Literal[True] = True, units: Literal[False] = False) -> object: ...

    @overload
    def get_global_parameter_value(self, id: str, *, required: Literal[False], units: Literal[False] = ...) -> object | None: ...

    def get_global_parameter_value(self, id: str, *, required: bool = True, units: bool = False) -> object | None:
        if id not in self.global_parameters:
            if not required:
                return None
            raise NodeError(self, f'Attempting to access global parameter {id} which is not declared in the node definition')
        if units:
            param = self.context.get_parameter(id, required=required)
            if param is None:
                return None
            if not hasattr(param, 'unit'):
                raise NodeError(self, f'Parameter {id} does not support units')
            assert isinstance(param, ParameterWithUnit)
            assert param.unit is not None
            return param.value * param.unit
        return self.context.get_parameter_value(id, required=required)

    def set_parameter_value(self, local_id: str, value: object | None, force: bool = False):
        # if force:  # FIXME if force, create this parameter
        #     self.parameters[local_id] = Parameter()
        if local_id not in self.parameters:
            raise NodeError(self, f'Local parameter {local_id} not found for node {self.id}')
        self.parameters[local_id].set(value)

    def get_input_datasets_pl(self, tag: str | None = None, exclude_tags: list[str] | None = None) -> list[ppl.PathsDataFrame]:
        dfs: list[ppl.PathsDataFrame] = []
        for ds in self.input_dataset_instances:
            if tag is not None and tag not in ds.tags:
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
                print(df.paths.duplicated_index_rows())
                raise NodeError(self, 'Input dataset has duplicate index rows. See rows above.')
            assert isinstance(df, ppl.PathsDataFrame)
            dfs.append(df)

        return dfs

    def get_input_datasets(self) -> list[pd.DataFrame]:
        dfs = self.get_input_datasets_pl()
        return [df.to_pandas() for df in dfs]

    @overload
    def get_input_dataset_pl(self, tag: str | None = None, *, required: Literal[False]) -> ppl.PathsDataFrame | None: ...

    @overload
    def get_input_dataset_pl(self, tag: str | None = None, required: Literal[True] = True) -> ppl.PathsDataFrame: ...

    @overload
    def get_input_dataset_pl(self, tag: str | None = None, required: bool = ...) -> ppl.PathsDataFrame | None: ...

    def get_input_dataset_pl(self, tag: str | None = None, required: bool = True) -> ppl.PathsDataFrame | None:
        """Get the first (and only) dataset if it exists."""
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
    def get_input_dataset(self, required: Literal[False]) -> pd.DataFrame | None: ...

    def get_input_dataset(self, required: bool = True) -> pd.DataFrame | None:
        df = self.get_input_dataset_pl(required=required)
        if df is None:
            return None
        return df.to_pandas()

    def get_input_nodes(self, tag: str | None = None, quantity: str | None = None) -> list[Node]:
        matching_nodes = []
        for edge in self.edges:
            if edge.output_node != self:
                continue
            node = edge.input_node
            if tag is not None and tag not in edge.tags and tag not in node.tags:
                continue
            if quantity is not None and node.quantity != quantity:
                # FIXME: Multi-metric support
                continue
            matching_nodes.append(node)
        return matching_nodes

    @overload
    def get_input_node(self, tag: str | None = None, quantity: str | None = None, *, required: Literal[False]) -> Node | None: ...

    @overload
    def get_input_node(self, tag: str | None = None, quantity: str | None = None, required: Literal[True] = True) -> Node: ...

    def get_input_node(self, tag: str | None = None, quantity: str | None = None, required: bool = True) -> Node | None:
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

    @overload
    def get_default_output_metric(self, required: Literal[True] = True) -> NodeMetric: ...

    @overload
    def get_default_output_metric(self, required: Literal[False]) -> NodeMetric | None: ...

    def get_default_output_metric(self, required: bool = True) -> NodeMetric | None:
        if DEFAULT_METRIC in self.output_metrics:
            return self.output_metrics[DEFAULT_METRIC]
        if len(self.output_metrics) > 1:
            if required:
                raise NodeError(self, 'Node outputs multiple output metrics, but none of them is set as default')
            return None
        return next(iter(self.output_metrics.values()))

    def _get_output_for_node(self, df: ppl.PathsDataFrame, edge: Edge) -> ppl.PathsDataFrame:  # noqa: C901, PLR0912
        """
        Filter and select the dataframe.

        It performs the following steps:

        * Choose rows that are specific to the target node.
        * Choose the column that is defined in edge.metrics.
        * Choose the only metric column.
        * Remove rows where there are only nulls.
        * Choose the column that has the name of the target node.
        * Choose the column that has the name of the target node quantity.
        """
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
            if len(edge.metrics) == 1 and edge.metrics[0] != VALUE_COLUMN:
                df = df.rename({edge.metrics[0]: VALUE_COLUMN})
                remain_cols = [VALUE_COLUMN]
            # Drop rows where all metric cols are null
            df = df.filter(~pl.all_horizontal(pl.col(col).is_null() for col in remain_cols))
            return df

        col_name: str | None = None

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

    def _get_output_for_target(  # noqa: C901, PLR0912, PLR0915
            self, df: ppl.PathsDataFrame,
            target_node: Node,
            skip_dim_test: bool = False
        ) -> ppl.PathsDataFrame:
        for edge in self.edges:
            if edge.output_node == target_node:
                break
        else:
            raise NodeError(self, 'No connection to target node %s' % target_node.id)

        df = self._get_output_for_node(df, edge)

        # Slice a dimension if the parameter is given.
        params = self.get_parameter_value_str('slice_category_at_edge', required=False)
        if params:
            param_arr = params.split(',')
            assert len(param_arr) == 1
            param_arr = param_arr[0].split(':')
            df = df.filter(pl.col(params[0]).eq(params[1]))

        if edge.from_dimensions:
            meta = df.get_meta()
            if not meta.dim_ids:
                raise NodeError(self, 'No dimensions in node output')
            for dim_id, edge_dim in edge.from_dimensions.items():
                cat_ids = [cat.id for cat in edge_dim.categories]
                if dim_id not in meta.dim_ids:
                    raise NodeError(self, 'Dimension %s not in output df' % dim_id)
                filter_expr = pl.col(dim_id).is_in(cat_ids)
                if edge_dim.exclude:
                    filter_expr = pl.col(dim_id).is_null() | ~filter_expr
                df = df.filter(filter_expr)
                if len(df) == 0:
                    raise NodeError(self, 'No rows left after filtering by %s' % dim_id)
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
                    raise NodeError(self, 'to_dimensions can have only one category for now')
                if dim_id in df.columns:
                    raise NodeError(self, 'attempting to assign a category to an existing dimension')
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
                    self,
                    'Dimensions (%s) do not match in output for %s (expecting %s)'
                    % (
                        set(df.dim_ids),
                        target_node,
                        output_dimensions,
                    ),
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
        # if set(meta.dim_ids) != output_dimensions and not skip_dim_test: # TODO Consider testing this in node validation
        #     print(df)
        #     out_dims = ', '.join(meta.dim_ids)
        #     target_in_dims = ', '.join(output_dimensions)
        #     warnings.warn( # Changed from NodeError to print because blocks useful generic operations.
        #         # self,
        #         'Dimensions of output [%s] do not match the input dimensions of %s [%s]'
        #         % (
        #             out_dims,
        #             target_node.id,
        #             target_in_dims,
        #         ),
        #     )

        return df

    def validate_output(self, df: ppl.PathsDataFrame) -> None:  # noqa: C901, PLR0912
        meta = df.get_meta()

        if YEAR_COLUMN not in meta.primary_keys:
            raise NodeError(self, "'%s' column missing" % YEAR_COLUMN)

        if df.schema[YEAR_COLUMN] not in INTEGER_DTYPES:
            raise NodeError(self, "Invalid dtype for 'Year': %s" % df.schema[YEAR_COLUMN])

        ldf = df.lazy()
        dupe_rows = ldf.group_by(meta.primary_keys).agg(pl.len().alias('count')).filter(pl.col('count') > 1).sort(YEAR_COLUMN)
        has_dupes = bool(len(dupe_rows.limit(1).collect()))
        if has_dupes:
            print(dupe_rows.collect())
            print(df)
            raise NodeError(self, 'Node output has duplicate index rows')

        if FORECAST_COLUMN in df.columns:
            if df[FORECAST_COLUMN].dtype != pl.Boolean:
                raise NodeError(self, 'Forecast column is not a boolean')
        else:
            raise NodeError(self, 'Forecast column missing')

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

            dt = df.schema[metric.column_id]
            if dt not in NUMERIC_DTYPES:
                raise NodeError(self, "Output column '%s' is of wrong type (%s)" % (metric.column_id, dt))
            continue
            # all_hsy_emissions still has nulls
            col = df[metric.column_id]
            if (col.is_nan() | col.is_null()).sum() > 0:
                self.print(df)
                raise NodeError(self, "Output column '%s' has NaN or null values" % metric.column_id)

        if NODE_COLUMN in meta.primary_keys:
            # FIXME
            return

        self.validate_dims(df)

    def validate_dims(self, df: ppl.PathsDataFrame) -> None:
        meta = df.get_meta()

        for dim_id, dim in self.output_dimensions.items():
            if dim_id not in meta.primary_keys:
                raise NodeError(self, "Dimension column '%s' not included in index" % dim_id)
            dt = df[dim_id].dtype
            if dt not in (pl.Utf8, pl.Categorical):
                raise NodeError(self, "Dimension column '%s' is of wrong type (%s)" % (dim_id, dt))

            if dim.is_internal or dim_id == UNCERTAINTY_COLUMN:
                # Skip validation for internal dimensions and uncertainty column
                continue

            cats = set(df[dim_id].cast(pl.Utf8).unique())
            if getattr(self, 'allow_null_categories', None):
                cats -= {'', None}

            dim_cats = dim.get_cat_ids()
            diff = cats - dim_cats
            if diff:
                raise NodeError(self, "Unknown categories in dimension column '%s': %s" % (dim_id, ', '.join(diff)))

        dim_ids = set(meta.dim_ids)
        node_dims = set(self.output_dimensions.keys())
        if dim_ids != node_dims and not getattr(self, 'allow_unknown_dimensions', None):
            raise NodeError(self, 'Output has unknown dimensions: %s (expecting %s)' % (', '.join(dim_ids), ', '.join(node_dims)))

    def get_output(self, target_node: Node | None = None, metric: str | None = None) -> pd.DataFrame:
        df = self.get_output_pl(target_node, metric)
        return df.to_pandas()

    def _process_edge_output(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        for edge in self.edges:
            if edge.output_node == target_node:
                break
        else:
            raise NodeError(self, 'No connection to target node %s' % target_node.id)

        for tag in edge.tags:
            try:
                edge_function = getattr(self, f'_{tag}')
                df = edge_function(df, target_node)
            except AttributeError:
                continue # Not every tag has a function attached.

        return df

    def get_output_pl(  # noqa: C901, PLR0912
        self,
        target_node: Node | None = None,
        metric: str | None = None,
        extra_span_desc: str | None = None,
        skip_dim_test: bool = False
    ) -> ppl.PathsDataFrame:
        perf_cm = self.context.perf_context
        span_ctx: AbstractContextManager[None | Span]
        if self.hasher.is_cached(run=True, local=True):
            span_ctx = nullcontext()
        else:
            if extra_span_desc:
                span_name = '%s:get (%s)' % (self.id, extra_span_desc)
            else:
                span_name = '%s:get' % self.id
            span_ctx = self.context.start_span(
                span_name, op=NODE_CALC_OP, attributes=dict(node_id=self.id, node_class=self.__class__.__name__),
            )
        with span_ctx as span, perf_cm.exec_node(self) as node_run:
            try:
                df, cache_res = self._get_output_pl(target_node=target_node, metric=metric, skip_dim_test=skip_dim_test)
            except Exception as e:
                if isinstance(e, NodeComputationError):
                    e.add_node(self)
                    raise
                if target_node:
                    raise NodeComputationError(self, "Error getting output for node '%s'" % target_node.id) from e
                raise NodeComputationError(self, 'Error getting output') from e
            if span is not None:
                if cache_res is None or not cache_res.is_hit:
                    span.set_data('cache', 'miss')
                else:
                    span.set_data('cache', cache_res.kind.name)
            if node_run is not None and cache_res is not None:
                node_run.mark_cache(cache_res)

            if target_node is not None:
                df = self._process_edge_output(df, target_node)

            if span is not None:
                span.set_status(SPANSTATUS.OK)
        return df

    def _get_output_pl(  # noqa: C901
        self,
        target_node: Node | None = None,
        metric: str | None = None,
        skip_dim_test: bool = False,
    ) -> tuple[ppl.PathsDataFrame, CacheResult[ppl.PathsDataFrame] | None]:
        use_cache = not (self.disable_cache or self.context.skip_cache)
        cache_res = None
        if use_cache:
            cache_res = self.hasher.get_cached_output()

        if cache_res is None or not cache_res.is_hit:
            try:
                df = self.compute()
            except Exception as e:
                self.context.log.error('Exception when computing node %s: %s' % (str(self), str(e)))
                raise
            if df is None:
                raise NodeError(self, 'Node returned no output')

            if isinstance(df, pd.DataFrame):
                df = ppl.from_pandas(df)

            self.validate_output(df)
            if cache_res is not None:
                self.hasher.cache_output(cache_res, df)
        else:
            assert cache_res.obj is not None
            df = cache_res.obj

        assert isinstance(df, ppl.PathsDataFrame)

        meta = df.get_meta()

        if self.minimum_year is not None:
            df = df.filter(pl.col(YEAR_COLUMN) >= self.minimum_year)

        # If a node has multiple outputs, we can specify only one series
        # to include.
        if metric is not None:
            assert metric in df.columns  # If target_node has metric, self MUST have that also.
            cols = meta.primary_keys.copy()
            cols.append(metric)
            if FORECAST_COLUMN in df.columns:
                cols.append(FORECAST_COLUMN)
            df = df.select(cols)
            df = df.rename({metric: VALUE_COLUMN})
            return (df, cache_res)  # FIXME I'd like to comment this out because if metric, to_dimensions is ignored.

        if target_node is not None:
            df = self._get_output_for_target(df, target_node, skip_dim_test=skip_dim_test)

        return (df, cache_res)

    def print_output(self, only_years: list[int] | None = None, filters: list[str] | None = None):
        from .debug import print_node_output

        print_node_output(self, only_years=only_years, filters=filters)

    def plot_output(self, filters: list[str] | None = None):
        from .debug import plot_node_output

        plot_node_output(self, filters=filters)

    def print_pint_df(self, df: pd.DataFrame | pd.Series):
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
        if isinstance(obj, pd.DataFrame | pd.Series):
            self.print_pint_df(obj)
            return
        # if isinstance(obj, ppl.PathsDataFrame):
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

    def is_compatible_unit(self, unit_a: str | Unit | None, unit_b: str | Unit | None):
        assert unit_a is not None, f'Unit is missing in node {self.id}. Is it multimetric?'
        assert unit_b is not None, f'Unit {unit_b} is missing when comparing to node {self.id}'
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
            raise NodeError(
                self,
                'Series with type %s is not compatible with %s'
                % (
                    s.pint.units,
                    unit,
                ),
            )
        return s.astype(pint_pandas.PintType(unit))

    def ensure_output_unit(self, s: pd.Series, input_node: Node | None = None):
        if self.unit is None:
            raise NodeError(self, 'Does currently not work on multi-metric nodes')

        metric = self.output_metrics.get(DEFAULT_METRIC)
        if metric is None:
            assert len(self.output_metrics) == 1
            metric = next(iter(self.output_metrics.values()))
        return metric.ensure_output_unit(self, s, input_node)

    def get_downstream_nodes(
        self, *, max_depth: int | None = None, only_outcome: bool = False, until_node: Node | None = None
    ) -> list[Node]:
        import networkx as nx

        node_ids = set[str]()
        if until_node is not None:
            res = nx.all_simple_paths(G=self.context.node_graph, source=self.id, target=until_node.id, cutoff=max_depth)
            for path in res:
                node_ids.update(path)
        else:
            res = nx.bfs_successors(G=self.context.node_graph, source=self.id, depth_limit=max_depth)
            for __, children in res:
                node_ids.update(children)
        nodes: list[Node] = []
        if self.id in node_ids:
            node_ids.remove(self.id)
        for node_id in node_ids:
            node = self.context.get_node(node_id)
            if only_outcome and not node.is_outcome:
                continue
            nodes.append(node)
        return nodes

    def get_upstream_nodes(self, filter_func: Callable[[Node], bool] | None = None) -> list[Node]:
        from nodes.actions.action import ActionNode
        from nodes.actions.parent import ParentActionNode

        result = []
        closed = set()
        open_nodes = self.input_nodes.copy()
        while open_nodes:
            current = open_nodes.pop()
            if current in closed:
                continue
            closed.add(current)
            if filter_func is None or filter_func(current):
                result.append(current)
            if isinstance(current, ActionNode) and current.parent_action is not None:
                open_nodes.append(current.parent_action)
            if isinstance(current, ParentActionNode):
                open_nodes += current.subactions
            open_nodes += current.input_nodes
        return result

    def on_scenario_created(self, scenario: Scenario):
        """Called when a scenario is created with this node among the nodes to be notified."""  # noqa: D401
        pass

    def get_icon(self) -> str | None:
        from nodes.actions.action import ActionNode

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

    def add_input_node(self, node: Node, tags: list[str] | None = None):
        if tags is None:
            tags = []
        if node in self.input_nodes:
            msg = f'Node {node} already added to input nodes for {self.id}'
            raise Exception(msg)
        self.edges.append(Edge(input_node=node, output_node=self, tags=tags))

    def add_output_node(self, node: Node, tags: list[str] | None = None):
        if tags is None:
            tags = []
        if node in self.output_nodes:
            msg = f'Node {node} already added to input nodes for {self.id}'
            raise Exception(msg)
        self.edges.append(Edge(output_node=node, input_node=self, tags=tags))

    def add_edge(self, edge: Edge):
        if edge.input_node == self:
            if edge.output_node in self.output_nodes:
                raise NodeError(self, f'Node {edge.output_node} already added to output nodes')
        elif edge.output_node == self:
            if edge.input_node in self.input_nodes:
                raise NodeError(self, f'Node {edge.input_node} already added to input nodes')
        else:
            raise NodeError(self, f'Attempting to add edge: {edge.input_node} -> {edge.output_node}')
        self.edges.append(edge)

    def is_connected_to(self, other: Node):
        import networkx as nx

        return nx.has_path(self.context.node_graph, self.id, other.id)

    def get_baseline_values(self) -> ppl.PathsDataFrame:
        if self._baseline_values is not None:
            return self._baseline_values
        if self.context.active_scenario.id != 'baseline':
            baseline = self.context.scenarios['baseline']
            with baseline.override():
                df = self.get_output_pl()
        else:
            df = self.get_output_pl()
        self._baseline_values = df
        return df

    def baseline_values_calculated(self) -> bool:
        return self._baseline_values is not None

    def generate_baseline_values(self):
        if self._baseline_values is not None:
            self.logger.error('Baseline values already calculated')
        assert self.context.active_scenario.id == 'baseline'
        self._baseline_values = self.get_output_pl()

    def serialize_input_data(self):
        if len(self.input_dataset_instances) != 1:
            datasets = [x.id for x in self.input_dataset_instances]
            raise NodeError(self, 'Too many input datasets: %s' % ', '.join(datasets))
        ds = self.input_dataset_instances[0]
        df = ds.get_copy(self.context)
        if not isinstance(df, ppl.PathsDataFrame) or FORECAST_COLUMN not in df.columns:
            msg = f'Dataset {ds.id} is not suitable for serialization'
            raise Exception(msg)

        out = JSONDataset.serialize_df(df)
        return out

    def as_node_config_attributes(self):
        """Return a dict that can be used to set corresponding fields of NodeConfig."""
        attributes: dict[str, str | dict[str, str]] = {
            'identifier': self.id,
        }

        i18n = {}
        default_lang = self.context.instance.default_language

        def set_from_translated_str(s: str | TranslatedString | None, field_name: str) -> None:
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

    def warning(self, msg: str, depth: int = 0, **kwargs):
        ctx = {'node.id': self.id}
        self.context.warning('%s %s' % (str(self), msg), depth=depth + 1, **ctx, **kwargs)

    def add_nodes_pl(  # noqa: C901, PLR0912, PLR0915
        self,
        df: ppl.PathsDataFrame | None,
        nodes: list[Node],
        metric: str | None = None,
        keep_nodes: bool = False,
        node_multipliers: list[float] | None = None,
        unit: Unit | None = None,
        start_from_year: int | None = None,
        ignore_unit: bool = False,
    ) -> ppl.PathsDataFrame:
        if len(nodes) == 0:
            if df is None:
                raise NodeError(self, 'No input dataset and no input nodes')
            return df
        if self.debug:
            print('%s: input dataset:' % str(self.id))
            if df is not None:
                print(self.print(df))
            else:
                print('\tNo input dataset')

        node_outputs: list[tuple[Node, ppl.PathsDataFrame]] = []
        for node in nodes:
            node_df = node.get_output_pl(self, metric=metric)
            if start_from_year is not None:
                node_df = node_df.filter(pl.col(YEAR_COLUMN) >= start_from_year)
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
        elif keep_nodes:
            df = df.with_columns(pl.lit('').alias(NODE_COLUMN)).add_to_index(NODE_COLUMN)

        cols = df.columns
        if VALUE_COLUMN not in cols:
            raise NodeError(self, 'Value column missing in data')
        if FORECAST_COLUMN not in cols:
            raise NodeError(self, 'Forecast column missing in data')
        if not ignore_unit:
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
                raise NodeError(self, 'Value column missing in output of %s' % node.id)

            if node_multipliers:
                mult = node_multipliers.pop(0)
                node_df = node_df.with_columns(pl.col(VALUE_COLUMN) * mult)  # noqa: PLW2901

            ndf_meta = node_df.get_meta()
            if set(ndf_meta.dim_ids) != set(meta.dim_ids):
                raise NodeError(self, 'Dimensions do not match with %s (%s vs. %s)' % (node.id, ndf_meta.dim_ids, meta.dim_ids))
            df = df.paths.add_with_dims(node_df, how='outer')

        df = df.select([YEAR_COLUMN, *meta.dim_ids, VALUE_COLUMN, FORECAST_COLUMN])
        return df

    def multiply_nodes_pl(
            self,
            df: ppl.PathsDataFrame | None,
            nodes: list[Node],
            metric: str | None = None,
            unit: Unit | None = None,
            start_from_year: int | None = None,
        ) -> ppl.PathsDataFrame | None:
        """Multiply outputs from the given nodes using inner join and union of dimensions."""
        if len(nodes) == 0:
            if df is None:
                return None
            return df

        result_df = df

        for node in nodes:
            node_df = node.get_output_pl(self, metric=metric)
            if start_from_year is not None:
                node_df = node_df.filter(pl.col(YEAR_COLUMN) >= start_from_year)

            if self.debug:
                print('%s: multiplying with output from node %s' % (self.id, node.id))
                self.print(node_df)

            if VALUE_COLUMN not in node_df.columns:
                raise NodeError(self, f'Value column missing in output of {node.id}')

            if result_df is None:
                result_df = node_df
            else:
                result_df = result_df.paths.multiply_with_dims(node_df)

        if unit is not None and result_df is not None:
            result_df = result_df.ensure_unit(VALUE_COLUMN, unit)

        return result_df

    def check(self):
        from nodes.metric import DimensionalMetric

        df = self.get_output_pl()
        for m in self.output_metrics.values():
            nulls = df.filter(pl.col(m.column_id).is_null())
            if len(nulls) and not self.allow_nulls:
                raise NodeError(self, 'Output has nulls in column %s' % m.column_id)
            nans = df.filter(pl.col(m.column_id).is_nan())
            if len(nans):
                raise NodeError(self, 'Output has NaNs in column %s' % m.column_id)

        if self._baseline_values is not None:
            bdf = self._baseline_values
            for m in self.output_metrics.values():
                nulls = bdf.filter(pl.col(m.column_id).is_null() | pl.col(m.column_id).is_nan())
                if len(nulls):
                    raise NodeError(self, 'Baseline output has nulls or NaNs in column %s' % m.column_id)

        dm = DimensionalMetric.from_node(self)
        if dm is not None:
            for v in dm.values:
                if math.isnan(v):
                    raise NodeError(self, 'Output metric has NaNs')

    def _scale_by_reference_year(self, df: ppl.PathsDataFrame, year: int | None = None) -> ppl.PathsDataFrame:
        if not year:
            return df
        if len(df.dim_ids) == 0:
            reference = df.filter(pl.col(YEAR_COLUMN).eq(year))[VALUE_COLUMN][0]
            df = df.with_columns((pl.col(VALUE_COLUMN) / pl.lit(reference)).alias(VALUE_COLUMN))
        else:
            meta = df.get_meta()
            reference = df.filter(pl.col(YEAR_COLUMN).eq(year))
            zdf = df.join(reference, on=df.dim_ids)
            zdf = zdf.with_columns((pl.col(VALUE_COLUMN) / pl.col(VALUE_COLUMN + '_right')).alias(VALUE_COLUMN))
            zdf = zdf.drop([VALUE_COLUMN + '_right', FORECAST_COLUMN + '_right', YEAR_COLUMN + '_right'])
            df = ppl.to_ppdf(zdf, meta=meta)

        df = df.clear_unit(VALUE_COLUMN)
        df = df.set_unit(VALUE_COLUMN, 'dimensionless')
        return df

    def _absolute(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        return df.with_columns(pl.col(VALUE_COLUMN).abs().alias(VALUE_COLUMN))

    def _arithmetic_inverse(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        return df.multiply_quantity(VALUE_COLUMN, unit_registry('-1 * dimensionless'))

    def _complement(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        if not df.get_unit(VALUE_COLUMN).is_compatible_with('dimensionless'):
            raise NodeError(
                self, 'The unit of node %s must be compatible with dimensionless for taking complement' % self.id,
            )
        if self.quantity not in ['fraction', 'probability']:
            raise NodeError(
                self, 'The quantity of node %s must be fraction or probability for taking complement' % self.id,
            )
        df = df.ensure_unit(VALUE_COLUMN, unit='dimensionless')  # TODO CHECK
        return df.with_columns((pl.lit(1.0) - pl.col(VALUE_COLUMN)).alias(VALUE_COLUMN))

    def _complement_cumulative_product(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        return df.cumprod(VALUE_COLUMN, complement=True)

    def _cumulative(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        return df.cumulate(VALUE_COLUMN)

    def _cumulative_product(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        return df.cumprod(VALUE_COLUMN)

    def _difference(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        return df.diff(VALUE_COLUMN)

    def _empty_to_zero(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        return df.with_columns(
            pl.when(pl.col(VALUE_COLUMN).is_nan())
            .then(pl.lit(0.0))
            .otherwise(pl.col(VALUE_COLUMN))
            .alias(VALUE_COLUMN),
        )

    def _expectation(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        if UNCERTAINTY_COLUMN in df.columns:
            meta = df.get_meta()
            cols = [col for col in df.primary_keys if col != UNCERTAINTY_COLUMN]
            dfp = df.group_by(cols, maintain_order=True).agg([
                pl.col(VALUE_COLUMN).mean().alias(VALUE_COLUMN),
                pl.col(FORECAST_COLUMN).any().alias(FORECAST_COLUMN)
            ])
            dfp = dfp.with_columns(pl.lit('expectation').alias(UNCERTAINTY_COLUMN))
            df = ppl.to_ppdf(dfp, meta)
        return df

    def _extend_values(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        return extend_last_historical_value_pl(df, self.get_end_year())

    def _extend_forecast_values(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        return extend_last_forecast_value_pl(df, self.get_end_year())

    def _forecast_only(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        return df.filter(pl.col(FORECAST_COLUMN))

    def _geometric_inverse(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        return df.divide_quantity(VALUE_COLUMN, unit_registry('1 * dimensionless'))

    def _ignore_content(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        no_effect_value = getattr(self, 'no_effect_value', 0.0)
        df = df.with_columns(pl.lit(no_effect_value).alias(VALUE_COLUMN))
        m = target_node.get_default_output_metric()
        return df.set_unit(VALUE_COLUMN, m.unit, force=True)

    def _indifferent_history_ratio(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        return df.with_columns(
            pl.when(pl.col(FORECAST_COLUMN))
            .then(pl.col(VALUE_COLUMN))
            .otherwise(pl.lit(1.0)).alias(VALUE_COLUMN)
        )

    def _inventory_only(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        df = df.with_columns(  # TODO A non-elegant way to ensure there is at least one historical row.
            pl.when(pl.col(FORECAST_COLUMN) & (pl.count() == 1))
            .then(pl.lit(value=False))
            .otherwise(pl.col(FORECAST_COLUMN))
            .alias(FORECAST_COLUMN),
        )
        return df.filter(pl.col(FORECAST_COLUMN) == pl.lit(value=False))

    def _make_nonnegative(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        return df.with_columns(pl.max_horizontal(VALUE_COLUMN, 0.0))

    def _make_nonpositive(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        return df.with_columns(pl.min_horizontal(VALUE_COLUMN, 0.0))

    def _ratio_to_last_historical_value(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        year = cast('int', df.filter(~df[FORECAST_COLUMN])[YEAR_COLUMN].max())
        return self._scale_by_reference_year(df, year)

    def _truncate_before_start(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        baseline_year = self.context.instance.reference_year
        return df.filter(pl.col(YEAR_COLUMN).ge(baseline_year))

    def _truncate_beyond_end(self, df: ppl.PathsDataFrame, target_node: Node) -> ppl.PathsDataFrame:
        return df.filter(pl.col(YEAR_COLUMN).le(self.get_end_year()))

    tag_descriptions = {
        'additive': _("Add input node values (even if the units don't match with the node units)."),
        'arithmetic_inverse': _('Take the arithmetic inverse of the values (-x).'),
        'complement': _('Take the complement of the dimensionless values (1-x).'),
        'complement_cumulative_product': _('Take the cumulative product of the dimensionless complement values over time.'),
        'cumulative': _('Take the cumulative sum over time.'),
        'cumulative_product': _('Take the cumulative product of the dimensionless values over time.'),
        'difference': _('Take the difference over time (i.e. annual changes)'),
        'empty_to_zero': _('Convert NaNs to zeros.'),
        'expectation': _('Take the expected value over the uncertainty dimension.'),
        'extend_values': _('Extend the last historical values to the remaining missing years.'),
        'extend_forecast_values': _('Extend the last forecast values to the remaining missing years.'),
        'geometric_inverse': _('Take the geometric inverse of the values (1/x).'),
        'goal': _('The node is used as the goal for the action.'),
        'historical': _('The node is used as the historical starting point.'),
        'existing': _('This is used as the baseline.'),
        'incoming': _('This is used for the incoming stock.'),
        'ignore_content': _('Show edge on graphs but ignore upstream content.'),
        'inserting': _('This is the rate of new stock coming in.'),
        'inventory_only': _('Truncate the forecast values.'),
        'make_nonnegative': _('Negative result values are replaced with 0.'),
        'make_nonpositive': _('Positive result values are replaced with 0.'),
        'non_additive': _('Input node values are not added but operated despite matching units.'),
        'ratio_to_last_historical_value': _('Take the ratio of the values compared with the last historical value.'),
        'removing': _('This is the rate of stock removal.'),
        'truncate_before_start': _('Truncate values before the reference year. There may be some from data'),
        'truncate_beyond_end': _('Truncate values beyond the model end year. There may be some from data'),
    }

    def get_explanation(self):
        """Generate an HTML explanation of this node's processing logic and inputs."""

        html = []

        # Start with the explanation text
        if self.explanation:
            html.append(f"<p>{self.explanation}")
        if 'operations' in self.parameters.keys():
            operations = self.get_parameter_value_str('operations', required=False)
            html.append(f"_(The order of operations is) {operations}.")
        html.append("</p>")

        # Add formula if available # TODO Also describe other parameters.
        if 'formula' in self.parameters.keys():
            formula = self.get_parameter_value_str('formula', required=False)
            html.append(f"<p>{_('The formula is:')}</p>")
            html.append(f"<pre>{formula}</pre>")

        # # Handle input nodes # FIXME Add operations when the buckets is a node attribute.
        # operation_nodes = [getattr(n, 'translated_name', n.name) for n in self.input_nodes]
        # if operation_nodes:
        #     html.append(f"<p>{_('The node has the following input nodes:')}</p>")
        #     html.append("<ul>")
        #     html.extend([f"<li>{node_name}</li>" for node_name in operation_nodes])
        #     html.append("</ul>")
        # else:
        #     html.append(f"<p>{_('The node does not have input nodes.')}</p>")


        # Add datasets information
        dataset_html = []
        if self.input_dataset_instances:
            df = self.get_output_pl()
            dataset_html.append(f"<p>{_('The node has the following datasets:')}</p>")
            dataset_html.append("<ul>")
            dataset_html.extend(df.explanation)
            dataset_html.append("</ul>")

        edge_html = self.get_edge_explanation()
        # Combine all parts
        if edge_html:
            html.extend(edge_html)
        if dataset_html:
            html.extend(dataset_html)

        return "".join(html)

    def get_edge_explanation(self):
        edge_html = []
        edge_html.append(f"<p>{_(
            'The input nodes are processed in the following way before using as input for calculations in this node:'
        )}</p>")
        edge_html.append("<ul>")  # Start the main list for nodes
        edge_html0 = edge_html.copy()

        for node in self.input_nodes:
            for edge in node.edges:
                if edge.output_node != self:
                    continue

                tag_html = self.get_explanation_for_edge_tag(edge)
                from_html = self.get_explanation_for_edge_from(edge)
                to_html = self.get_explanation_for_edge_to(edge)

            if tag_html or from_html or to_html:

                node_name = getattr(node, 'translated_name', node.name)

                # Create a list item for the node with nested list
                edge_html.append(f"<li>{_('Node')} <i>{node_name}</i>:")
                edge_html.append("<ul>")  # Start nested list for this node
                edge_html.extend(tag_html)
                edge_html.extend(from_html)
                edge_html.extend(to_html)
                edge_html.append("</ul>")  # Close node's nested list
                edge_html.append("</li>")  # Close node list item

        if edge_html == edge_html0:
            return []
        edge_html.append("</ul>")  # Close main nodes list
        return edge_html

    def get_explanation_for_edge_tag(self, edge):
        edge_html = []
        # Process edge tags using the lookup dictionary
        if edge.tags:
            for tag in edge.tags:
                description = self.tag_descriptions.get(tag, _('The tag <i>"%s"</i> is given.') % tag)
                edge_html.append(f"<li>{description}</li>")
        return edge_html

    def get_explanation_for_edge_from(self, edge):
        edge_html = []
        from_dims = edge.from_dimensions
        if from_dims is not None:
            for dim in from_dims:
                dimlabel = self.context.dimensions[dim].label
                cats = [str(cat.label) for cat in from_dims[dim].categories]

                if cats:
                    do = _('exclude') if from_dims[dim].exclude else _('include')
                    edge_html.append(
                        f"<li>{_('From dimension <i>%s</i>, %s categories: <i>%s</i>.') % (dimlabel, do, ', '.join(cats))}</li>"
                    )

                if from_dims[dim].flatten:
                    edge_html.append(f"<li>{_('Sum over dimension <i>%s</i>.') % dimlabel}</li>")
        return edge_html

    def get_explanation_for_edge_to(self, edge):
        edge_html = []
        to_dims = edge.to_dimensions
        if to_dims is not None:
            for dim in to_dims:
                dimlabel = self.context.dimensions[dim].label
                cats = [str(cat.label) for cat in to_dims[dim].categories]

                if cats:
                    cat_str = ', '.join(cats)
                    edge_html.append(
                        f"<li>{_('Categorize the values to <i>%s</i> in a new dimension <i>%s</i>.') % (cat_str, dimlabel)}</li>"
                    )
        return edge_html
