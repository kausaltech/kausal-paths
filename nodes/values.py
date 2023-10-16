from nodes.calc import extend_last_historical_value_pl, nafill_all_forecast_years
from params.param import Parameter, BoolParameter, NumberParameter, ParameterWithUnit, StringParameter
from typing import List, ClassVar, Tuple
import polars as pl
import pandas as pd
import pint

from common.i18n import TranslatedString
from common import polars as ppl
from .constants import FORECAST_COLUMN, NODE_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from .node import Node
from .simple import SimpleNode, AdditiveNode
from .actions.simple import AdditiveAction
from .exceptions import NodeError


