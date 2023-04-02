from nodes.calc import extend_last_historical_value_pl, nafill_all_forecast_years
from params.param import Parameter, BoolParameter, NumberParameter, ParameterWithUnit, StringParameter
from typing import List, ClassVar, Tuple
import polars as pl
import pandas as pd
import pint

from common.i18n import TranslatedString
from common import polars as ppl
from .simple import AdditiveAction


class PriorityNode(AdditiveAction):
    allowed_parameters: ClassVar[List[Parameter]] = [
        StringParameter(
            local_id='endorsers',
            label=TranslatedString(en="Identifiers for people that endorse this priority"),  # FIXME Should be a list
            is_customizable=True
        ),
        StringParameter(
            local_id='action_id',
            label=TranslatedString(en="Action to be prioritized"),
            is_customizable=True
        ),
        BoolParameter(
            local_id='priority',
            label=TranslatedString(en='Should the action be implemented?'),
            is_customizable=True
        )
    ]
