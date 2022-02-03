import pandas as pd
import numpy as np
from .context import unit_registry
from params.param import NumberParameter, PercentageParameter, StringParameter
from .constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN, FORECAST_x, FORECAST_y, VALUE_x, VALUE_y
from .simple import AdditiveNode, FixedMultiplierNode, SimpleNode
from .ovariable import Ovariable, OvariableFrame

DISCOUNT_RATE = 0.03


class CostNode(AdditiveNode):
    allowed_parameters = [
        NumberParameter(local_id='investment_lifetime'),
        NumberParameter(local_id='investment_cost'),
        NumberParameter(local_id='investment_year'),
        NumberParameter(local_id='operation_cost'),
        NumberParameter(local_id='operation_year_start'),
        NumberParameter(local_id='operation_year_end'),
    ] + Ovariable.allowed_parameters

    quantity = 'monetary_amount'

    def compute(self):
        df = pd.DataFrame({
            # YEAR_COLUMN
            YEAR_COLUMN: [
                2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026,
                2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 3035
            ],
            FORECAST_COLUMN: [True] * 19,
            VALUE_COLUMN: [unit_registry('0 EUR')] * 19
        }).set_index(YEAR_COLUMN)
        addition = self.get_parameter_value('investment_cost') * self.get_parameter('investment_cost').unit
        investment_year = self.get_parameter_value('investment_year')
        df.at[investment_year, VALUE_COLUMN] = addition + df.at[investment_year, VALUE_COLUMN]
        self.print_pint_df(df)
        print(self.unit)
        df[VALUE_COLUMN] = self.ensure_output_unit(df[VALUE_COLUMN])

        return(df)
