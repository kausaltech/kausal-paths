import pandas as pd
import numpy as np
from .context import unit_registry
from params.param import NumberParameter, PercentageParameter, StringParameter
from .constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN, FORECAST_x, FORECAST_y, VALUE_x, VALUE_y
from .simple import AdditiveNode, FixedMultiplierNode, SimpleNode
from .ovariable import Ovariable, OvariableFrame

DISCOUNT_RATE = 0.03


class CostNode(Ovariable):
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
        costs = self.get_input('monetary_amount')
        investment_cost = self.get_parameter_value('investment_cost') * self.get_parameter('investment_cost').unit
        investment_year = self.get_parameter_value('investment_year')
        operation_cost = self.get_parameter_value('operation_cost') * self.get_parameter('operation_cost').unit
        operation_year_start = self.get_parameter_value('operation_year_start')
        operation_year_end = self.get_parameter_value('operation_year_end')
        print(investment_year)

        discount = 1

        for i in costs.reset_index()[YEAR_COLUMN]:
            print(i)
            change = unit_registry('0 kEUR')
            if i == investment_year:
                change = change + investment_cost
            if i >= operation_year_start and i <= operation_year_end:
                change = change + operation_cost
            print(change)
            if costs.at[i, FORECAST_COLUMN]:
                discount = discount * (1 + DISCOUNT_RATE)
            costs.at[i, VALUE_COLUMN] = (change + costs.at[i, VALUE_COLUMN]) / discount



        self.print_pint_df(costs)
        print(self.unit)
        #  df[VALUE_COLUMN] = self.ensure_output_unit(df[VALUE_COLUMN])

        return(costs)
