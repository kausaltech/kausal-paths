import pandas as pd
import numpy as np
from .context import unit_registry
from params.param import NumberParameter, PercentageParameter, StringParameter
from .constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN, FORECAST_x, FORECAST_y, VALUE_x, VALUE_y
from .simple import AdditiveNode, FixedMultiplierNode, SimpleNode
from .ovariable import Ovariable, OvariableFrame

DISCOUNT_RATE = 0.04


class CostNode(Ovariable):
    allowed_parameters = [
        NumberParameter(local_id='investment_lifetime'),
        NumberParameter(local_id='investment_cost'),
        NumberParameter(local_id='operation_cost'),
        NumberParameter(local_id='investment_years'),
        NumberParameter(local_id='investment_numbers'),
    ] + Ovariable.allowed_parameters

    quantity = 'monetary_amount'

    def compute(self):
        costs = self.get_input('monetary_amount')
        investment_cost = self.get_parameter_value('investment_cost') * self.get_parameter('investment_cost').unit
        operation_cost = self.get_parameter_value('operation_cost') * self.get_parameter('operation_cost').unit
        investment_lifetime = self.get_parameter_value('investment_lifetime')
        investment_years = self.get_parameter_value('investment_years')
        investment_numbers = self.get_parameter_value('investment_numbers') * self.get_parameter('investment_numbers').unit

        discount = 1

        for time in costs.reset_index()[YEAR_COLUMN]:
            change = unit_registry('0 kEUR')
            for j in range(len(investment_years)):
                assert len(investment_years) == len(investment_numbers)
                year = investment_years[j]
                number = investment_numbers[j]
                if time == year:
                    change = change + (investment_cost * number)
                if time >= year and time < year + investment_lifetime:
                    change = change + (operation_cost * number)
            if costs.at[time, FORECAST_COLUMN]:
                discount = discount * (1 + DISCOUNT_RATE)  # FIXME Discounting should NOT be shown on graph
            costs.at[time, VALUE_COLUMN] = (change + costs.at[time, VALUE_COLUMN]) / discount

        return(costs)
