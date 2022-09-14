import pandas as pd
import numpy as np

from nodes.exceptions import NodeError

from .context import unit_registry
from params.param import NumberParameter, PercentageParameter, StringParameter
from .constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN, FORECAST_x, FORECAST_y, VALUE_x, VALUE_y
from .node import Node
from .simple import AdditiveNode, FixedMultiplierNode, SimpleNode
from .ovariable import Ovariable, OvariableFrame
from .actions.energy_saving import BuildingEnergySavingAction


class SocialCost(SimpleNode):

    def compute(self):

        # Global parameters
        health_impacts_per_kwh = self.context.get_parameter_value_w_unit('health_impacts_per_kwh')
        avoided_electricity_capacity_price = self.context.get_parameter_value_w_unit('avoided_electricity_capacity_price')
        heat_co2_ef = self.context.get_parameter_value_w_unit('heat_co2_ef')
        electricity_co2_ef = self.context.get_parameter_value_w_unit('electricity_co2_ef')
        include_co2 = self.context.get_parameter_value('include_co2')
        include_health = self.context.get_parameter_value('include_health')
        include_el_avoided = self.context.get_parameter_value('include_el_avoided')

        # Input nodes
        df = self.get_input_node(tag='floor_area').get_output()
        cost_co2 = self.get_input_node(tag='cost_co2').get_output()
        he_price = self.get_input_node(tag='price_of_heat').get_output()
        el_price = self.get_input_node(tag='price_of_electricity').get_output()
        df['HePrice'] = he_price[VALUE_COLUMN]
        df['ElPrice'] = el_price[VALUE_COLUMN]
        df = df.rename(columns={VALUE_COLUMN: 'FloorArea'})

        out = None

        for node in self.input_nodes:
            if not isinstance(node, BuildingEnergySavingAction):
                continue
            else:
                heat = node.get_output(dimension='Heat')[VALUE_COLUMN]
                electricity = node.get_output(dimension='Electricity')[VALUE_COLUMN]
                renov_cost = node.get_output(dimension='RenovCost')[VALUE_COLUMN]

            df['CostSaving'] = (
                df['ElPrice'] * electricity
                + df['HePrice'] * heat)
            df['PrivateProfit'] = (df['CostSaving'] - renov_cost)
            df['ElAvoided'] = electricity * avoided_electricity_capacity_price
            df['CO2Saved'] = (
                (heat * heat_co2_ef
                + electricity * electricity_co2_ef) * cost_co2[VALUE_COLUMN]
                ).astype('pint[SEK/a/m**2]')
            df['EnSaving'] = heat + electricity
            df['Health'] = df['EnSaving'] * health_impacts_per_kwh
            s = df['PrivateProfit']
            if include_el_avoided:
                s += df['ElAvoided']
            if include_co2:
                s += df['CO2Saved']
            if include_health:
                s += df['Health']
            df['SocialProfit'] = s
            df[VALUE_COLUMN] = df['SocialProfit'] * df['FloorArea']
            if out is None:
                out = df[[VALUE_COLUMN, FORECAST_COLUMN]].copy()
            else:
                out[VALUE_COLUMN] += df[VALUE_COLUMN]
        out[VALUE_COLUMN] = self.ensure_output_unit(out[VALUE_COLUMN])
        return out


class SocialCostb(SimpleNode):

    def compute(self):

        # Global parameters
        include_co2 = self.context.get_parameter_value('include_co2')
        include_health = self.context.get_parameter_value('include_health')
        include_el_avoided = self.context.get_parameter_value('include_el_avoided')

        # Input nodes
        df = self.get_input_node(tag='renovation').get_output()

        health_heat = self.get_input_node(tag='health_heat').get_output()[VALUE_COLUMN]
        co2_heat = self.get_input_node(tag='co2_heat').get_output()[VALUE_COLUMN]
        heat = self.get_input_node(tag='heat').get_output()[VALUE_COLUMN]
        avoided_electricity = self.get_input_node(tag='avoided_electricity').get_output()[VALUE_COLUMN]
        health_electricity = self.get_input_node(tag='health_electricity').get_output()[VALUE_COLUMN]
        co2_electricity = self.get_input_node(tag='co2_electricity').get_output()[VALUE_COLUMN]
        electricity = self.get_input_node(tag='electricity').get_output()[VALUE_COLUMN]

        s = df[VALUE_COLUMN] + heat + electricity
        if include_health:
            s = s + health_heat + health_electricity
        if include_co2:
            s = s + co2_heat + co2_electricity
        if include_el_avoided:
            s = s + avoided_electricity
        df[VALUE_COLUMN] = s
        df[VALUE_COLUMN] = self.ensure_output_unit(df[VALUE_COLUMN])
        return df


class DiscountNode(SimpleNode):
    '''All input nodes must be additive'''
    def get_discount_factor(self, base_value):
        target_year = self.context.target_year
        start_year = self.context.instance.minimum_historical_year
        current_time = self.context.instance.maximum_historical_year - start_year
        duration = target_year - start_year + 1
        year = []
        forecast = []
        factor = [1]

        for i in range(duration):
            if i > current_time:
                factor = factor + [factor[-1] * base_value]
                forecast = forecast + [True]
            else:
                factor = factor + [factor[-1]]
                forecast = forecast + [False]
            year = year + [start_year + i]

        df = pd.DataFrame({
            YEAR_COLUMN: year,
            VALUE_COLUMN: pd.Series(factor[1:]),
            FORECAST_COLUMN: forecast}).set_index([YEAR_COLUMN])
        return df

    def compute(self):
        base_value = self.context.get_parameter_value_w_unit('discount_rate')
        assert {str(base_value.units)} <= {'%/a', '%/year', '%'}
        base_value = 1 / (1 + base_value).m
        discount_factor = self.get_discount_factor(base_value)

        df = None
        for node in self.input_nodes:
            if df is None:
                df = node.get_output()
            else:
                df[VALUE_COLUMN] += node.get_output()[VALUE_COLUMN]
                df[FORECAST_COLUMN] = df[FORECAST_COLUMN] | node[FORECAST_COLUMN]
        df[VALUE_COLUMN] *= discount_factor[VALUE_COLUMN]
        return df


class EnergyConsumption(SimpleNode):

    def compute(self):
        # Input nodes
        floor = self.get_input_node(tag='floor_area').get_output()
        out = floor[[VALUE_COLUMN, FORECAST_COLUMN]]
        floor = floor[VALUE_COLUMN]
        first = True

        for node in self.input_nodes:
            if not isinstance(node, BuildingEnergySavingAction):
                continue
            else:
                heat = node.get_output(dimension='Heat')[VALUE_COLUMN]
                electricity = node.get_output(dimension='Electricity')[VALUE_COLUMN]

                energy = (heat + electricity) * floor
            if first:
                out[VALUE_COLUMN] = energy
                first = False
            else:
                out[VALUE_COLUMN] += energy
        out[VALUE_COLUMN] = self.ensure_output_unit(out[VALUE_COLUMN])

        return out


class AddUsingDimensionNode(SimpleNode):
    allowed_parameters = [
        StringParameter(
            local_id='dimension',
            is_customizable=False,
        ),
    ]

    def compute(self):
        dimension = self.get_parameter_value('dimension')
        first = True
        out = pd.DataFrame()

        for node in self.input_nodes:
            df = node.get_output(dimension=dimension)
            if first:
                out = df
                first = False
            else:
                out[VALUE_COLUMN] += df[VALUE_COLUMN]
        out[VALUE_COLUMN] = self.ensure_output_unit(out[VALUE_COLUMN])

        return out


# Grön logik and marginal abatement cost (MAC) curves, notes
# https://data-88e.github.io/textbook/content/12-environmental/textbook1.html
# https://plotly.com/python/bar-charts/

# Codes from file Beräkningar 24 år.xlsx
# Bestånd: småhus, flerbostadshus, kontor, skolor, småhus utan sol

# Constants:
# Q37 Antal m^2
# Q39 Elpris (SEK/kWh)
# Q40 Värmepris
# Q41 Diskonteringsränta
# Q42 CO2-kostnad (SEK/mtCO2)
# Q43 Hälsovinster (kr/kwh): 15×1.36÷277.78 Danska Grön Logik (15 DKR/GJ)
# Q44 Tidshorisont
# Q47 Utrullningstakt Referensalternativ?
# NEPP:
# N68 Ökning med 50 TWH? Till 190, från?: data
# N69 Kostar mellan 560 och 640 mdr: data
# N70 kostnad per twh: N69/N68
# N71 per kwh: N70/1e9
# N72 utspritt över åren (2021-2050, 30 år): N71/30
# Men denna investering är ju inte återkommande varje år (som mina beräkningar är just nu…)
# https://www.nepp.se/pdf/Det_kr%C3%A4vs_stora_investeringar.pdf
# T36 Värme -> CO2 g/kWh (CO2e): data
# T37 El -> CO2	g/kWh (CO2e): data
# https://www.energiforetagen.se/energifakta/miljo-och-klimat/elens-miljopaverkan/vaxthuseffekten/
# https://smed.se/luft-och-klimat/4708?fbclid=IwAR1mhwqqEHH4h2NuRr8P7KlENuPxWRLmYAMeQ3r1fTmgPOhTRw0Cdh2UJJ0
# https://www.energiforetagen.se/energifakta/miljo-och-klimat/fjarrvarmens-miljopaverkan/fjarrvarmens-miljonytta/
# Boverkets klimatdatabas!

# A Kod: data
# B Åtgärd (Rimlig tabell att utgå ifrån i BeSmå Energieffektiviseringspotential Småhus): data
# C Livslängd: data
# Per m^2
# D Investerings-kostnad (kr/m2): data
# E Energi-besparing (kWh/m2/år): F+G
# F Värme-besparing: data
# G Elbesparing: data
# H NPV Investeringskostnad: D×(1−Q$41)^0+D×(1−Q$41)^15  # add a new monome for each investment year within tidshorisont
# I Kostnads-besparing: G×Q$39+F×Q$40
# J NPV Kostnads-besparing: I×(1−(1÷(1+Q$41))^Q$44)÷(1−(1÷(1+Q$41)))
# Proof: If you denote a = 1/(1+r) where r is discount rate, you can solve
# sum a^n, n=0 to k = (a^(k+1)-1)/(a-1)
# https://www.wolframalpha.com/input?i=sum+a%5En%2C+n%3D0+to+k
# (You can check the formula by polynomial division, and you get a^k+a^(k-1)+...+a+1).
# This is equal to the excel formulation when tidshorisont = k+1 and you multiply both numerator and denominator by -1.
# When you start from 0 (now, no discounting) and go on to k, you count k+1 years in total, which is tidshorisont.

# K Privat-ekonomisk vinst: -H+J
# L Marginalnetto-kostnad för energibesparing, privat: -K/E
# M Kostnads-effektivitet, privat: (K+D)/D
# N MB (marginal benefit): Undvikt elutbyggnad: G×N$72
# O MB: Minskade CO2-utsläpp: (F×T$36+G×T$37)÷1000000×Q$42
# P MB: Hälsovinster inomhusklimat: E*Q$43
# Q NPV MB: (N+O+P)×(1−(1÷(1+Q$41))^Q$44)÷(1−(1÷(1+Q$41)))
# R Samhälls-ekonomisk vinst: K+Q
# S Marginalnetto-kostnad för energibesparing, samhälle: -R/E
# Total
# T Potential av småhus: data
# U Utrullningstakt: 1/C
# V Potential, antal m2: T*Q$37
# W Privat-ekonomisk vinst: V×K×(U−Q$47)×(1−(1÷(1+Q$41))^Q$44)÷(1−(1÷(1+Q$41)))
# X Samhälls-ekonomisk vinst: V×R×(U−Q$47)×(1−(1÷(1+Q$41))^Q$44)÷(1−(1÷(1+Q$41)))
# Y Total energibesparing, kWh/år: E*V*U
# Z Värmebesparing, årlig: F*V*U
# AA Elbesparing, årlig: G*V*U
# AB Energibesparing vid T: Y×MIN(C,Q$44)
# AC Värmebesparing vid T: Z×MIN(C,Q$44)
# AD Elbesparing vid T: AA×MIN(C,Q$44)

# MAC curve plots
# legend: B
# X axis: cumulative of Y over B when ordered by S
# Y axis: S

# Comments:
# Investeringskostnad, Energibesparing: Här har jag använt bedömd merkostnad från HEFTIG (enl. motivation i den
# rapporten). Alltså: Jämfört med vad som annars hade gjorts. Detta inkluderar också moms!!
# Hälsovinster inomhusklimat: Använt danska rapporten, oklart varifrån 15 dkr kommer.
# Flerbostadshus T17:T18 Grov uppskattning, för att slippa fördela fastigheter efter byggår.
