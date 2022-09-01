import pandas as pd
import numpy as np

from nodes.exceptions import NodeError

from .context import unit_registry
from params.param import NumberParameter, PercentageParameter, StringParameter
from .constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN, FORECAST_x, FORECAST_y, VALUE_x, VALUE_y
from .node import Node
from .simple import AdditiveNode, FixedMultiplierNode, SimpleNode
from .ovariable import Ovariable, OvariableFrame
from .actions.energy_saving import BuildingEnergySavingAction, BuildingEnergySavingActionb


class CostNode(Ovariable):
    allowed_parameters = [
        NumberParameter(local_id='investment_lifetime'),
        NumberParameter(local_id='investment_cost'),
        NumberParameter(local_id='operation_cost'),
        NumberParameter(local_id='investment_years'),
        NumberParameter(local_id='investment_numbers'),
    ] + Ovariable.allowed_parameters

    quantity = 'currency'

    def compute(self):
        costs = self.get_input('currency')
        investment_cost = self.get_parameter_value('investment_cost') * self.get_parameter('investment_cost').unit
        operation_cost = self.get_parameter_value('operation_cost') * self.get_parameter('operation_cost').unit
        investment_lifetime = self.get_parameter_value('investment_lifetime')
        investment_years = self.get_parameter_value('investment_years')
        investment_numbers = self.get_parameter_value('investment_numbers') * self.get_parameter(
            'investment_numbers').unit

        discount_rate = self.context.get_parameter_value('discount_rate')
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
                discount = discount * (1 + discount_rate)  # FIXME Discounting should NOT be shown on graph
            costs.at[time, VALUE_COLUMN] = (change + costs.at[time, VALUE_COLUMN]) / discount

        return(costs)


class SocialCost(SimpleNode):

    def compute(self):
        def net_present_value(discount_rate, timespan, lifetime=None):
            if lifetime is None:
                lifetime = 1
                unit = unit_registry('1 a')
            else:
                assert {str(lifetime.units)} <= {'year', 'a'}
                lifetime = round(lifetime.m)
                unit = 1
            out = 0
            for i in range(timespan):
                if (i % lifetime) == 0:
                    out += (1 / (1 + discount_rate)) ** i
            return out * unit

        # Global parameters
        discount_rate = self.context.get_parameter_value_w_unit('discount_rate')
        health_impacts_per_kwh = self.context.get_parameter_value_w_unit('health_impacts_per_kwh')
        avoided_electricity_capacity_price = self.context.get_parameter_value_w_unit('avoided_electricity_capacity_price')
        heat_co2_ef = self.context.get_parameter_value_w_unit('heat_co2_ef')
        electricity_co2_ef = self.context.get_parameter_value_w_unit('electricity_co2_ef')
        cost_co2 = self.context.get_parameter_value_w_unit('cost_co2')
        carbon_price_change = self.context.get_parameter_value_w_unit('carbon_price_change')
        target_year = self.get_target_year()

        # Input nodes
        df = self.get_input_node(tag='floor_area').get_output()
        he_price = self.get_input_node(tag='price_of_heat').get_output()
        el_price = self.get_input_node(tag='price_of_electricity').get_output()
        df['HePrice'] = he_price[VALUE_COLUMN]
        df['ElPrice'] = el_price[VALUE_COLUMN]
        df = df.rename(columns={VALUE_COLUMN: 'FloorArea'})

        last_hist_year = df.loc[~df[FORECAST_COLUMN]].index.max()
        timespan = target_year - last_hist_year
        npv = net_present_value(discount_rate, timespan)
        out = None

        for node in self.input_nodes:
            if not isinstance(node, BuildingEnergySavingActionb):
                continue
            else:
                heat = node.get_output(dimension='HeSaving')[VALUE_COLUMN]
                electricity = node.get_output(dimension='ElSaving')[VALUE_COLUMN]
                renov_cost = node.get_output(dimension='RenovCost')[VALUE_COLUMN]
                renovation = node.get_output(dimension=VALUE_COLUMN)[VALUE_COLUMN]

            df['CostSaving'] = (
                df['ElPrice'] * electricity
                + df['HePrice'] * heat) * npv
            df['PrivateProfit'] = (df['CostSaving'] - renov_cost)
            df['ElAvoided'] = electricity * avoided_electricity_capacity_price
            df['CO2Saved'] = (
                (heat * heat_co2_ef
                + electricity * electricity_co2_ef) * cost_co2
                ).astype('pint[EUR/a/m**2]')
            df['EnSaving'] = heat + electricity
            df['Health'] = df['EnSaving'] * health_impacts_per_kwh
            df['SocialProfit'] = (
                df['ElAvoided'] 
                + df['CO2Saved'] 
                + df['Health']
                ) * npv + df['PrivateProfit']
            potential_area = df['FloorArea'] * renovation
            df[VALUE_COLUMN] = df['SocialProfit'] * potential_area * npv  # FIXME See Erik's email 2022-08-29 about npv
            if out is None:
                out = df[[VALUE_COLUMN, FORECAST_COLUMN]].copy()
            else:
                out[VALUE_COLUMN] += df[VALUE_COLUMN]

        out[VALUE_COLUMN] = self.ensure_output_unit(out[VALUE_COLUMN])
        return out


class EnergyConsumption(SimpleNode):

    def compute(self):
        # Input nodes
        df = self.get_input_node(tag='floor_area').get_output()
        out = df[[VALUE_COLUMN, FORECAST_COLUMN]]
        first = True

        for node in self.input_nodes:
            if not isinstance(node, BuildingEnergySavingActionb):
                continue
            else:
                heat = node.get_output(dimension='HeSaving')
                electricity = node.get_output(dimension='ElSaving')
                renovation = node.get_output(dimension=VALUE_COLUMN)

                energy = (heat[VALUE_COLUMN] + electricity[VALUE_COLUMN])
                energy = energy * df[VALUE_COLUMN] * renovation[VALUE_COLUMN] * unit_registry('-1 a')
            if first:
                out[VALUE_COLUMN] = energy
                first = False
            else:
                out[VALUE_COLUMN] += energy
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
