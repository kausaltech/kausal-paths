import pandas as pd
import numpy as np
import pint_pandas

from nodes.exceptions import NodeError

from nodes import NodeMetric
from .context import unit_registry
from params.param import NumberParameter, PercentageParameter, StringParameter, BoolParameter
from params.utils import sep_unit_pt
from .constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN, FORECAST_x, FORECAST_y, VALUE_x, VALUE_y
from .node import Node
from .simple import AdditiveNode, FixedMultiplierNode, SimpleNode
from .ovariable import Ovariable, OvariableFrame
from .actions.energy_saving import BuildingEnergySavingAction


class SelectiveNode(AdditiveNode):
    global_parameters: list[str] = [
        'include_co2', 'include_health', 'include_el_avoided',
    ]

    def compute(self):
        # Global parameters
        include_co2 = self.get_global_parameter_value('include_co2')
        include_health = self.get_global_parameter_value('include_health')
        include_el_avoided = self.get_global_parameter_value('include_el_avoided')

        # Input nodes
        nodes = self.input_nodes
        out = None
        for node in nodes:
            df = node.get_output()
            if 'co2_cost' in node.tags:
                if not include_co2:
                    df[VALUE_COLUMN] *= 0
            if 'capacity_cost' in node.tags:
                if not include_el_avoided:
                    df[VALUE_COLUMN] *= 0
            if 'health_cost' in node.tags:
                if not include_health:
                    df[VALUE_COLUMN] *= 0
            if out is None:
                out = df
            else:
                out[VALUE_COLUMN] += df[VALUE_COLUMN]

        out[VALUE_COLUMN] = self.ensure_output_unit(out[VALUE_COLUMN])
        return out


class ExponentialNode(SimpleNode):
    allowed_parameters = [
        NumberParameter(
            local_id='current_value',
            is_customizable=True,
        ),
        NumberParameter(
            local_id='annual_change',
            is_customizable=True,
        ),
        StringParameter(
            local_id='current_value_name',
            is_customizable=True,
        ),
        StringParameter(
            local_id='annual_change_name',
            is_customizable=True,
        ),
        BoolParameter(
            local_id='decreasing_rate',
            is_customizable=True
        )
    ]

    def compute_exponential(self):
        current_value = self.get_parameter('current_value', required=False)
        if not current_value:  # If the local parameter is not given, use a global parameter
            current_value_name = self.get_parameter_value('current_value_name', required=True)
            current_value = self.context.get_parameter(current_value_name, required=True)
        pt = pint_pandas.PintType(current_value.get_unit())
        annual_change = self.get_parameter('annual_change', required=False)
        if not annual_change:
            annual_change_name = self.get_parameter_value('annual_change_name', required=True)
            annual_change = self.context.get_parameter(annual_change_name, required=True)
        base_unit = annual_change.get_unit()
        current_value = current_value.value
        annual_change = annual_change.value
        base_value = 1 + (annual_change * base_unit).to('dimensionless').m
        decreasing_rate = self.get_parameter_value('decreasing_rate', required=False)
        if decreasing_rate:
            base_value = 1 / base_value
        start_year = self.context.instance.minimum_historical_year
        target_year = self.get_target_year()
        current_year = self.context.instance.maximum_historical_year

        df = pd.DataFrame(
            {VALUE_COLUMN: range(start_year - current_year, target_year - current_year + 1)},
            index=range(start_year, target_year + 1))
        val = current_value * base_value ** df[VALUE_COLUMN]
        df[VALUE_COLUMN] = val.astype(pt)
        df[FORECAST_COLUMN] = df.index > current_year

        return df

    def compute(self):
        return self.compute_exponential()


class EnergyCostNode(AdditiveNode):
    metrics = {
        VALUE_COLUMN: NodeMetric('SEK/kWh', 'currency'),
        'EnergyPrice': NodeMetric('SEK/kWh', 'currency'),
        'AddedValueTax': NodeMetric('SEK/kWh', 'currency'),
        'NetworkPrice': NodeMetric('SEK/kWh', 'currency'),
        'HandlingFee': NodeMetric('SEK/kWh', 'currency'),
        'Certificate': NodeMetric('SEK/kWh', 'currency'),
        'EnergyTax': NodeMetric('SEK/kWh', 'currency')
    }
    global_parameters: list[str] = ['include_energy_taxes']
    allowed_parameters = AdditiveNode.allowed_parameters + [
        NumberParameter(
            local_id='added_value_tax',
            label='Added value tax (%)',
            unit_str='%',
            is_customizable=False
        ),
        NumberParameter(
            local_id='network_price',
            label='Network price (SEK/kWh)',
            unit_str='SEK/kWh',
            is_customizable=False
        ),
        NumberParameter(
            local_id='handling_fee',
            label='Handling fee (SEK/kWh)',
            unit_str='SEK/kWh',
            is_customizable=False
        ),
        NumberParameter(
            local_id='certificate',
            label='Certificate (SEK/kWh)',
            unit_str='SEK/kWh',
            is_customizable=False
        ),
        NumberParameter(
            local_id='energy_tax',
            label='Energy tax (SEK/kWh)',
            unit_str='SEK/kWh',
            is_customizable=False
        ),
    ]

    def compute(self):
        added_value_tax = self.get_parameter_value('added_value_tax', units=True)
        network_price, net_pt = sep_unit_pt(self.get_parameter_value('network_price', units=True))
        handling_fee, han_pt = sep_unit_pt(self.get_parameter_value('handling_fee', units=True))
        certificate, cer_pt = sep_unit_pt(self.get_parameter_value('certificate', units=True))
        energy_tax, ene_pt = sep_unit_pt(self.get_parameter_value('energy_tax', units=True))
        include_energy_taxes = self.get_global_parameter_value('include_energy_taxes')
        print('include energy taxes: ', include_energy_taxes)

        metric = self.get_parameter_value('metric', required=False)
        if self.get_parameter_value('fill_gaps_using_input_dataset', required=False):
            df = self.add_nodes(None, self.input_nodes, metric)
            df = self.fill_gaps_using_input_dataset(df)
        else:
            df = self.add_nodes(None, self.input_nodes, metric)
        df['EnergyPrice'] = df[VALUE_COLUMN]
        self.print_pint_df(df)
        added_value_tax = added_value_tax.to('dimensionless').m
        df['AddedValueTax'] = df['EnergyPrice'] * added_value_tax
        df['NetworkPrice'] = pd.Series(network_price, index=df.index, dtype=net_pt)
        df['HandlingFee'] = pd.Series(handling_fee, index=df.index, dtype=han_pt)
        df['Certificate'] = pd.Series(certificate, index=df.index, dtype=cer_pt)
        df['EnergyTax'] = pd.Series(energy_tax, index=df.index, dtype=ene_pt)

        if include_energy_taxes:
            cols = ['AddedValueTax', 'NetworkPrice', 'HandlingFee', 'Certificate', 'EnergyTax']
        else:
            cols = ['NetworkPrice']
        for col in cols:
            df[VALUE_COLUMN] += df[col]

        return df


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
