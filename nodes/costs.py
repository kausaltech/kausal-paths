import functools

import pandas as pd
import polars as pl
import pint_pandas

from nodes import NodeMetric, Node
from nodes.units import Quantity
import common.polars as ppl
from params.param import NumberParameter, StringParameter, BoolParameter
from params.utils import sep_unit, sep_unit_pt
from .constants import DEFAULT_METRIC, FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from .simple import AdditiveNode, SimpleNode


class SelectiveNode(AdditiveNode):
    global_parameters: list[str] = [
        'include_co2', 'include_health', 'include_el_avoided',
    ]

    def compute(self) -> ppl.PathsDataFrame:
        # Global parameters
        include_co2 = self.get_global_parameter_value('include_co2')
        include_health = self.get_global_parameter_value('include_health')
        include_el_avoided = self.get_global_parameter_value('include_el_avoided')

        # Input nodes
        nodes = self.input_nodes
        out = None
        included_nodes: list[Node] = []

        for node in nodes:
            df = node.get_output_pl()
            if 'co2_cost' in node.tags:
                if include_co2:
                    included_nodes.append(node)
            elif 'capacity_cost' in node.tags:
                if include_el_avoided:
                    included_nodes.append(node)
            elif 'health_cost' in node.tags:
                if include_health:
                    included_nodes.append(node)
            else:
                included_nodes.append(node)

        assert len(included_nodes)
        output_unit = self.output_metrics[DEFAULT_METRIC].unit
        for node in included_nodes:
            df = node.get_output_pl().ensure_unit(VALUE_COLUMN, output_unit)
            if out is None:
                out = df
            else:
                out = out.paths.add_with_dims(df, how='outer')
        assert out is not None
        return out


def compute_exponential(
    start_year: int, current_year: int, model_end_year: int, current_value: Quantity | float, annual_change: Quantity,
    decreasing_rate: bool
):
    base_value = 1 + annual_change.to('dimensionless').m
    if decreasing_rate:
        base_value = 1 / base_value

    values = range(start_year - current_year, model_end_year - current_year + 1)
    years = range(start_year, model_end_year + 1)
    df = pl.DataFrame({YEAR_COLUMN: years, 'nr': values})
    df = df.with_columns([
        (pl.lit(base_value) ** pl.col('nr')).alias('mult')
    ])
    val = current_value.m if isinstance(current_value, Quantity) else current_value
    df = df.with_columns([
        (pl.lit(val) * pl.col('mult')).alias(VALUE_COLUMN),
        (pl.col(YEAR_COLUMN) > current_year).alias(FORECAST_COLUMN)
    ])
    return df.select([YEAR_COLUMN, VALUE_COLUMN, FORECAST_COLUMN])


class ExponentialNode(AdditiveNode):
    '''
    Takes in either input nodes as AdditiveNode, or builds a dataframe from current_value.
    Builds an exponential multiplier based on annual_change and multiplies the VALUE_COLUMN.
    Optionally, touches also historical values.
    Parameter is_decreasing_rate is used to give discount rates instead.
    '''
    allowed_parameters = AdditiveNode.allowed_parameters + [
        NumberParameter(
            local_id='current_value',
            is_customizable=True,
        ),
        NumberParameter(
            local_id='annual_change',
            is_customizable=True,
        ),
        BoolParameter(
            local_id='is_decreasing_rate',
            is_customizable=True,
            value=False
        ),
        BoolParameter(
            local_id='touch_historical_values',
            is_customizable=True,
            value=False
        )
    ]

    def compute(self):
        current_value = self.get_parameter_value('current_value', required=False, units=True)

        if current_value is not None:
            if len(self.input_nodes) > 0:
                raise Exception('You must give either input node(s) or parameter current_value but not both.')

            unit = current_value.units
            current_value = current_value.m
            start_year = self.context.instance.minimum_historical_year
            current_year = self.context.instance.maximum_historical_year
            end_year = self.context.instance.model_end_year

            df = pl.DataFrame({
                YEAR_COLUMN: range(start_year, end_year)
            })
            df = df.with_columns([
                pl.when(pl.col(YEAR_COLUMN) > pl.lit(current_year))
                .then(pl.lit(True)).otherwise(pl.lit(False)).alias(FORECAST_COLUMN),
                pl.lit(current_value).alias(VALUE_COLUMN)
            ])
            meta = ppl.DataFrameMeta(units={VALUE_COLUMN: unit}, primary_keys=[YEAR_COLUMN])
            df = ppl.to_ppdf(df, meta=meta)

        else:
            df = super().compute()
            current_year = df.filter(~pl.col(FORECAST_COLUMN))[YEAR_COLUMN].max()

        annual_change = self.get_parameter_value('annual_change', required=True, units=True)
        base_value = 1 + annual_change.to('dimensionless').m

        if self.get_parameter_value('is_decreasing_rate', required=False):
            base_value = 1 / base_value
        
        df = df.with_columns(
            (pl.col(YEAR_COLUMN) - pl.lit(current_year)).alias('power')
        )
        if not self.get_parameter_value('touch_historical_values', required=False):
            df = df.with_columns(
                pl.when(pl.col('power') < pl.lit(0)).then(pl.lit(0))
                .otherwise(pl.col('power')).alias('power')
            )

        df = df.with_columns(
            (pl.lit(base_value) ** pl.col('power') * pl.col(VALUE_COLUMN)).alias(VALUE_COLUMN)
        ).drop('power')

        return df


class EnergyCostNode(AdditiveNode):
    output_metrics = {
        VALUE_COLUMN: NodeMetric('SEK/kWh', 'currency'),
        #'EnergyPrice': NodeMetric('SEK/kWh', 'currency'),
        #'AddedValueTax': NodeMetric('SEK/kWh', 'currency'),
        #'NetworkPrice': NodeMetric('SEK/kWh', 'currency'),
        #'HandlingFee': NodeMetric('SEK/kWh', 'currency'),
        #'Certificate': NodeMetric('SEK/kWh', 'currency'),
        #'EnergyTax': NodeMetric('SEK/kWh', 'currency')
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
        added_value_tax = self.get_parameter_value('added_value_tax', units=True).to('dimensionless').m
        output_unit = self.output_metrics[VALUE_COLUMN].unit
        network_price, net_pt = sep_unit(self.get_parameter_value('network_price', units=True), output_unit)
        handling_fee, han_pt = sep_unit(self.get_parameter_value('handling_fee', units=True), output_unit)
        certificate, cer_pt = sep_unit(self.get_parameter_value('certificate', units=True), output_unit)
        energy_tax, ene_pt = sep_unit(self.get_parameter_value('energy_tax', units=True), output_unit)
        include_energy_taxes = self.get_global_parameter_value('include_energy_taxes')

        metric = self.get_parameter_value('metric', required=False)
        if self.get_parameter_value('fill_gaps_using_input_dataset', required=False):
            df = self.add_nodes_pl(None, self.input_nodes, metric)
            df = self.fill_gaps_using_input_dataset_pl(df)
        else:
            df = self.add_nodes_pl(None, self.input_nodes, metric)

        meta = df.get_meta()
        df = df.with_columns([
            pl.col(VALUE_COLUMN).alias('EnergyPrice'),
            (pl.col(VALUE_COLUMN) * added_value_tax).alias('AddedValueTax'),
            pl.lit(network_price).alias('NetworkPrice'),
            pl.lit(handling_fee).alias('HandlingFee'),
            pl.lit(certificate).alias('Certificate'),
            pl.lit(energy_tax).alias('EnergyTax'),
        ])
        meta.units.update(dict(
            EnergyPrice=meta.units[VALUE_COLUMN],
            AddedValueTax=meta.units[VALUE_COLUMN],
            NetworkPrice=net_pt,
            HandlingFee=han_pt,
            Certificate=cer_pt,
            EnergyTax=ene_pt,
        ))
        df = ppl.to_ppdf(df=df, meta=meta)

        if include_energy_taxes:
            cols = ['AddedValueTax', 'NetworkPrice', 'HandlingFee', 'Certificate', 'EnergyTax']
        else:
            cols = ['NetworkPrice']

        add_expr = functools.reduce(lambda x, y: x + y, [pl.col(x) for x in cols])
        df = df.with_columns([
            (pl.col(VALUE_COLUMN) + add_expr).alias(VALUE_COLUMN)
        ])
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
