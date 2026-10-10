"""Prepare auditable BISKO activation sources for a pinned DVC repository."""

import math
from collections import defaultdict
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping


POPULATION_DATASET = 'kommune/bevoelkerung'
POPULATION_SOURCE = 'bisko/defaults/population'
ENERGY_DATASET = 'kommune/endenergieverbrauch'
ENERGY_SOURCE = 'bisko/defaults/stationary_energy'
DEFAULT_VARIANTS = {'03': 'trend', '07': 'projection', '12': 'middle'}
SECTORS = ('private_households', 'commerce_trade_services', 'industry', 'municipal_facilities')


class DefaultProvenance(BaseModel):
    """Derivation retained with each provider value, including copied defaults."""

    source_dataset: str
    source_edition: str | None = None
    source_vintage: str | None = None
    source_url: str | None = None
    source_sha256: str
    method: str
    variant: str | None = None
    source_region_code: str | None = None
    allocation_year: int | None = None
    allocation_share: float | None = None
    source_year: int | None = None
    is_estimate: bool = False
    prior_sha256: str | None = None
    upstream_method: str | None = None
    empirical_lower_mwh: float | None = None
    empirical_upper_mwh: float | None = None
    source_kind: str | None = None
    source_status: str | None = None
    source_location_code: str | None = None
    normalizer: str | None = None
    normalizer_value: float | None = None
    coefficient_mwh_per_unit: float | None = None
    input_sha256: str | None = None

    def describe(self, source_revision: str) -> str:
        """Explain the derivation in citations; retain machine-readable details in dataset metadata."""
        method = self.method.split(':')[-1]
        descriptions = {
            'observed': 'Official population observation.',
            'published': 'Published population projection.',
            'linear_interpolation': 'Linear interpolation between published years.',
            'hold_last': 'The last supplied value is held constant.',
            'grid_delivery': 'Grid electricity deliveries, supplied as a provisional provider default.',
            'mainz_2018_carrier_split': 'Carrier estimate using the sector proportions of the Mainz 2018 inventory.',
        }
        parts = [descriptions.get(method, 'Provider estimate.')]
        if self.allocation_year is not None and self.allocation_share is not None:
            parts.append(
                f"Allocated from the association projection using the municipality's "
                f'{self.allocation_share:.2%} population share in {self.allocation_year}.'
            )
        if self.variant is not None:
            parts.append(f'Projection variant: {self.variant}.')
        if self.source_year is not None:
            parts.append(f'Original source year: {self.source_year}.')
        parts.append(f'Provider dataset revision: {source_revision}.')
        return ' '.join(parts)


class PopulationDefault(BaseModel):
    ags: str = Field(pattern=r'^\d{8}$')
    year: int = Field(ge=2000)
    value: int = Field(ge=0)
    is_forecast: bool
    provenance: DefaultProvenance


class EnergyDefault(BaseModel):
    ags: str = Field(pattern=r'^\d{8}$')
    year: int = Field(ge=2000)
    sector: str
    energy_carrier: str
    value: float = Field(ge=0, allow_inf_nan=False)
    provenance: DefaultProvenance


class StationaryEstimate(BaseModel):
    ags: str = Field(pattern=r'^\d{8}$')
    year: int = Field(ge=2000)
    sector: Literal['private_households', 'commerce_trade_services', 'industry', 'municipal_facilities']
    quantity: Literal['stationary_final_energy', 'grid_electricity']
    value_mwh_per_year: float = Field(ge=0, allow_inf_nan=False)
    needs_review: bool
    source_year: int | None
    source_sha256: str | None
    method: str
    empirical_lower_mwh: float | None
    empirical_upper_mwh: float | None
    source_kind: str | None = None
    source_status: str | None = None
    source_location_code: str | None = None
    normalizer: str | None = None
    normalizer_value: float | None = None
    coefficient_mwh_per_unit: float | None = None


def _annual_value(values: Mapping[int, int], year: int) -> tuple[int, str]:
    if year in values:
        return values[year], 'published'
    earlier = max((y for y in values if y < year), default=None)
    later = min((y for y in values if y > year), default=None)
    if earlier is None:
        raise ValueError(f'Population forecast begins after {year}.')
    if later is None:
        return values[earlier], 'hold_last'
    value = Fraction(values[earlier]) + Fraction(year - earlier, later - earlier) * (values[later] - values[earlier])
    return round(value), 'linear_interpolation'


def _allocate(total: int, weights: Mapping[str, int]) -> dict[str, int]:
    """Largest remainder allocation preserves the aggregate, including rounding."""
    weight = sum(weights.values())
    if weight <= 0:
        raise ValueError('Forecast allocation needs positive observed population.')
    shares = {ags: Fraction(total * value, weight) for ags, value in weights.items()}
    result = {ags: math.floor(value) for ags, value in shares.items()}
    order = sorted(shares, key=lambda ags: (-(shares[ags] - result[ags]), ags))
    for ags in order[: total - sum(result.values())]:
        result[ags] += 1
    return result


def prepare_population_defaults(  # noqa: C901, PLR0912, PLR0915
    history: pl.DataFrame,
    forecasts: pl.DataFrame,
    geography: pl.DataFrame,
    *,
    historical_sha256: str,
    horizon: int = 2045,
) -> list[PopulationDefault]:
    """Use observed history, default Land variants and fixed latest-observed member shares."""
    parents = dict(geography.select('ars', 'parent_ars').iter_rows())
    municipalities = dict(geography.filter(pl.col('ags').is_not_null()).select('ags', 'ars').iter_rows())
    histories: dict[str, dict[int, int]] = defaultdict(dict)
    history_urls: dict[tuple[str, int], str] = {}
    for ags, year, population, url in history.select('ags', 'year', 'population', 'source_url').iter_rows():
        if ags not in municipalities or year < 2000:
            continue
        value = PopulationDefault(
            ags=ags,
            year=year,
            value=population,
            is_forecast=False,
            provenance=DefaultProvenance(source_dataset='destatis', source_sha256=historical_sha256, method='observed'),
        ).value
        if year in histories[ags]:
            raise ValueError(f'Duplicate historical population for {ags} in {year}.')
        histories[ags][year] = value
        history_urls[ags, year] = url
    missing = municipalities.keys() - histories.keys()
    if missing:
        raise ValueError(f'{len(missing)} municipalities lack observed population: {sorted(missing)[:5]}')
    latest_year = max(y for values in histories.values() for y in values)
    if horizon < latest_year:
        raise ValueError('Population horizon precedes the latest observation.')

    # A published geographic series, with provenance; aggregates are never imported as leaves.
    series: dict[str, dict[int, int]] = defaultdict(dict)
    sources: dict[str, DefaultProvenance] = {}
    direct: dict[str, str] = {}
    aggregates: dict[str, str] = {}
    for row in forecasts.iter_rows(named=True):
        if row['variant'] != DEFAULT_VARIANTS.get(row['state_code']):
            continue
        level = row['administrative_level']
        if level not in ('municipality', 'association', 'combined_municipalities'):
            continue
        code = row['source_region_code']
        if row['year'] in series[code]:
            raise ValueError(f'Duplicate default population forecast for {code} in {row["year"]}.')
        series[code][row['year']] = row['population']
        sources[code] = DefaultProvenance(
            source_dataset='land_population_forecasts',
            source_edition=row['source_edition'],
            source_vintage=row['source_vintage'].isoformat(),
            source_url=row['source_url'],
            source_sha256=row['source_sha256'],
            method='published',
            variant=row['variant'],
            source_region_code=code,
            is_estimate=True,
        )
        if level == 'municipality':
            direct[row['ags']] = code
        else:
            aggregates[row['ars'] or code] = code

    groups: dict[str, list[str]] = defaultdict(list)
    for ags, ars in municipalities.items():
        code = direct.get(ags)
        current = ars
        while code is None and current is not None:
            code = aggregates.get(current)
            current = parents.get(current)
        if code is not None:
            groups[code].append(ags)
    allocated: dict[tuple[str, int], tuple[int, DefaultProvenance]] = {}
    for code, members in groups.items():
        is_direct = len(members) == 1 and direct.get(members[0]) == code
        common = set(histories[members[0]])
        for ags in members[1:]:
            common.intersection_update(histories[ags])
        if not common:
            raise ValueError(f'{code} has no common observed year for forecast allocation.')
        allocation_year = max(common)
        weights = {ags: histories[ags][allocation_year] for ags in members}
        for year in range(latest_year + 1, horizon + 1):
            value, method = _annual_value(series[code], year)
            values = {members[0]: value} if is_direct else _allocate(value, weights)
            for ags, population in values.items():
                provenance = sources[code].model_copy(
                    update={
                        'method': method if is_direct else f'association_allocation:{method}',
                        'allocation_year': None if is_direct else allocation_year,
                        'allocation_share': None if is_direct else weights[ags] / sum(weights.values()),
                    }
                )
                allocated[ags, year] = population, provenance

    output: list[PopulationDefault] = []
    for ags in sorted(municipalities):
        observed = histories[ags]
        last = max(observed)
        # Internal historical gaps are interpolated and labelled, never presented as observed.
        for year in range(min(observed), horizon + 1):
            if year > latest_year and (forecast := allocated.get((ags, year))) is not None:
                value, provenance = forecast
            else:
                value, method = _annual_value(observed, year)
                provenance = DefaultProvenance(
                    source_dataset='destatis',
                    source_url=history_urls[ags, year if year in observed else last],
                    source_sha256=historical_sha256,
                    method='observed' if year in observed else method,
                    is_estimate=year not in observed,
                )
            output.append(PopulationDefault(ags=ags, year=year, value=value, is_forecast=year > last, provenance=provenance))
    return output


def prepare_energy_defaults(  # noqa: C901
    estimates: pl.DataFrame,
    prior: pl.DataFrame,
    *,
    prior_sha256: str,
    input_sha256: str | None = None,
) -> tuple[list[EnergyDefault], set[str]]:
    """Skip conflicted municipalities; split non-grid energy using the demo's Mainz prior."""
    excluded = set(estimates.filter(pl.col('needs_review'))['ags'].to_list())
    weights: dict[str, dict[str, float]] = defaultdict(dict)
    for sector, carrier, value in prior.select('sector', 'energy_carrier', 'value_mwh_per_year').iter_rows():
        if sector not in SECTORS or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid carrier prior.')
        if carrier in weights[sector]:
            raise ValueError(f'Duplicate prior cell: {sector}/{carrier}')
        weights[sector][carrier] = value
    if set(weights) != set(SECTORS) or any('electricity' not in row for row in weights.values()):
        raise ValueError('Carrier prior lacks sectors or electricity cells.')
    cells: dict[tuple[str, int, str], dict[str, StationaryEstimate]] = defaultdict(dict)
    for raw in estimates.iter_rows(named=True):
        row = StationaryEstimate.model_validate(raw)
        if row.ags in excluded:
            continue
        key = row.ags, row.year, row.sector
        quantity = row.quantity
        if quantity in cells[key]:
            raise ValueError(f'Duplicate energy estimate: {key}/{quantity}')
        cells[key][quantity] = row
    output: list[EnergyDefault] = []
    for (ags, year, sector), quantities in sorted(cells.items()):
        if set(quantities) != {'stationary_final_energy', 'grid_electricity'}:
            raise ValueError(f'Incomplete energy estimates for {ags}/{year}/{sector}.')
        total_row, grid_row = quantities['stationary_final_energy'], quantities['grid_electricity']
        total, grid = total_row.value_mwh_per_year, grid_row.value_mwh_per_year
        if not all(math.isfinite(value) and value >= 0 for value in (total, grid)) or grid > total:
            raise ValueError(f'Unflagged inconsistent energy estimates for {ags}/{sector}.')
        remaining = {carrier: value for carrier, value in weights[sector].items() if carrier != 'electricity'}
        denominator = sum(remaining.values())
        if denominator <= 0 and total > grid:
            raise ValueError(f'No non-electric carrier prior for {sector}.')
        split = {
            'electricity': grid,
            **{carrier: (total - grid) * value / denominator if denominator else 0 for carrier, value in remaining.items()},
        }
        for carrier, value in split.items():
            source = grid_row if carrier == 'electricity' else total_row
            provenance = DefaultProvenance(
                source_dataset='stationary_estimates',
                source_sha256=source.source_sha256 or '',
                method='grid_delivery' if carrier == 'electricity' else 'mainz_2018_carrier_split',
                source_year=source.source_year,
                is_estimate=True,
                prior_sha256=prior_sha256,
                upstream_method=source.method,
                empirical_lower_mwh=source.empirical_lower_mwh,
                empirical_upper_mwh=source.empirical_upper_mwh,
                source_kind=source.source_kind,
                source_status=source.source_status,
                source_location_code=source.source_location_code,
                normalizer=source.normalizer,
                normalizer_value=source.normalizer_value,
                coefficient_mwh_per_unit=source.coefficient_mwh_per_unit,
                input_sha256=input_sha256,
            )
            output.append(
                EnergyDefault(ags=ags, year=year, sector=sector, energy_carrier=carrier, value=value, provenance=provenance)
            )
    return output, excluded


def population_frame(values: list[PopulationDefault]) -> pl.DataFrame:
    return pl.DataFrame([
        {
            'ags': value.ags,
            'lau': f'DE_{value.ags}',
            'Year': value.year,
            'population': value.value,
            'Forecast': value.is_forecast,
            'provenance': value.provenance.model_dump_json(),
        }
        for value in values
    ])


def energy_frame(values: list[EnergyDefault]) -> pl.DataFrame:
    return pl.DataFrame([
        {
            'ags': value.ags,
            'Year': value.year,
            'sector': value.sector,
            'energy_carrier': value.energy_carrier,
            'Value': value.value,
            'provenance': value.provenance.model_dump_json(),
        }
        for value in values
    ])
