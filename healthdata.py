import pandas as pd
from dvc_pandas import Dataset, Repository
from nodes.context import unit_registry

repo = Repository(repo_url='https://github.com/kausaltech/dvctest.git')

# Air pollution

df = pd.DataFrame({
    'ERF': pd.Series([
        'PM2_5 mortality',
        'PM2_5 work_days_lost',
        'NOx mortality',
        'PM10 chronic_bronchitis']),
    'pollutant': pd.Series(['PM2_5', 'PM2_5', 'NOx', 'PM10']),
    'response': pd.Series(['mortality', 'work_days_lost', 'mortality', 'chronic_bronchitis']),
    'period': pd.Series([1.] * 4),
    'route': pd.Series(['inhalation'] * 4),
    'ef_function': pd.Series(['relative risk'] * 4),
    'param_inv_inhalation': pd.Series([0.007696104114, 0.00449733656, 0.0019802627, 0.0076961941]),
    'param_inhalation': pd.Series([0.] * 4),
    'default_incidence': pd.Series([18496 / 5533793, 12, 18496 / 5533793, 390 / 100000]),
    'case_burden': pd.Series([10.6, 0.00027, 10.6, 0.99]),
    'case_cost': pd.Series([0., 152, 0, 62712]),
})
unit = dict({
    'period': 'year / incident',
    'param_inv_inhalation': 'm**3 / ug',
    'param_inhalation': 'ug / m**3',
    'default_incidence': 'cases / personyear',
    'case_burden': 'DALY / case',
    'case_cost': 'EUR / case',
})

metadata = {
    'references': {
        'PM2_5 mortality': {
            'general': 'http://en.opasnet.org/w/ERF_of_outdoor_air_pollution',
            'function': 'log(1.08)/10 Chen & Hoek, 2020',
            'default_-incidence': 'https://stat.fi/til/ksyyt/2020/ksyyt_2020_2021-12-10_tau_001_fi.html',
            'case_burden': 'De Leeuw & Horàlek 2016/5 http://fi.opasnet.org/fi/Kiltova#PAQ2018'
        },
        'PM2_5 work_days_lost': {
            'general': 'http://fi.opasnet.org/fi/Kiltova',
            'function': 'log(1.046)/10 HRAPIE',
            'incidence': 'PAQ2018',
            'case_burden': '0.099 DW * 0.00274 a, Heimtsa & Intarese http://fi.opasnet.org/fi/Kiltova#PAQ2018',
            'case_cost': 'Holland et al., 2014'
        },
        'NOx mortality': {
            'general': 'For NO2 atm.  http://fi.opasnet.org/fi/Kiltova, Atkinson et al., 2017',
            'function': 'log(1.02)/10',
            'default_incidence': 'Same as PM2.5 https://stat.fi/til/ksyyt/2020/ksyyt_2020_2021-12-10_tau_001_fi.html',
            'case_burden': 'Same as PM2.5 De Leeuw & Horàlek 2016/5 http://fi.opasnet.org/fi/Kiltova#PAQ2018'
        },
        'PM10 chronic bronchitis': {
            'general': 'http://fi.opasnet.org/fi/Kiltova',
            'function': 'log(1.08)/10 HRAPIE',
            'default_incicende': 'PAQ2018',
            'case_burden': 'http://fi.opasnet.org/fi/Kiltova#PAQ2018',
            'case_cost': 'Holland et al., 2014'
        }
    }
}

ds_air = Dataset(
    df=df,
    identifier='hia/exposure_response/air_pollution',
    units=unit,
    metadata=metadata)

repo.add(ds_air)

# Noise

df = pd.DataFrame({
    'ERF': pd.Series([
        'noise highly_annoyed_road',
        'noise highly_annoyed_rail',
        'noise highly_annoyed_air',
        'noise highly_sleep_disturbed_road',
        'noise highly_sleep_disturbed_rail',
        'noise highly_sleep_disturbed_air']),
    'pollutant': pd.Series(['noise'] * 6),
    'response': pd.Series(['highly_annoyed'] * 3 + ['highly_sleep_disturbed'] * 3),
    'period': pd.Series([1] * 6),
    'route': pd.Series(['noise'] * 6),
    'er_function': pd.Series(['polynomial'] * 6),
    'param_exposure': pd.Series([42, 42, 42, 0, 0, 0]),
    'param_poly0': pd.Series([0, 0, 0, 0.208, 0.113, 0.18147]),
    'param_poly1': pd.Series([5.118E-03, 1.695E-03, 2.939E-03, -1.050E-02, -5.500E-03, -9.560E-03]),
    'param_poly2': pd.Series([-1.436E-04, -7.851E-05, 3.932E-04, 1.486E-04, 7.590E-05, 1.482E-04]),
    'param_poly3': pd.Series([9.868E-06, 7.239E-06, -9.199E-07, 0, 0, 0]),
    'case_burden': pd.Series([0.02, 0.02, 0.02, 0.07, 0.07, 0.07]),
})

unit = dict({
    'period': 'a / incident',
    'param_exposure': 'Lden dB',
    'param_poly1': '(Lden dB)**-1',
    'param_poly2': '(Lden dB)**-2',
    'param_poly3': '(Lden dB)**-3',
    'case_burden': 'DALY / case',
})

metadata = {
    'references': {
        'general': 'http://fi.opasnet.org/fi/Liikenteen_terveysvaikutukset. For exposure data, see e.g. https://cdr.eionet.europa.eu/fi/eu/noise/df8/2017/envwjdfiq',
        'function': 'All exposure-response functions: WHO & JRC 2011 (values scaled from % to fraction). https://apps.who.int/iris/handle/10665/326424',
        'case_burden': 'disability weight 0.02, duration 1 year',
    }
}

ds_noise = Dataset(
    df=df, identifier='hia/exposure_response/noise',
    units=unit,
    metadata=metadata)

repo.add(ds_noise)

# Waterborne microbes

df = pd.DataFrame({
    'ERF': pd.Series([
        'campylobacter infection',
        'rotavirus infection',
        'norovirus infection',
        'sapovirus infection',
        'cryptosporidium infection',
        'EColiO157H7 infection',
        'giardia infection']),
    'pollutant': pd.Series(['campylobacter', 'rotavirus', 'norovirus', 'sapovirus',
                            'cryptosporidium', 'EColiO15H7', 'giardia']),
    'response': pd.Series(['infection'] * 7),
    'period': pd.Series([1] * 7),
    'route': pd.Series(['ingestion'] * 7),
    'er_function': pd.Series(['beta poisson approximation'] + ['exact beta poisson'] * 5 + ['exponential']),
    'case_burden': pd.Series([0.002] * 7),
    'param_ingestion': pd.Series([0.011, 0.167, 0.04, 0.04, 0.115, 0.157, None]),
    'param_ingestion2': pd.Series([None, 0.191, 0.055, 0.055, 0.176, 9.16, None]),
    'param_scale': pd.Series([0.024] + [None] * 6),
    'param_inv_ingestion': pd.Series([None] * 6 + [0.0199]),
})

units = {
    'period': 'days / incident',
    'case_burden': 'DALY / case',
    'param_ingestion': 'microbes / day',
    'param_ingestion2': 'microbes / day',
    'param_inv_ingestion': 'day / microbes'
}

metadata = {
    'general': 'http://en.opasnet.org/w/Water_guide',
    'function': 'http://en.opasnet.org/w/ERF_of_waterborne_microbes'
}

ds = Dataset(
    df=df,
    identifier='hia/exposure_response/microbes',
    units=units,
    metadata=metadata)

repo.add(ds)

# Intake fractions

df = pd.DataFrame({
    'pollutant': pd.Series(['PM10-2_5'] * 4 + ['PM2_5'] * 4 + ['SO2', 'NOx', 'NH3']),
    'emission height': pd.Series(['high', 'low', 'ground', 'average'] * 2 + [None] * 3),
    'urban': pd.Series([8.8, 13, 40, 37, 11, 15, 44, 26, 0.99, 0.2, 1.7]),
    'rural': pd.Series([0.7, 1.1, 3.7, 3.4, 1.6, 2, 3.8, 2.6, 0.79, 0.17, 1.7]),
    'remote': pd.Series([0.04, 0.04, 0.04, 0.04, 0.1, 0.1, 0.1, 0.1, 0.05, 0.01, 0.1]),
    'average': pd.Series([5, 7.5, 23, 21, 6.8, 6.8, 25, 15, 0.89, 0.18, 1.7]),
})

df = df.melt(id_vars=['pollutant', 'emission height'], value_name='Value')

metadata = {
    'general': 'Humbert et al. 2011 https://doi.org/10.1021/es103563z http://en.opasnet.org/w/Intake_fractions_of_PM#Data',
}

unit = dict({
    'Value': 'ppm'
})

ds_if = Dataset(
    df=df,
    identifier='hia/intake_fraction/air_pollution',
    units=unit,
    metadata=metadata)

repo.add(ds_if)
repo.push()