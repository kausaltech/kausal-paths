import pandas as pd
from dvc_pandas import Dataset, Repository
from nodes.context import unit_registry

repo = Repository(repo_url='https://github.com/kausaltech/dvctest.git')

# Physical activity

df = pd.DataFrame({
    'vehicle': pd.Series(['walking', 'cycling']),
    'velocity': pd.Series([5.3, 14.]),
    'metabolic_equivalent': pd.Series([4., 6.8]),
    'weekly_activity': pd.Series([3., 3.]),
    'ERF': pd.Series(['walking mortality', 'cycling mortality']),
    'pollutant': pd.Series(['physical activity'] * 2),
    'response': pd.Series(['mortaility'] * 2),
    'period': pd.Series([1, 1]),
    'route': pd.Series(['exercise'] * 2),
    'er_function': pd.Series(['relative risk'] * 2),
    'param_inv_exposure': pd.Series([-0.0093653792] * 2),
})

unit = dict({
    'velocity': 'km / h',
    'metabolic_equivalent': 'METh / h',
    'weekly_activity': 'day / week',
    'period': 'a / incident',
    'param_inv_exposure': 'week / METh',
})

metadata = dict({
    'references': {
        'general': 'http://fi.opasnet.org/fi/Liikenteen_terveysvaikutukset ' +
        'Lehtomäki, H., Karvosenoja, N., Paunu, V-V., Korhonen, A., Hänninen, O., ' +
        'Tuomisto, J., Karppinen, A., Kukkonen, J. & Tainio, M. 2021. Liikenteen ' +
        'terveysvaikutukset Suomessa ja suurimmissa kaupungeissa. Suomen ympäristökeskuksen ' +
        'raportteja 16/2021. http://hdl.handle.net/10138/329273',
        'velocity': 'Kelly ym. 2014',
        'metabolic_equivalent': 'Kahlmeier, S., Götschi, T., Cavill, N., Castro Fernandez, ' +
        'A., Brand, C., Rojas Rueda, D., Woodcock, J., Kelly, P., Lieb, C., Oja, P., Foster, ' +
        'C., Rutter, H., & Racioppi, F. 2017. Health economic assessment tool (HEAT) for walking ' +
        'and for cy- cling. Methods and user guide on physical activity, air pollution, injuries ' +
        'and carbon impact assessments. World Health Organization. Denmark, Copenhagen. ' +
        'https://www.euro.who.int/__data/assets/pdf_file/0010/352963/Heat.pdf',
        'weekly_activity': 'Cambridgen yliopisto 2020, Woodcock, J., Tainio, M., Cheshire, J., ' +
        'O’Brien, O. & Goodman, A. 2014. Health effects of the London bicycle sharing system: ' +
        'health impact modelling study. The BMJ 348:g425.',
        'er_function': 'Kelly, P., Kahlmeier, S., Götschi, T. et al. Systematic review and ' +
        'meta-analysis of reduction in all-cause mortality from walking and cycling and shape ' +
        'of dose response relationship. Int J Behav Nutr Phys Act 11, 132 (2014). ' +
        'https://doi.org/10.1186/s12966-014-0132-x',
        'param_inv_exposure': 'log(0.9)/(11.25 METh/week)',
    }
})

ds_act = Dataset(
    df,
    identifier='hia/exposure_response/physical_activity',
    units=unit,
    metadata=metadata)

repo.add(ds_act)

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
        'general': 'http://fi.opasnet.org/fi/Liikenteen_terveysvaikutukset. For exposure data, ' +
        'see e.g. https://cdr.eionet.europa.eu/fi/eu/noise/df8/2017/envwjdfiq',
        'function': 'All exposure-response functions: WHO & JRC 2011 (values scaled from % to ' +
        'fraction). https://apps.who.int/iris/handle/10665/326424',
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
    'references': {
        'general': 'http://en.opasnet.org/w/Water_guide',
        'function': 'http://en.opasnet.org/w/ERF_of_waterborne_microbes'
    }
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
    'references': {
        'general': 'Humbert et al. 2011 https://doi.org/10.1021/es103563z ' +
        'http://en.opasnet.org/w/Intake_fractions_of_PM#Data',
    }
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

# Toxic equivalency factors TEF

df = pd.DataFrame({
    'group': pd.Series(
        ['chlorinated dibenzo-p-dioxins'] * 7 +
        ['chlorinatd dibenzofurans'] * 10 +
        ['non-ortho-substituted PCBs'] * 4 +
        ['mono-ortho-substituted PCBs'] * 8),
    'compound': pd.Series([
        '2378-TCDD', '12378-PeCDD', '123478-HxCDD', '123678-HxCDD', '123789-HxCDD', '1234678-HpCDD', 'OCDD',
        '2378-TCDF', '12378-PeCDF', '23478-PeCDF', '123478-HxCDF', '123678-HxCDF', '123789-HxCDF', 
        '234678-HxCDF', '1234678-HpCDF', '1234789-HpCDF', 'OCDF', "3,3',4,4'-tetraCB", "3,4,4',5-tetraCB", 
        "3,3',4,4',5-pentaCB", "3,3',4,4',5,5'-hexaCB", "2,3,3',4,4'-pentaCB", "2,3,4,4',5-pentaCB", 
        "2,3',4,4',5-pentaCB", "2',3,4,4',5-pentaCB", "2,3,3',4,4',5-hexaCB", "2,3,3',4,4',5'-hexaCB", 
        "2,3',4,4',5,5'-hexaCB", "2,3,3',4,4',5,5'-heptaCB"]),
    'compound2': pd.Series([
        '2378TCDD', '12378PeCDD', '123478HxCDD', '123678HxCDD', '123789HxCDD', '1234678HpCDD',
        'OCDD', '2378TCDF', '12378PeCDF', '23478PeCDF', '123478HxCDF', '123678HxCDF',
        '123789HxCDF', '234678HxCDF', '1234678HpCDF', '1234789HpCDF', 'OCDF',
        '33_44_tetraCB', '344_5tetraCB', '33_44_5pentaCB', '33_44_55_hexaCB', '233_44_pentaCB',
        '2344_5pentaCB', '23_44_5pentaCB', '2_344_5pentaCB', '233_44_5hexaCB', 
        '233_44_5_hexaCB', '23_44_55_hexaCB', '233_44_55_heptaCB']),
    'compound3': pd.Series([
        'TCDD', 'PeCDD', '123478HCDD', '123678HCDD', '123789HCDD', None, None, 'TCDF', None, None,
        '123478HCDF', '123678HCDF', '123789HCDF', '234678HCDF', None, None, None,
        'PCB77', 'PCB81', 'PCB126', 'PCB169', 'PCB105', 'PCB114', 'PCB118', 'PCB123', 'PCB156', 'PCB157',
        'PCB167', 'PCB189'
    ]),
    'compound4': pd.Series([None] * 17 + ['CoPCB-77', 'CoPCB-81', 'CoPCB-126', 'CoPCB-169'] + [None] * 8),
    'default_paradigm': pd.Series([False] * 29 + [True] * 29 + [False] * 29),
    'WHO1998': pd.Series([1, 1, 0.1, 0.1, 0.1, 0.01, 0.0001, 0.1, 0.05, 0.5, 0.1, 0.1, 0.1, 0.1,
        0.01, 0.01, 0.0001, 0.0001, 0.0001, 0.1, 0.01, 0.0001, 0.0005, 0.0001, 0.0001, 0.0005,
        0.0005, 0.00001, 0.0001]),
    'WHO2005': pd.Series([1, 1, 0.1, 0.1, 0.1, 0.01, 0.0003, 0.1, 0.03, 0.3, 0.1, 0.1, 0.1, 0.1,
        0.01, 0.01, 0.0003, 0.0001, 0.0003, 0.1, 0.03, 0.00003, 0.00003, 0.00003, 0.00003, 0.00003,
        0.00003, 0.00003, 0.00003]),
    'EU_IED2014': pd.Series([1, 0.5, 0.1, 0.1, 0.1, 0.01, 0.001, 0.1, 0.05, 0.5, 0.1, 0.1, 0.1,
        0.1, 0.01, 0.01, 0.001])
})

out = df.copy()[['group', 'compound', 'WHO1998', 'WHO2005', 'EU_IED2014']]
out = out.melt(id_vars=['group', 'compound'], var_name='paradigm', value_name='Value')
tmp = df.copy()[['group', 'compound2', 'WHO1998', 'WHO2005', 'EU_IED2014']]
tmp.rename(columns={'compound2': 'compound'}, inplace=True)
tmp = tmp.melt(id_vars=['group', 'compound'], var_name='paradigm', value_name='Value')
out = out.append(tmp)
tmp = df.copy()[['group', 'compound3', 'WHO1998', 'WHO2005', 'EU_IED2014']]
tmp.rename(columns={'compound3': 'compound'}, inplace=True)
tmp = tmp.melt(id_vars=['group', 'compound'], var_name='paradigm', value_name='Value')
out = out.append(tmp)
tmp = df.copy()[['group', 'compound4', 'WHO1998', 'WHO2005', 'EU_IED2014']]
tmp.rename(columns={'compound4': 'compound'}, inplace=True)
tmp = tmp.melt(id_vars=['group', 'compound'], var_name='paradigm', value_name='Value')
out = out.append(tmp)
out = out.dropna(how='any')

unit = {'Value': 'TEF'}

metadata = {
    'references': {
        'general': 'http://en.opasnet.org/w/Toxic_equivalency_factor http://en.opasnet.org/w/Toxic_equivalency_factor_references',
        'WHO2005': 'Martin Van den Berg, Linda S. Birnbaum, Michael Denison, Mike De Vito, ' +
            'William Farland, Mark Feeley, Heidelore Fiedler, Helen Hakansson, Annika Hanberg, ' +
            'Laurie Haws, Martin Rose, Stephen Safe, Dieter Schrenk, Chiharu Tohyama, Angelika ' +
            'Tritscher, Jouko Tuomisto, Mats Tysklind, Nigel Walker, and Richard E. Peterson: ' +
            'The 2005 World Health Organization Reevaluation of Human and Mammalian Toxic Equivalency ' +
            'Factors for Dioxins and Dioxin-Like Compounds. Toxicological Sciences 93(2), 223–241 ' +
            '(2006) doi:10.1093/toxsci/kfl055.'
    }
}

ds_tef = Dataset(
    df=out,
    identifier='hia/dioxin/toxic_equivalency_factors',
    units=unit,
    metadata=metadata
)

repo.add(ds_tef)

repo.push()
