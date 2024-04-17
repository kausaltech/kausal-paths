import itertools
import pandas as pd
import polars as pl
import sys

from dotenv import load_dotenv
load_dotenv()

incsvpath = sys.argv[1]
incsvsep = sys.argv[2]
outcsvpath = sys.argv[3]
outdvcpath = sys.argv[4]

# ---------------------------------------------------------------------------------------
sectors = ['I', 'II', 'III', 'IV', 'V',
           'I.1', 'I.1.1', 'I.1.2', 'I.1.3',
           'I.2', 'I.2.1', 'I.2.2', 'I.2.3',
           'I.3', 'I.3.1', 'I.3.2', 'I.3.3',
           'I.4', 'I.4.1', 'I.4.2', 'I.4.3',
           'I.5', 'I.5.1', 'I.5.2', 'I.5.3',
           'I.6', 'I.6.1', 'I.6.2', 'I.6.3',
           'I.7.1', 'I.8.1',
           'II.1', 'II.1.1', 'II.1.2', 'II.1.3',
           'II.2', 'II.2.1', 'II.2.2', 'II.2.3',
           'II.3', 'II.3.1', 'II.3.2', 'II.3.3',
           'II.4', 'II.4.1', 'II.4.2', 'II.4.3',
           'II.5', 'II.5.1', 'II.5.2', 'II.5.3',
           'III.1', 'III.1.1', 'III.1.2',
           'III.2', 'III.2.1', 'III.2.2',
           'III.3', 'III.3.1', 'III.3.2',
           'III.4', 'III.4.1', 'III.4.2',
           'IV.1', 'IV.2',
           'V.1', 'V.2', 'V.3']

qtypelookup = {'Emissions': 'Emissions',
               'Emission Factor': 'Emission Factor',
               'Energy Consumption': 'Activity',
               'Energy Factor': 'Activity Factor',
               'Floor Area': 'Activity',
               'Fuel Consumption': 'Activity',
               'Fuel Factor': 'Activity Factor',
               'Mileage': 'Activity',
               'Mileage Factor': 'Activity Factor',
               'Price': 'Price',
               'Unit Price': 'Unit Price'}

unitreplace = [['tCO2e', 't'],
               ['vkm', 'km'], ['p-km', 'km'], ['Mkm', 'Gm'],
               ['€', 'EUR']]

# ---------------------------------------------------------------------------------------
from pint import UnitRegistry
ureg = UnitRegistry()
ureg.define('EUR = [currency]')

def pconvert(value):
    if value['Value']:
        pquantity = value['Value'] * ureg(value['Unit'])
        return pquantity.to(value['ToUnit'])._magnitude
    else:
        return float('nan')

# ---------------------------------------------------------------------------------------
df = pl.read_csv(incsvpath, separator = incsvsep)

df = df.drop(['Description', 'Action'])
context = []
values = []
for c in df.columns:
    if c.isdigit():
        values.append(c)
    else:
        context.append(c)

dims = [c for c in context if c not in ['Quantity', 'Unit']]

df = df.with_columns(pl.col('Quantity').map_dict(qtypelookup).alias('Quantity Type'))

unitcol = df.select('Unit').to_series(0).to_list()
for ur in unitreplace:
    unitcol = [x.replace(ur[0], ur[1]) for x in unitcol]
df = df.with_columns(pl.Series(name = 'Unit', values = unitcol))

scopecol = df.select('Scope').to_series(0).to_list()
labels = []
for x in scopecol:
    if x:
        labels.append('Scope %i' % x)
    else:
        labels.append(x)
df = df.with_columns(pl.Series(name = 'Scope', values = labels))

# False = all zero values; true = one or more non-zero values; null = one or more null values, with zero values.
df = df.with_columns(pl.any_horizontal(values).alias('Status'))

dfaccum = df.clear()

# ---------------------------------------------------------------------------------------
report = open('Report.txt', 'w')

def uniqueidcheck(df, qtype, dlist, rfile):
    text = '%s: ' % qtype
    dff = df.filter(pl.col('Quantity Type') == qtype)
    if len(dff) == len(dff.select(dlist).unique()):
        text += '(%i)\n' % len(dff)
    else:
        text += "ERROR! Not uniquely identified by dimension category combinations!\n"

    rfile.write(text)
    return dff

factordf = {}
for qtype in ['Activity Factor', 'Emission Factor', 'Unit Price']:
    factordf[qtype] = uniqueidcheck(df, qtype, dims, report)

# ---------------------------------------------------------------------------------------
def factorfinder(df, dimcols, factorframe, accumcols, aunit, rfile):
    dfa = df.select(accumcols).clear()

    if not factorframe.is_empty():
        # Create list of possible join dimensions for the sector's activity records. Begin with all non-null
        # dimensions; never join on 'Scope'; reverse dimension order, so that for combinations of given length,
        # custom dimensions are first considered, then 'Energy Carrier,' and 'Sector' only last.
        joinkeys = list(dimcols)
        joinkeys.remove('Scope')
        joinkeys.reverse()

        df = df.with_row_count(name = 'Index')
        for i in range(len(df)):
            dfi = df.filter(pl.col('Index') == i)
            dfj = dfi.join(factorframe, on = joinkeys)

            ijoinkeys = False
            if not dfj.is_empty():
                ijoinkeys = joinkeys
            else:
                keycount = len(joinkeys)
                while dfj.is_empty() and keycount > 1:
                    keycount -= 1
                    for combo in list(itertools.combinations(joinkeys, keycount)):
                        dfj = dfi.join(factorframe, on = combo)

                        remainder = list(set(joinkeys) - set(combo))
                        for r in remainder:
                            dfj = dfj.filter(pl.col('%s_right' % r).is_null())

                        if not dfj.is_empty():
                            # Having found a set of dimension(s) on which to join, check units for compatibility.
                            funits = dfj.select('Unit_right').unique(maintain_order = True).to_series(0).to_list()
                            fseries = [funit.split('/')[1] == aunit.split('/')[0] for funit in funits]

                            dfj = dfj.filter(pl.Series(fseries))
                            if not dfj.is_empty():
                                ijoinkeys = list(combo)
                                break

            if ijoinkeys:
                rfile.write('\n\t\t%i: %s: %s (%i) ' % (i, dfi.select(dimcols).row(0), ijoinkeys, len(dfj)))

                # Always take 'Sector' and 'Scope' values from activity records.
                acols = [x if x in ijoinkeys else '%s_right' % x for x in accumcols]
                acols[0] = 'Sector'
                acols[1] = 'Scope'
                aframe = dfj.select(acols)
                aframe.columns = accumcols

                aframe = unitcheck(aframe, values, rfile)

                dfa = pl.concat([dfa, aframe])

    dfa = dfa.unique(maintain_order = True)
    return dfa

# ---------------------------------------------------------------------------------------
def unitcheck(df, vlist, rfile):
    rfile.write('Units: ')
    units = df.select('Unit').unique(maintain_order = True).to_series(0).to_list()
    if units.count(None) == 1:
        rfile.write('ERROR! One or more missing units!')
    elif len(units) == 1:
        rfile.write('%s' % units[0])
    else:
        rfile.write('%s -> %s' % (units, units[0]))

        for y in vlist:
            df = df.with_columns(pl.struct(pl.col(y).alias('Value'), pl.col('Unit'),
                                           pl.lit(units[0]).alias('ToUnit')).map_elements(pconvert).alias(y))
        df = df.with_columns(pl.lit(units[0]).alias('Unit'))

    return df

def dimensioncheck(df, dlist, rfile):
    rfile.write('\n\t\tDimensions: ')
    dimcats = df.select(dlist).unique()
    if len(df) != len(dimcats):
        rfile.write('\n\t\t\tERROR! Values not uniquely identified by dimension category combinations!')

    for dim in dlist:
        cats = dimcats.select(dim).unique().to_series(0).to_list()
        if cats == [None]:
            dimcats = dimcats.drop(dim)
        elif None in cats:
            report.write('\n\t\t\tERROR! One or more missing values in dimension "%s".' % dim)

    rfile.write('%s' % dimcats.columns)
    return dimcats.columns

# ---------------------------------------------------------------------------------------
for sector in sectors:
    report.write('\nGPC Sector %s' % sector)

    for qtype in ['Emissions', 'Activity', 'Price']:
        dff = df.filter((pl.col('Sector') == sector) &
                        (pl.col('Quantity Type') == qtype))

        if not dff.is_empty():
            report.write('\n\t%s (%i T, %i F, %i N):\n\t\t' % (qtype,
                                                               len(dff.filter(pl.col('Status') == True)),
                                                               len(dff.filter(pl.col('Status') == False)),
                                                               len(dff.filter(pl.col('Status').is_null()))))
            dff = unitcheck(dff, values, report)
            qunit = dff.select('Unit').unique().to_series(0).to_list()[0]
            qdims = dimensioncheck(dff, dims, report)

            dfaccum = pl.concat([dfaccum, dff.select(dfaccum.columns)])

            # ---------------------------------------------------------------------------
            if qtype == 'Activity':
                afdf = factorfinder(dff, qdims, factordf['Activity Factor'], dfaccum.columns, qunit, report)
                updf1 = factorfinder(dff, qdims, factordf['Unit Price'], dfaccum.columns, qunit, report)

                if not afdf.is_empty():
                    report.write('\n\t\tConversion via Activity Factor')
                    aunit = afdf.select('Unit').unique().to_series(0).to_list()[0]
                    adims = dimensioncheck(afdf, dims, report)

                    if len(qdims) >= len(adims):
                        tframe = dff.with_columns(pl.lit(aunit).alias('Unit'))
                        efdf = factorfinder(tframe, qdims, factordf['Emission Factor'], dfaccum.columns, aunit, report)
                        updf2 = factorfinder(tframe, qdims, factordf['Unit Price'], dfaccum.columns, aunit, report)
                    else:
                        efdf = factorfinder(afdf, adims, factordf['Emission Factor'], dfaccum.columns, aunit, report)
                        updf2 = factorfinder(afdf, adims, factordf['Unit Price'], dfaccum.columns, aunit, report)

                else:
                    efdf = factorfinder(dff, qdims, factordf['Emission Factor'], dfaccum.columns, qunit, report)
                    updf2 = dff.clear()

                if efdf.is_empty():
                    report.write('\n\t\tERROR! No emission factor(s) found.')

                for frame in [afdf, efdf, updf1, updf2]:
                    if not frame.is_empty():
                        dfaccum = pl.concat([dfaccum, frame])

# ---------------------------------------------------------------------------------------
dfmain = df.head(1).select(context).with_columns([(pl.lit(0.0).alias('Value').cast(pl.Float64)),
                                                  (pl.lit(0).alias('Year').cast(pl.Int64))]).clear()

dfaccum = dfaccum.with_row_count(name = 'Index')
for i in range(len(dfaccum)):
    for y in values:
        mcols = list(context)
        mcols.extend([y])

        mframe = dfaccum.filter(pl.col('Index') == i).select(mcols).with_columns(pl.lit(y).cast(pl.Int64))
        mframe.columns = dfmain.columns
        if mframe['Value'][0] is not None:
            dfmain = pl.concat([dfmain, mframe])

# dfmain = dfmain.unique(maintain_order = True)
dfmain = dfmain.with_columns(pl.col('Scope').cast(pl.Utf8).alias('Scope'))

report.close()

if outcsvpath.upper() not in ['N', 'NONE']:
    dfmain.write_csv(outcsvpath)

if outdvcpath.upper() not in ['N', 'NONE']:
    from dvc_pandas import Dataset, Repository

    indexcols = list(dims)
    indexcols.extend(['Quantity', 'Year'])
    pdindex = pd.MultiIndex.from_frame(pd.DataFrame(dfmain.select(indexcols).fill_null('.'), columns = indexcols))

    valuecols = list(set(dfmain.columns) - set(indexcols))
    pdframe = pd.DataFrame(dfmain.select(valuecols), index = pdindex, columns = valuecols)

    ds = Dataset(pdframe, identifier = outdvcpath)
    repo = Repository(repo_url = 'git@github.com:kausaltech/dvctest.git', dvc_remote = 'kausal-s3')
    repo.push_dataset(ds)
