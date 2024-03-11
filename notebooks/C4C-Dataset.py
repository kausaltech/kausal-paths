import itertools
import pandas as pd
import polars as pl
import os, sys

os.chdir('/Users/mechenich/Projects/Climate-4-CAST')

incsvpath = 'Tampere-Sheet.csv'
outcsvpath = 'Tampere-Parquet.csv'
outdvcpath = 'gpc/demonstration'

# incsvpath = sys.argv[1]
# outcsvpath = sys.argv[2]
# outdvcpath = sys.argv[3]

# ---------------------------------------------------------------------------------------
from pint import UnitRegistry
ureg = UnitRegistry()
ureg.define('EUR = mass')

def pconvert(value):
    if value['Value']:
        pquantity = value['Value'] * ureg(value['Unit'])
        return pquantity.to(value['ToUnit'])._magnitude
    else:
        return float('nan')

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

df = pl.read_csv(incsvpath, separator = ',')
df = df.drop(['Description', 'Action'])

valueindex = df.dtypes.index(pl.Float64)
context = df.columns[:valueindex]
values = df.columns[valueindex:]

dims = ['Sector', 'Scope', 'Energy Carrier', 'GHG']
dims.extend(df.columns[6:valueindex])

df = df.with_columns(pl.when(pl.col('Quantity') == pl.lit('Emissions')).then(pl.lit('Emissions'))
                       .when(pl.col('Quantity') == pl.lit('Emission Factor')).then(pl.lit('Emission Factor'))
                       .when(pl.col('Quantity') == pl.lit('Energy Consumption')).then(pl.lit('Activity'))
                       .when(pl.col('Quantity') == pl.lit('Fuel Consumption')).then(pl.lit('Activity'))
                       .when(pl.col('Quantity') == pl.lit('Mileage')).then(pl.lit('Activity'))
                       .when(pl.col('Quantity') == pl.lit('Price')).then(pl.lit('Price'))
                       .when(pl.col('Quantity') == pl.lit('Unit Price')).then(pl.lit('Unit Price'))
                       .otherwise(pl.lit('Unknown'))
                       .alias('Quantity Type'))

# False = all zero values; true = one or more non-zero values; null = all null values.
df = df.with_columns(pl.any_horizontal(values).alias('Status'))
# df = df.filter(pl.col('Status').is_not_null())

unitcol = df.select('Unit').to_series(0).to_list()
df = df.with_columns(pl.Series(name = 'Unit', values = [x.replace('tCO2e', 't').replace('Mvkm', 'Gm').replace('vkm', 'km').replace('p-km', 'km').replace('€', 'EUR') for x in unitcol]))

scopecol = df.select('Scope').to_series(0).to_list()
idvalues = []
for x in scopecol:
    if x:
        idvalues.append('scope%i' % x)
    else:
        idvalues.append(x)
df = df.with_columns(pl.Series(name = 'Scope', values = idvalues))

for dim in dims[2:]:
    dimcol = df.select(dim).to_series(0).to_list()
    idvalues = []
    for x in dimcol:
        if x:
            idvalues.append(x.lower().replace(' ', '_').replace('-', '_').replace('&', '').replace(',', ''))
        else:
            idvalues.append(x)

    df = df.with_columns(pl.Series(name = dim, values = idvalues))

dfaccum = df.clear()

# ---------------------------------------------------------------------------------------
report = open('Report.txt', 'w')

report.write('Emission Factors: ')
dfef = df.filter(pl.col('Quantity Type') == 'Emission Factor')
if len(dfef) == len(dfef.select(dims).unique()):
    report.write('(%i)' % len(dfef))
else:
    report.write('ERROR! Factors not uniquely identified by dimension category combinations!')

report.write('\n\nUnit Prices: ')
dfup = df.filter(pl.col('Quantity Type') == 'Unit Price')
if len(dfup) == len(dfup.select(dims).unique()):
    report.write('(%i)' % len(dfup))
else:
    report.write('ERROR! Prices not uniquely identified by dimension category combinations!')

# ---------------------------------------------------------------------------------------
for sector in sectors:
    report.write('\n\nGPC Sector %s' % sector)

    for quantity in ['Emissions', 'Activity', 'Price']:
        dff = df.filter((pl.col('Sector') == sector) &
                        (pl.col('Quantity Type') == quantity))

        if not dff.is_empty():
            report.write('\n\t%s (%i, %i)\n\t\tUnits: ' % (quantity,
                                                           len(dff.filter(pl.col('Status') == False)),
                                                           len(dff.filter(pl.col('Status') == True))))

            units = dff.select('Unit').unique(maintain_order = True).to_series(0).to_list()
            if units.count(None) == 1:
                report.write('ERROR! One or more missing units!')
            elif len(units) == 1:
                report.write('%s' % units[0])
            else:
                report.write('%s -> %s' % (units, units[0]))

                for y in values:
                    dff = dff.with_columns(pl.struct(pl.col(y).alias('Value'), pl.col('Unit'),
                                                     pl.lit(units[0]).alias('ToUnit')).map_elements(pconvert).alias(y))
                dff = dff.with_columns(pl.lit(units[0]).alias('Unit'))

            # ---------------------------------------------------------------------------
            report.write('\n\t\tDimensions: ')
            dimcats = dff.select(dims).unique()
            if len(dff) != len(dimcats):
                report.write('\n\t\t\tERROR! Values not uniquely identified by dimension category combinations!')

            for dim in dims:
                cats = dimcats.select(dim).unique().to_series(0).to_list()
                if cats == [None]:
                    dimcats = dimcats.drop(dim)
                elif None in cats:
                    report.write('\n\t\t\tERROR! One or more missing values in dimension "%s".' % dim)

            report.write('%s' % dimcats.columns)

            dfaccum = pl.concat([dfaccum, dff.select(dfaccum.columns)])

            if quantity == 'Activity':
                dff = dff.with_row_count(name = 'Index')

                # Create list of possible join dimensions for the sector's activity records. Begin with all non-null
                # dimensions; never join on 'Scope'; reverse dimension order, so that for combinations of given length,
                # custom dimensions are first considered, then 'Energy Carrier,' and 'Sector' only last.
                joinkeys = dimcats.columns
                joinkeys.remove('Scope')
                joinkeys.reverse()

                for i in range(len(dff)):
                    dfi = dff.filter(pl.col('Index') == i)
                    dfj = dfi.join(dfef, on = joinkeys)

                    if not dfj.is_empty():
                        ijoinkeys = joinkeys
                    else:
                        keycount = len(joinkeys)
                        while dfj.is_empty():
                            keycount -= 1
                            for combo in list(itertools.combinations(joinkeys, keycount)):
                                dfj = dfi.join(dfef, on = combo)

                                remainder = list(set(joinkeys) - set(combo))
                                for r in remainder:
                                    dfj = dfj.filter(pl.col('%s_right' % r).is_null())

                                if not dfj.is_empty():
                                    ijoinkeys = list(combo)
                                    break

                    report.write('\n\t\t%i: %s: %s (%i)' % (i, dfi.select(dimcats.columns).row(0), ijoinkeys, len(dfj)))

                    # -------------------------------------------------------------------
                    # Always take 'Sector' and 'Scope' values from activity records.
                    acols = [x if x in ijoinkeys else '%s_right' % x for x in dfaccum.columns]
                    acols[0] = 'Sector'
                    acols[1] = 'Scope'
                    aframe = dfj.select(acols)
                    aframe.columns = dfaccum.columns

                    funits = aframe.select('Unit').unique(maintain_order = True).to_series(0).to_list()
                    funit = '%s/%s' % (funits[0].split('/')[0], units[0].split('/')[0])

                    report.write('\n\t\t\tFactors: ')
                    if (len(funits) == 1) & (funits[0] == funit):
                        report.write('%s' % funit)
                    else:
                        report.write('%s -> %s' % (funits, funit))

                        for y in values:
                            aframe = aframe.with_columns(pl.struct(pl.col(y).alias('Value'), pl.col('Unit'),
                                                                   pl.lit(funit).alias('ToUnit')).map_elements(pconvert).alias(y))
                        aframe = aframe.with_columns(pl.lit(funit).alias('Unit'))

                    dfaccum = pl.concat([dfaccum, aframe])



                # -----------------------------------------------------------------------
                for i in range(len(dff)):
                    dfi = dff.filter(pl.col('Index') == i)
                    dfj = dfi.join(dfup, on = joinkeys)

                    ijoinkeys = False
                    if not dfj.is_empty():
                        ijoinkeys = joinkeys
                    else:
                        keycount = len(joinkeys)
                        while dfj.is_empty() and keycount > 1:
                            keycount -= 1
                            for combo in list(itertools.combinations(joinkeys, keycount)):
                                dfj = dfi.join(dfup, on = combo)

                                remainder = list(set(joinkeys) - set(combo))
                                for r in remainder:
                                    dfj = dfj.filter(pl.col('%s_right' % r).is_null())

                                if not dfj.is_empty():
                                    ijoinkeys = list(combo)
                                    break

                    if ijoinkeys:
                        report.write('\n\t\t%i: %s: %s (%i)' % (i, dfi.select(dimcats.columns).row(0), ijoinkeys, len(dfj)))

                        # ---------------------------------------------------------------
                        # Always take 'Sector' and 'Scope' values from activity records.
                        acols = [x if x in ijoinkeys else '%s_right' % x for x in dfaccum.columns]
                        acols[0] = 'Sector'
                        acols[1] = 'Scope'
                        aframe = dfj.select(acols)
                        aframe.columns = dfaccum.columns

                        funits = aframe.select('Unit').unique(maintain_order = True).to_series(0).to_list()
                        funit = '%s/%s' % (funits[0].split('/')[0], units[0].split('/')[0])

                        report.write('\n\t\t\tPrices: ')
                        if (len(funits) == 1) & (funits[0] == funit):
                            report.write('%s' % funit)
                        else:
                            report.write('%s -> %s' % (funits, funit))

                            for y in values:
                                aframe = aframe.with_columns(pl.struct(pl.col(y).alias('Value'), pl.col('Unit'),
                                                                       pl.lit(funit).alias('ToUnit')).map_elements(pconvert).alias(y))
                            aframe = aframe.with_columns(pl.lit(funit).alias('Unit'))

                        dfaccum = pl.concat([dfaccum, aframe])



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

        dfmain = pl.concat([dfmain, mframe])

dfmain = dfmain.unique(maintain_order = True)
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
