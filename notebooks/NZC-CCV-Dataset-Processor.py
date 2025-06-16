from __future__ import annotations  # noqa: N999

import polars as pl

# ---------------------------------------------------------------------------------------
pop_measures = ['022', '023', '024', '025', '046', '051', '058', '059', '081', '083', '090', '122',
                '162', '186', '189', '190', '191', '192', '193', '194', '250', '253', '254', '255',
                '256', '257', '258', '261', '262', '263', '266', '267', '268', '271', '272', '273',
                '274', '277', '278', '279', '015']

dh_measures = ['F130', 'F131', 'F132', 'F133', '367', '368', '369', '370']
incin_measures = ['236', '237', '238', '239']
w_measures = ['F137', 'F138']
w2030_measures = ['374', '375']

# ---------------------------------------------------------------------------------------
df = pl.read_csv('42_cities_input_9_clusters.csv')

cols = [col.split('/')[0] for col in df.columns]
df.columns = cols
df = df.drop(['City', 'Country', 'NUT3', 'HDD_y', 'CDD_y', 'HDD', 'Population density', 'Cluster'])

# ---------------------------------------------------------------------------------------
all_df = df.group_by('Cluster_42').agg(pl.all().mean())

dh_df = df.filter(pl.col('F125').gt(0.0)).group_by('Cluster_42').agg(pl.all().mean()).select(dh_measures + ['Cluster_42'])

incin_df = df.filter(pl.col('200').gt(0.0) |
                     pl.col('206').gt(0.0) |
                     pl.col('212').gt(0.0) |
                     pl.col('218').gt(0.0) |
                     pl.col('224').gt(0.0) |
                     pl.col('230').gt(0.0)).group_by('Cluster_42').agg(pl.all().mean()).select(incin_measures + ['Cluster_42'])

w_df = df.filter((pl.col('200').gt(0.0) |
                  pl.col('206').gt(0.0) |
                  pl.col('212').gt(0.0) |
                  pl.col('218').gt(0.0) |
                  pl.col('224').gt(0.0) |
                  pl.col('230').gt(0.0)) &
                 pl.col('F133').gt(0.0)).group_by('Cluster_42').agg(pl.all().mean()).select(w_measures + ['Cluster_42'])


w2030_df = df.filter(pl.col('370').gt(0.0)).group_by('Cluster_42').agg(pl.all().mean()).select(w2030_measures + ['Cluster_42'])

# ---------------------------------------------------------------------------------------
droplist = []
for mlist in [dh_measures, incin_measures, w_measures, w2030_measures]:
    droplist.extend(mlist)

all_df = all_df.drop(droplist)

for jdf in [dh_df, incin_df, w_df, w2030_df]:
    all_df = all_df.join(jdf, on = 'Cluster_42')

for measure in pop_measures:
    all_df = all_df.with_columns(pl.col(measure) / pl.col('015'))

# ---------------------------------------------------------------------------------------
all_df = all_df.transpose(include_header = True)
headers = ['%i' % cluster for cluster in list(all_df.row(0))[1:]]
headers.insert(0, 'Measure')

all_df = all_df.tail(-1)
all_df.columns = headers

all_df = all_df.with_columns(pl.when(pl.col('Measure').is_in(pop_measures))
                               .then(True).otherwise(False).alias('PerCapita'))  # noqa: FBT003

uuiddf = pl.read_csv('../NZC-Placeholders-V2/UUID-Lookup-V2.tsv', separator = '\t')
all_df = all_df.join(uuiddf, how = 'inner', left_on = 'Measure', right_on = 'MID')

all_df = all_df.fill_null(0)

# ---------------------------------------------------------------------------------------
pdf = pl.read_csv('../NZC-Placeholders-V3/Percentage-Measures.csv')

all_df = all_df.join(pdf, on = 'UUID', how = 'left')

for cluster in ['0', '1', '2', '3']:
    all_df = all_df.with_columns(pl.when(pl.col('Unit').is_not_null())
                                   .then(pl.col(cluster) * 100)
                                   .otherwise(pl.col(cluster)).alias(cluster))

all_df.select(['Measure', '0', '1', '2', '3', 'UUID', 'PerCapita']).write_parquet('placeholders.parquet')
