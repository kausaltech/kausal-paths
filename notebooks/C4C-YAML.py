import os
import polars as pl

os.chdir('/Users/jouni/devel/climate4cast')

yamlparameters = {'instance': 'potsdam',
                  'language': 'de',
                  'commit'  : 'a34aed40faff06ea4062650c382443200bae8fe3',
                  'maxyear' : '2022',
                  'theme'   : 'default',
                  'title'   : 'Potsdam Zielwerte Masterplan',
                  'forecast_from': '2023'}

yamlfile = open('%s_c4c.yaml' % yamlparameters['instance'], 'w')

df = pl.read_csv('Potsdam-Parquet.csv')

years = df.select('Year').unique().to_series(0).to_list()
yamlparameters['targetyear']   = '%i' % max(years)
yamlparameters['endyear']      = '%i' % max(years)
yamlparameters['minyear']      = '%i' % min(years)
yamlparameters['forecastyear'] = '%i' % (int(yamlparameters['maxyear']) + 1)
yamlparameters['Instance']     = yamlparameters['instance'].title()

# -------------------------------------------------------------------------------------------------
def makeid(name: str):
    return (name.lower().replace('.', '').replace(',', '').replace(':', '').replace('-', '').replace(' ', '_')
            .replace('&', 'and').replace('å', 'a').replace('ä', 'a').replace('ö', 'o'))

def dimlist(df):
    dimlist = []
    for dim in list(set(df.columns) - set(['Sector', 'Sector name', 'Quantity', 'Unit', 'Value', 'Year'])):
        if df.select(dim).unique().to_series(0).to_list() != [None]:
            dimlist.append(makeid(dim))

    return dimlist

# -------------------------------------------------------------------------------------------------
levellookup = {
               3: {'I.1.1'  : 'Residential Fuel',
                   'I.1.2'  : 'Residential Grid Energy',
                   'I.1.3'  : 'Residential T & D Loss',
                   'I.2.1'  : 'Commercial & Institutional Fuel',
                   'I.2.2'  : 'Commercial & Institutional Grid Energy',
                   'I.2.3'  : 'Commercial & Institutional T & D Loss',
                   'I.3.1'  : 'Manufacturing Fuel',
                   'I.3.2'  : 'Manufacturing Grid Energy',
                   'I.3.3'  : 'Manufacturing T & D Loss',
                   'I.4.1'  : 'Energy Fuel',
                   'I.4.2'  : 'Energy Grid Energy',
                   'I.4.3'  : 'Energy T & D Loss',
                   'I.5.1'  : 'Ag, Forestry, & Fishing Fuel',
                   'I.5.2'  : 'Ag, Forestry, & Fishing Grid Energy',
                   'I.5.3'  : 'Ag, Forestry, & Fishing T & D Loss',
                   'I.6.1'  : 'Non-Specified Fuel',
                   'I.6.2'  : 'Non-Specified Grid Energy',
                   'I.6.3'  : 'Non-Specified T & D Loss',
                   'I.7.1'  : 'Fugitive Coal',
                   'I.8.1'  : 'Fugitive Oil & Natural Gas',
                   'II.1.1' : 'On-Road Transport Fuel',
                   'II.1.2' : 'On-Road Transport Grid Energy',
                   'II.1.3' : 'On-Road Transport Outside City, T & D Loss',
                   'II.2.1' : 'Railway Fuel',
                   'II.2.2' : 'Railway Grid Energy',
                   'II.2.3' : 'Railway Outside City, T & D Loss',
                   'II.3.1' : 'Waterborne Fuel',
                   'II.3.2' : 'Waterborne Grid Energy',
                   'II.3.3' : 'Waterborne Outside City, T & D Loss',
                   'II.4.1' : 'Aviation Fuel',
                   'II.4.2' : 'Aviation Grid Energy',
                   'II.4.3' : 'Aviation Outside City, T & D Loss',
                   'II.5.1' : 'Off-Road Transport Fuel',
                   'II.5.2' : 'Off-Road Transport Grid Energy',
                   'II.5.3' : 'Off-Road Transport Outside City, T & D Loss',
                   'III.1.1': 'Solid Waste Disposed in City',
                   'III.1.2': 'Solid Waste Disposed outside City',
                   'III.2.1': 'Biological Waste Treated in City',
                   'III.2.2': 'Biological Waste Treated outside City',
                   'III.3.1': 'Incineration Treated in City',
                   'III.3.2': 'Incineration Treated outside City',
                   'III.4.1': 'Wastewater Treated in City',
                   'III.4.2': 'Wastewater Treated outside City'},
               2: {'I.1'  : 'Residential',
                   'I.2'  : 'Commercial & Institutional',
                   'I.3'  : 'Manufacturing',
                   'I.4'  : 'Energy',
                   'I.5'  : 'Ag, Forestry, & Fishing',
                   'I.6'  : 'Non-Specified',
                   'I.7'  : 'Fugitive Coal',
                   'I.8'  : 'Fugitive Oil & Natural Gas',
                   'II.1' : 'On-Road Transport',
                   'II.2' : 'Railway',
                   'II.3' : 'Waterborne',
                   'II.4' : 'Aviation',
                   'II.5' : 'Off-Road Transport',
                   'III.1': 'Solid Waste',
                   'III.2': 'Biological Waste',
                   'III.3': 'Incineration',
                   'III.4': 'Wastewater',
                   'IV.1' : 'Industrial Processes',
                   'IV.2' : 'Product Use',
                   'V.1'  : 'Livestock',
                   'V.2'  : 'Land',
                   'V.3'  : 'Aggregate Sources'},
               1: {'I'  : 'Stationary',
                   'II' : 'Transportation',
                   'III': 'Waste',
                   'IV' : 'IPPU',
                   'V'  : 'AFOLU'}
              }

quantitylookup = {'Emission Factor'   : {'name': 'Emission Factor', 'quantity': 'emission_factor'},
                  'Emissions'         : {'name': 'Emissions'      , 'quantity': 'emissions'},
                  'Energy Consumption': {'name': 'Consumption'    , 'quantity': 'energy'},
                  'Fuel Consumption'  : {'name': 'Combustion'     , 'quantity': 'fuel_consumption'},
                  'Mileage'           : {'name': 'Mileage'        , 'quantity': 'mileage'},
                  'Price'             : {'name': 'Price'          , 'quantity': 'currency'},
                  'Unit Price'        : {'name': 'Unit Price'     , 'quantity': 'unit_price'}}

activityquantities = set(['Energy Consumption', 'Fuel Consumption', 'Mileage'])

# -------------------------------------------------------------------------------------------------
nodetemplates = {
                 'dataset': ('- id: [nodeid]\n' +
                             '  name: [name]\n' +
                             '  type: gpc.DatasetNode\n' +
                             '  quantity: [quantity]\n' +
                             '  unit: [unit]\n' +
                             '  input_dimensions: [dimlist]\n' +
                             '  output_dimensions: [dimlist]\n' +
                             '  params:\n' +
                             '    gpc_sector: [gpcid]\n' +
                             '  input_datasets:\n' +
                             '  - id: gpc/[instance]\n' +
                             '    forecast_from: [forecast_from]\n'),

                 'simple': ('- id: [nodeid]\n' +
                            '  name: [name]\n' +
                            '  type: simple.[nodetype]Node\n' +
                            '  quantity: [quantity]\n' +
                            '  unit: [unit]\n' +
                            '  input_dimensions: [dimlist]\n' +
                            '  output_dimensions: [dimlist]\n'),

                 'emissions': ('emission_sectors:\n' +
                               '- id: [nodeid]\n' +
                               '  name: [name]\n' +
                               '  is_outcome: true\n' +
                               '\n\n' +
                               'nodes:\n')
                }

instanceyaml = ('id: [instance]_c4c\n' +
                'default_language: [language]\n' +
                'supported_languages: []\n' +
                'site_url: https://[instance]-c4c.paths.staging.kausal.tech\n' +
                'dataset_repo:\n' +
                '  url: https://github.com/kausaltech/dvctest.git\n' +
                '  commit: [commit]\n' +
                '  dvc_remote: kausal-s3\n' +
                'name: [Instance] BASIC+ Greenhouse Gas Inventory\n' +
                'owner: City of [Instance]\n' +
                'theme_identifier: [theme]\n' +
                'target_year: [targetyear]\n' +
                'model_end_year: [endyear]\n' +
                'minimum_historical_year: [minyear]\n' +
                'maximum_historical_year: [maxyear]\n' +
                'emission_unit: [unit]\n' +
                'emission_forecast_from: [forecastyear]\n' +
                'emission_dimensions: [dimlist]\n' +
                'features:\n' +
                '  baseline_visible_in_graphs: true\n\n')

pageyaml = ('pages:\n' +
            '- id: home\n' +
            '  name: [Instance] BASIC+ Greenhouse Gas Inventory\n' +
            '  path: /\n' +
            '  type: emission\n' +
            '  outcome_node: net_emissions\n' +
            '  lead_title: [title]\n' +
            '  lead_paragraph: GPC BASIC+ greenhouse gas inventory ([minyear]-[maxyear]) and forecast ([forecastyear]-[targetyear]) for the City of [Instance].\n\n' +
            'scenarios:\n' +
            '- id: baseline\n' +
            '  name: Business as Usual\n\n' +
            '- id: default\n' +
            '  default: true\n' +
            '  name: Climate Action Plan\n' +
            '  all_actions_enabled: true\n')

# -------------------------------------------------------------------------------------------------
terminal = {'e': 'Net Emissions', 'p': 'Net Price'}
suffix = {'e': 'Total Emissions', 'p': 'Total Price'}

enodes = {makeid(terminal['e']): {'template': 'emissions',
                                  'name': terminal['e'],
                                  'upstream': []}}
pnodes = {makeid(terminal['p']): {'template': 'simple',
                                  'name': terminal['p'],
                                  'nodetype': 'Additive',
                                  'quantity': quantitylookup['Price']['quantity'],
                                  'upstream': []}}

for level in [3, 2, 1]:
    levellist = list(levellookup[level].keys())
    levellist.sort()
    for GPCID in levellist:
        prefix = levellookup[level][GPCID]

        enodename = '%s %s %s' % (GPCID, prefix, suffix['e'])
        enodeid = makeid(enodename)
        pnodename = '%s %s %s' % (GPCID, prefix, suffix['p'])
        pnodeid = makeid(pnodename)

        if level == 1:
            edown = makeid(terminal['e'])
            pdown = makeid(terminal['p'])
        else:
            downid = GPCID.rsplit('.', 1)[0]
            edown = makeid('%s %s %s' % (downid, levellookup[level - 1][downid], suffix['e']))
            pdown = makeid('%s %s %s' % (downid, levellookup[level - 1][downid], suffix['p']))

        enodes[enodeid] = {'template': 'simple',
                           'name': enodename,
                           'nodetype': 'Additive',
                           'quantity': quantitylookup['Emissions']['quantity'],
                           'upstream': [],
                           'downstream': [edown]}

        pnodes[pnodeid] = {'template': 'simple',
                           'name': pnodename,
                           'nodetype': 'Additive',
                           'quantity': quantitylookup['Price']['quantity'],
                           'upstream': [],
                           'downstream': [pdown]}


# -------------------------------------------------------------------------------------------------
leafnodes = {}

for level in [3, 2, 1]:
    levellist = list(levellookup[level].keys())
    levellist.sort()
    for GPCID in levellist:
        gpcdf = df.filter(pl.col('Sector') == GPCID)

        if not gpcdf.is_empty():
            prefix = levellookup[level][GPCID]
            edown = makeid('%s %s %s' % (GPCID, prefix, suffix['e']))
            pdown = makeid('%s %s %s' % (GPCID, prefix, suffix['p']))

            frames = {}
            for quantity in quantitylookup:
                qdf = gpcdf.filter(pl.col('Quantity') == quantity)
                if not qdf.is_empty():
                    frames[quantity] = qdf

            activities = list(set(frames.keys()) & activityquantities)

            # -------------------------------------------------------------------------------------
            if (len(activities) == 1) & ('Emission Factor' in list(frames.keys())):
                activity = activities[0]
                upstream = []
                enodename = '%s %s %s Emissions' % (GPCID, prefix, quantitylookup[activity]['name'])
                enodeid = makeid(enodename)

                for quantity in [activity, 'Emission Factor']:
                    qnodename = '%s %s %s' % (GPCID, prefix, quantitylookup[quantity]['name'])
                    qnodeid = makeid(qnodename)
                    upstream.append(qnodeid)

                    leafnodes[qnodeid] = {'template': 'dataset',
                                          'name': qnodename,
                                          'quantity': quantitylookup[quantity]['quantity'],
                                          'unit': frames[quantity].item(0, 'Unit'),
                                          'dimlist': dimlist(frames[quantity]),
                                          'gpcid': GPCID,
                                          'instance': yamlparameters['instance'],
                                          'downstream': [enodeid]}


                leafnodes[enodeid] = {'template': 'simple',
                                      'name': enodename,
                                      'nodetype': 'Multiplicative',
                                      'quantity': quantitylookup['Emissions']['quantity'],
                                      'unit': '%s/%s' % (leafnodes[upstream[1]]['unit'].split('/')[0],
                                                         leafnodes[upstream[0]]['unit'].split('/')[1]),
                                      'dimlist': list(set(leafnodes[upstream[0]]['dimlist']) |
                                                      set(leafnodes[upstream[1]]['dimlist'])),
                                      'upstream': upstream,
                                      'downstream': [edown]}
                enodes[edown]['upstream'].append(enodeid)

            # -------------------------------------------------------------------------------------
            if ('Emissions' in list(frames.keys())):
                enodename = '%s %s %s' % (GPCID, prefix, quantitylookup['Emissions']['name'])
                enodeid = makeid(enodename)

                leafnodes[enodeid] = {'template': 'dataset',
                                      'name': enodename,
                                      'quantity': quantitylookup['Emissions']['quantity'],
                                      'unit': frames['Emissions'].item(0, 'Unit'),
                                      'dimlist': dimlist(frames['Emissions']),
                                      'gpcid': GPCID,
                                      'instance': yamlparameters['instance'],
                                      'downstream': [edown]}
                enodes[edown]['upstream'].append(enodeid)

            # -------------------------------------------------------------------------------------
            if (len(activities) == 1) & ('Unit Price' in list(frames.keys())):
                activity = activities[0]
                upstream = []
                pnodename = '%s %s %s Price' % (GPCID, prefix, quantitylookup[activity]['name'])
                pnodeid = makeid(pnodename)

                for quantity in [activity, 'Unit Price']:
                    qnodename = '%s %s %s' % (GPCID, prefix, quantitylookup[quantity]['name'])
                    qnodeid = makeid(qnodename)
                    upstream.append(qnodeid)

                    if qnodeid in leafnodes:
                        leafnodes[qnodeid]['downstream'].append(pnodeid)
                    else:
                        leafnodes[qnodeid] = {'template': 'dataset',
                                              'name': qnodename,
                                              'quantity': quantitylookup[quantity]['quantity'],
                                              'unit': frames[quantity].item(0, 'Unit'),
                                              'dimlist': dimlist(frames[quantity]),
                                              'gpcid': GPCID,
                                              'instance': yamlparameters['instance'],
                                              'downstream': [pnodeid]}

                leafnodes[pnodeid] = {'template': 'simple',
                                      'name': pnodename,
                                      'nodetype': 'Multiplicative',
                                      'quantity': quantitylookup['Price']['quantity'],
                                      'unit': '%s/%s' % (leafnodes[upstream[1]]['unit'].split('/')[0],
                                                         leafnodes[upstream[0]]['unit'].split('/')[1]),
                                      'dimlist': list(set(leafnodes[upstream[0]]['dimlist']) |
                                                      set(leafnodes[upstream[1]]['dimlist'])),
                                      'upstream': upstream,
                                      'downstream': [pdown]}
                pnodes[pdown]['upstream'].append(pnodeid)

            # -------------------------------------------------------------------------------------
            if ('Price' in list(frames.keys())):
                pnodename = '%s %s %s' % (GPCID, prefix, quantitylookup['Price']['name'])
                pnodeid = makeid(pnodename)

                leafnodes[pnodeid] = {'template': 'dataset',
                                      'name': pnodename,
                                      'quantity': quantitylookup['Price']['quantity'],
                                      'unit': frames['Price'].item(0, 'Unit'),
                                      'dimlist': dimlist(frames['Price']),
                                      'gpcid': GPCID,
                                      'instance': yamlparameters['instance'],
                                      'downstream': [pdown]}
                pnodes[pdown]['upstream'].append(pnodeid)

# -------------------------------------------------------------------------------------------------
for level in [3, 2, 1]:
    levellist = list(levellookup[level].keys())
    levellist.sort()
    for GPCID in levellist:
        prefix = levellookup[level][GPCID]

        for nodeset in [[enodes, 'e'], [pnodes, 'p']]:
            nname = '%s %s %s' % (GPCID, prefix, suffix[nodeset[1]])
            nid = makeid(nname)

            if nodeset[0][nid]['upstream'] == []:
                del nodeset[0][nid]
            else:
                upstream = nodeset[0][nid]['upstream']
                downstream = nodeset[0][nid]['downstream'][0]

                if upstream[0] in leafnodes:
                    unit = leafnodes[upstream[0]]['unit']
                    dlist = leafnodes[upstream[0]]['dimlist']
                else:
                    unit = nodeset[0][upstream[0]]['unit']
                    dlist = nodeset[0][upstream[0]]['dimlist']

                for i in range(1, len(upstream)):
                    if upstream[i] in leafnodes:
                        dlist = list(set(dlist) & set(leafnodes[upstream[i]]['dimlist']))
                    else:
                        dlist = list(set(dlist) & set(nodeset[0][upstream[i]]['dimlist']))

                nodeset[0][nid]['unit'] = unit
                nodeset[0][nid]['dimlist'] = dlist
                nodeset[0][downstream]['upstream'].append(nid)

# -------------------------------------------------------------------------------------------------
for nodeset in [[enodes, 'e'], [pnodes, 'p']]:
    nid = makeid(terminal[nodeset[1]])

    if nodeset[0][nid]['upstream'] == []:
        del nodeset[0][nid]
    else:
        upstream = nodeset[0][nid]['upstream']
        nodeset[0][nid]['unit'] = nodeset[0][upstream[0]]['unit']

        dlist = nodeset[0][upstream[0]]['dimlist']
        for i in range(1, len(upstream)):
            dlist = list(set(dlist) & set(nodeset[0][upstream[i]]['dimlist']))
        nodeset[0][nid]['dimlist'] = dlist

# -------------------------------------------------------------------------------------------------
yaml = instanceyaml
for attribute in yamlparameters:
    yaml = yaml.replace('[%s]' % attribute, str(yamlparameters[attribute]))

for attribute in ['unit', 'dimlist']:
    yaml = yaml.replace('[%s]' % attribute, str(enodes[makeid(terminal['e'])][attribute]))

yamlfile.write(yaml)

# -------------------------------------------------------------------------------------------------
yamlfile.write('dimensions:\n')
dims = list(set(df.columns) - set(['Sector', 'Sector name', 'Quantity', 'Unit', 'Value', 'Year']))
for dim in dims:
    cats = df.select(pl.col(dim)).filter(pl.col(dim).is_null().not_()).unique().to_series(0).to_list()

    if cats != []:
        yamlfile.write('- id: %s\n  label: %s\n  categories:\n' % (makeid(dim), dim))

        cats.sort()
        for cat in cats:
            yamlfile.write("  - id: %s\n    label: '%s'\n" % (makeid(cat), cat))
        yamlfile.write('\n')

# -------------------------------------------------------------------------------------------------
eid = makeid(terminal['e'])

yaml = nodetemplates['emissions'].replace('[nodeid]', eid)
for attribute in enodes[eid]:
    yaml = yaml.replace('[%s]' % attribute, str(enodes[eid][attribute]))

yamlfile.write(yaml)

pid = makeid(terminal['p'])
if pid in pnodes:
    yaml = nodetemplates[pnodes[pid]['template']].replace('[nodeid]', pid)
    for attribute in pnodes[pid]:
        yaml = yaml.replace('[%s]' % attribute, str(pnodes[pid][attribute]))

    yaml += '  is_outcome: true\n\n'
    yamlfile.write(yaml)

# -------------------------------------------------------------------------------------------------
for nodeset in [leafnodes, enodes, pnodes]:
    nodeids = list(set(nodeset.keys()) - set([eid, pid]))
    nodeids.sort()
    for nid in nodeids:
        yaml = nodetemplates[nodeset[nid]['template']].replace('[nodeid]', nid)

        for attribute in nodeset[nid]:
            yaml = yaml.replace('[%s]' % attribute, str(nodeset[nid][attribute]))
        yaml +=  '  output_nodes:\n'

        # For each downstream node...
        for downid in nodeset[nid]['downstream']:
            yaml += '  - id: %s\n' % downid

            # Find the downstream node's nodeset...
            downset = False
            for ns in [leafnodes, enodes, pnodes]:
                if downid in ns:
                    downset = ns

            # Then check whether each dimension is present in the downstream node...
            dlist = [[], []]
            for dim in nodeset[nid]['dimlist']:
                if dim in downset[downid]['dimlist']:
                    dlist[1].append(dim)
                else:
                    dlist[0].append(dim)

            # If the dimension isn't present, flatten...
            if len(dlist[0]) > 0:
                yaml += '    from_dimensions:\n'
                for dim in dlist[0]:
                    yaml += '    - id: %s\n      flatten: true\n' % dim

            # And if the dimension is present, keep.
            if len(dlist[1]) > 0:
                yaml += '    to_dimensions:\n'
                for dim in dlist[1]:
                    yaml += '    - id: %s\n' % dim

        yamlfile.write(yaml + '\n')

# -------------------------------------------------------------------------------------------------
yaml = pageyaml
for attribute in yamlparameters:
    yaml = yaml.replace('[%s]' % attribute, str(yamlparameters[attribute]))

yamlfile.write(yaml)
yamlfile.close()
