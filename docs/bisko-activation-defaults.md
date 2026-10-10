# BISKO activation defaults

Population is an editable municipal input (`kommune/bevoelkerung`) in a population section.
`OrganizationPopulation` remains the provider reference for territory views, including
municipalities without instances. Municipal per-capita plausibility checks use the editable
population dataset; territory references continue to describe provider values.

The population extension is included explicitly by `configs/bisko.yaml`; the shared BISKO module
retains its existing population definition for other city YAML models. The template prefers local
population per year, with GISCO as the fallback while its municipal table is empty. Activated
municipalities receive the complete prepared series.

Activation reads `bisko/defaults/population` and `bisko/defaults/stationary_energy` from the DVC commit
in the municipality's **published template revision**. Weather defaults use that pin too. Both source datasets and both municipal input slots are required;
activation fails atomically if the published template or its data pin has not been upgraded.

## Prepare sources

Export the same carrier prior used by `load_bisko_demo` from the database containing the Mainz
reference inventory. The reference instance may be inactive; exporting does not change it.

```sh
python manage.py load_bisko_demo --export-prior /tmp/mainz-prior.parquet
python manage.py prepare_bisko_defaults \
  --history /path/to/germany-population-2000-2025.parquet \
  --forecasts /path/to/germany-population-forecasts.parquet \
  --geography /path/to/bkg-organizations-2024.parquet \
  --energy /path/to/stationary-estimates-2023.parquet \
  --prior /tmp/mainz-prior.parquet \
  --output-dir /tmp/bisko-defaults --horizon 2045
```

Preparation writes three Parquet files, companion `.metadata.yaml` files containing DVC `meta`
blocks, and an energy exclusion list. It refuses existing outputs and does not write to the DB.

Population defaults retain observations from 2000 onwards for the supplied geography. Historical
boundary changes are not retrospectively reconciled: missing observations for a municipality
fail preparation. Variants are `trend` (Niedersachsen), `projection` (Rheinland-Pfalz) and `middle`
(Brandenburg). Published years are linearly interpolated, then held constant beyond the final
year. Association and combined-municipality forecasts use member shares from their latest
common observed year; integer allocation conserves the aggregate. Municipalities without a
forecast hold their latest observation, labelled as an estimate. Internal historical gaps are
also interpolated and labelled; nothing is invented before the first observation. Provenance
retains source hashes, editions, variants and allocation methods.

Energy preparation excludes the **whole municipality** if any source row has `needs_review`.
The October 2026 input excludes 123 municipalities, leaving 3,530. Exclusion affects energy
defaults only; activation, population and weather remain available.

Electricity uses the grid-delivery estimate. Each sector's remainder is distributed among other
carriers using the Mainz 2018 prior proportions, preserving its total. These synthetic provider
defaults receive **C** (regional statistics, score 0.25), including cells whose upstream total
came from a published balance. The demo's invented grades are not reused. A zero prior share
produces a zero default, not an assertion of measured absence.

Grid deliveries exclude onsite generation. Transferring Mainz's carrier proportions does not
establish the local carrier mix. Stored empirical ranges describe the upstream quantity, not
individual carrier cells. The staging importer's validation and limitations still apply; the
input, source and prior hashes remain in the provenance.

## Release and refresh

Before syncing the changed template, register the three prepared files in DVC:

- `bisko/defaults/population.parquet`: annual municipality population and provenance;
- `bisko/defaults/stationary_energy.parquet`: eligible sector-by-carrier defaults;
- `kommune/bevoelkerung.parquet`: empty editable template input.

Run `dvc add` for each and merge its companion `.metadata.yaml` `meta` block into the generated
`.parquet.dvc` manifest. Store the data objects and publish the repository commit through the
normal release workflow; preparation does neither. Pin the template to that commit and sync its
datasets/model. Then preview and import the population references:

```sh
python manage.py import_organization_population --dry-run
python manage.py import_organization_population
```

The refresh uses `bisko/defaults/population` when available, retaining GISCO for old pins. Template
publication rejects population-reference revisions differing from its dataset pin. Publishing
does not advance dependent instances automatically.

New activations receive local defaults. Upgrade existing municipalities' template pins first,
then preview and apply:

```sh
python manage.py refresh_bisko_defaults --instance bisko-<ags>
python manage.py refresh_bisko_defaults --instance bisko-<ags> --apply
```

Without `--instance`, this visits all BISKO configurations. Locked instances and old template
pins are skipped. Population refresh changes only untouched provider-owned cells and retains
new provider references alongside municipal overrides in dataset metadata. Energy is seeded
once into an untouched empty grid. Final submissions retain their immutable dataset revisions.

Defaults keep commerce and municipal facilities separate, so the template's overlap parameter
is false. Utility data that includes facilities in commerce must explicitly declare the overlap.
Existing local parameter overrides are not reset by a reference refresh.

Old published templates must be upgraded before activation can succeed. Backend changes and
prepared files alone do not migrate or publish those revisions.
