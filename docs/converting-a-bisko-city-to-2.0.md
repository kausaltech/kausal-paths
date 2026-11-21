# Converting a BISKO 1.3 city to the 2.0 template

*Produced by Claude Opus 5.5 on 2026-11-19.*
*Responsible: Jouni Tuomisto.*

A BISKO city on version 1.3 holds its own copy of the frozen module
(`configs/modules/bisko/model.yaml`). On 2.0 it is a member of the `bisko` framework and inherits
the published `bisko-base` template. Conversion moves a city from the first to the second and
keeps what is the city's own: its pages, its UUIDs, its data, its bindings to its own datasets and
its node settings. Starting a fresh instance instead would lose the pages, so every 1.3 city is
converted.

The cities on 1.3 are listed in `FROZEN_MODULE_INSTANCES` (`src/nodes/bisko_certification.py`).
The two certification instances, `bisko` and `pruefstadt-bisko`, stay on 1.3 for good, and
`setup_bisko` refuses them.

## What conversion does

`python -m tools.setup_bisko --convert <city>` (`convert_checked_instance` in
`src/frameworks/bisko/validation.py`):

1. Checks, before writing anything, that every copied node has the template node's ports and
   that every dimension category with data has a counterpart in the framework. All failures are
   reported together.
2. Moves the city's dimensions to the framework's. A category renamed since the city was set up
   follows the alias the framework category declares (`geothermatl` → `geothermal`). A category
   that no data point uses and the framework does not have is left behind: a database dimension
   keeps every category it ever had.
3. Turns the city's numeric `quality` scores into grades of the framework's scale (A–D), attached
   to the values they grade. That is the form the template's datasets have; a 1.3 city belongs to
   no framework, so nothing has done it before.
   Then moves each of the city's datasets onto the schema of the template dataset of the same name, and a
   replacement dataset (`<city>/final_energy`) onto the schema of the input it replaces. From then
   on the template's changes to the schema, such as a new validation rule, reach the city. The
   data has to fit: the same dimensions, and every metric that holds data in the template schema
   with the same unit. Metric order, rules and labels are the template's to define and do not
   matter.
4. Points the city's datasets at the template's shapes. A YAML instance mints shape UUIDs from its
   own identity, so its copies of the template's shapes have other UUIDs.
5. Pins the city to the published template revision, deletes the copied nodes (pages that
   referred to them now refer to the template's), keeps nodes the template does not have as the
   city's own, and keeps the city's bindings to its own datasets.
6. Deletes the city's copies of datasets the template now supplies (the `de/*` tables).
7. Assigns the municipality's organization from its AGS and seeds the population series at the
   template's pin, as a template upgrade does.
8. Refuses if the result cannot be published, naming the node and dataset of each problem.

The command is one transaction: a refusal at any step leaves the database as it was.
`--dry-run` runs everything and rolls it back.

## Before you start

- **Take a copy of the database**, e.g. `pg_dump -Fc`. A conversion cannot be undone by command.
- **The template is published and current.** `python manage.py compare_draft_published bisko-base`
  should report everything identical. If not, publish it first
  (`python -m tools.setup_bisko --sync-template`, then `--publish`).
- **The city is database-sourced.** `InstanceConfig.config_source` is `database`.
- **Record the city's outputs**, to compare with afterwards:

  ```bash
  python manage.py test_instance <city> --store --state-dir /tmp/before-conversion/
  ```

## Step 1: make the city's own model the 2.0 model

`--convert` matches each copied node to the template node of the same name, port by port, so the
city's database model must first be a copy of `bisko-base` rather than of the 1.3 module.

In `configs/<city>.yaml`, mirror the includes of `configs/bisko-base.yaml`:

```yaml
include:
- file: modules/bisko-base/population-defaults.yaml
  allow_override: false
- file: modules/bisko-base/model.yaml
  # The extension above deliberately replaces only the population node.
  allow_override: true
  dataset_replacements:   # the city's own, unchanged
  - ...
```

Then, in the same file:

- remove the parameters 2.0 does not have, such as `municipal_facilities_included_in_commerce`;
- remove `result_excels` entries naming nodes 2.0 does not have, such as
  `municipal_facilities_energy_in_commerce`;
- leave `lau_code` in place if you like: no node reads it any more, but it identifies the
  municipality.

Remove the city from `FROZEN_MODULE_INSTANCES`, so that its YAML is no longer translated as the
1.3 module's. Then sync:

```bash
python manage.py sync_instance_to_db <city>
```

From here until step 4, the data-quality nodes fail with `Unknown qualifier 'bisko_quality'`.
The qualifier exists only for framework members, which is why the cities went back to 1.3 in
November 2026. Carry on to the conversion without publishing in between.

## Step 2: import the city's own inputs

A framework member's own inputs must be in the database, where the data entry edits them. Many 1.3
cities read theirs straight from DVC, through datasets that are empty placeholders in the
database. Ask which ones:

```bash
python manage.py dataset_status <city>
```

Import the city's own datasets: the `kommune/*` ones and the city's replacements (such as
`<city>/final_energy`). Leave out:

- the `de/*` tables, which the conversion deletes because the template supplies them;
- anything graded `template`: those were filled in the admin, and an import would blank them;
- anything graded `dvc only` with an index the instance has no dimension for
  (`demography/population_lau`).

```bash
python manage.py load_dvc_dataset <city> kommune/... <city>/... --plan
python manage.py load_dvc_dataset <city> kommune/... <city>/... --force
```

`--plan` shows the values each import keeps and adds; check that values entered in the admin are
not lost.

## Step 3: dry run

```bash
python -m tools.setup_bisko --convert <city> --dry-run
```

What a refusal means:

| Message | Cause | Fix |
|---|---|---|
| `<node> has an undeclared framework port` | The city's copy of the node is not the 2.0 node. | Step 1 is incomplete: an include or an override in the city's YAML still gives the node another shape. |
| `Category vocabulary differs for <dimension>: <categories> carry data and have no counterpart` | The city has data under a category the framework does not know. | Rename the category to the framework's, or add an alias for it to the template's dimension. |
| `... would duplicate data already under <category>` | The city has data under both a renamed category and its new name, for the same cells. | Decide which value is right and delete the other data point. |
| `Input data has not been imported (<node>, <dataset>)` | A city input still reads an empty placeholder. | Import that dataset (step 2). |
| `<dataset>: quality value <x> has no unique BISKO grade or matching value point` | A score that is not one of the scale's (1, 0.5, 0.25, 0), or a score with no value beside it. | Correct the score in the city's dataset to one of the grades. |
| `<dataset>: metric '<m>' is in <unit>, the template's in <unit>` (or has no such metric, or other dimensions) | The city's data does not fit the template's schema for that dataset. | Convert or move the data in the city's dataset first; the template's schema is the one a member uses. |

## Step 4: convert

```bash
python -m tools.setup_bisko --convert <city>
```

It prints what was done: shared and local nodes, the bindings kept, the datasets removed, the
organization and the population. `local_nodes` counts the nodes the city keeps as its own because
the template does not have them. Check that each is one the city means to keep: a node 2.0 retired
stays in the city's model until the city's YAML stops declaring it and the city is synced.

District heating plant data does not carry over. Since 21 November 2026 the template reads it per
plant and heat network, from `kommune/fernwaerme_anlagen` and `kommune/fernwaerme_netze`, and no
longer reads `kommune/fernwaerme_energieeinsatz` or `kommune/kwk_anlagenparameter`. A city that
had entered plant data or CHP parameters (or replaced those datasets, as Mainz did) enters them again
in the new tables: one plant in network 1 reproduces the 1.3 result, with the net electricity
E = f / (1 − f) × Q for electricity fraction f and heat output Q.

The population series at the template's pin covers only some Länder (in November 2026 those with
AGS prefixes 03, 07 and 12). Elsewhere the command says so, and the city's population is entered
in the data entry. Until then, the per-capita figures are empty.

## Step 5: check

```bash
python manage.py test_instance <city> --compare --state-dir /tmp/before-conversion/
python -m tools.debug_instance -i <city> --node net_emissions
python -m tools.debug_instance -i <city> --node total_weighted_data_quality
```

Expect every node identical except the data-quality nodes, which now compute. On `testcity-bisko`,
107 of 110 nodes were identical and the other three were those. A difference elsewhere is a
difference between 1.3 and 2.0, or between the city's data in DVC and in the database; find which
before publishing.

If the city has been published, its visitors still see the 1.3 revision. Compare and publish:

```bash
python manage.py compare_draft_published <city>
```

## After the conversion

The city's model now comes from the template. Template changes reach it through
`setup_bisko --sync-template` and `--publish`, which advances every city on the template; its
own YAML no longer matters to it. `sync_instance_to_db` refuses it by name, saying what to run
instead, and `--all` and `--all-db` skip framework members.

Where the template reads a national table by AGS (the IFEU road mileage and the other-transport
energy), a municipality the table does not cover gets an empty input instead of an error, and the
nodes concerned show as degraded, naming the AGS. The city cannot fix that itself; its own traffic
and consumption data replace the defaults.

## Cities converted before 20 November 2026

Until then, conversion moved a dataset onto the template's schema only if the two were identical
in every detail. A graded dataset never was, because the grade names the metric it grades by an
identifier unique to each schema, so a converted city kept its own copy of most of its schemas and
missed the template's later changes to them. Repair such a city in place:

```bash
python -m tools.setup_bisko --adopt-schemas <city> --dry-run
python -m tools.setup_bisko --adopt-schemas <city>
```

It moves the datasets, remaps the bindings that refer to the old metrics, and deletes the old
schemas once nothing uses them. Model outputs do not change. A second run reports nothing to move.

## In production

The order is the same, with the code deployed first:

1. Deploy the YAML change and the `FROZEN_MODULE_INSTANCES` change.
2. On the production database: `sync_instance_to_db`, the imports, the dry run and the
   conversion, as above.
3. Publish once the comparison is understood.

The YAML change does nothing to the city until the sync, so it can be deployed ahead of time.
