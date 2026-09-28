# BISKO administrative organizations

The source parser lives in `kausal-importers`. Paths owns the database write,
because `Organization` is a treebeard tree and existing model instances may
already point at municipality nodes.

## Import

1. Apply the Paths migrations and run `python -m tools.setup_bisko`. That
   provisions the BISKO framework, German organization classes, and the `ars`,
   `ags`, and `nuts3` identifier namespaces. Repeating setup preserves existing
   names. Once districts have been imported, setup checks that each has a
   NUTS3 identifier; if one is missing, complete steps 2 and 3, then rerun setup.
2. Generate the staged Parquet file from the dated BKG workbook:

   ```bash
   cd ../kausal-importers
   uv run kausal-importers germany administrative-divisions \
     --source tmp/bisko_import/vg250-ew_12-31.ee.excel.ebenen.zip \
     --vintage 2024-12-31 \
     --output /tmp/bkg-organizations-2024.parquet
   ```

3. From Paths, inspect and apply it:

   ```bash
   python manage.py import_bkg_organizations /tmp/bkg-organizations-2024.parquet --dry-run
   python manage.py import_bkg_organizations /tmp/bkg-organizations-2024.parquet
   PYTHONPATH=. python tools/setup_bisko.py
   ```

The staged Parquet must include `nuts3`, produced by the updated importer from
the workbook's district `NUTS` and municipality `NUTS3_CODE` fields. Old Parquet
files are rejected. The Paths import verifies that each municipality's code
matches its district and stores one `nuts3` identifier on each Landkreis or
district-free city. Activation resolves it through the municipality's district
ancestor and writes the instance's `nuts_code` parameter. A repeat import with
current administrative information creates only missing identifiers; it does
not save unchanged organizations. Rerunning setup then reconciles `nuts_code`
for existing BISKO municipal instances, fills wholly empty municipal
`kommune/witterungsbereinigung` datasets, and adds missing blank cells for the
current inventory year. A further setup run leaves them unchanged. Weather
defaults use the municipality's district NUTS3 Eurostat
heating-degree-day series from the template's pinned dataset repository:
the 1980–2014 mean divided by each year's degree days for households, commerce,
and municipal facilities; industry and transport receive 1. The dataset records
the source revision and region. Existing municipal weather values are preserved,
including partially filled datasets.

The command reconciles existing rows by ARS and municipality AGS, preserves
their UUIDs and instance links, moves them under their BKG parent, and attaches
the three state roots to BISKO. It preserves locally maintained names and
reports how many differ from the source. Conflicting identifiers or
classifications abort the transaction. Missing rows are never deleted. Keep
the dated source file and staged Parquet file as import evidence; the
`Organization` model does not itself store a source vintage.

## Delegated access

`OrganizationAccessGrant` attaches a viewer, editor, or admin role to an
existing user, a framework, and one organization. The role applies to the
organization and its descendants inside the framework's attached roots. It
does not create a `FrameworkConfig` or `InstanceConfig` for each municipality.

```bash
python manage.py grant_organization_access username 03 --role viewer --dry-run
python manage.py grant_organization_access username 03001 --role editor
```

Publish the BISKO template before activating any municipality. Activation pins
each new instance to that published revision; it does not copy the template's
nodes or datasets into a local model. To provision 12 password-login test
accounts and activate their three selected municipalities after importing the
BKG tree, set `BISKO_TEST_ACCOUNT_PASSWORD` in the environment and run:

```bash
python manage.py provision_bisko_test_accounts --dry-run
python manage.py provision_bisko_test_accounts
```

The command creates two state accounts (admin and editor), one Landkreis
editor, and one municipality editor in each of Niedersachsen (03),
Rheinland-Pfalz (07), and Brandenburg (12). It chooses the first Landkreis by
ARS that contains a municipality and grants the municipality account access
inside that Landkreis. The emails are `bisko-<state>-<level>.fake@kausal.tech`.
They are the login names; the command also prints each internal username and
ARS. It also prints the instance identifier and framework config UUID for each
selected municipality. Repeating the command keeps the same accounts, grants,
and instances and resets their
passwords to the supplied value. A changed, suspended, or extra grant requires
explicit resolution. The password is never printed. The command creates users
in the Paths database; a deployment using an external identity provider also
needs matching accounts there for password login through that provider.

Activation gives the municipality its own `kommune/` datasets against
the shared BISKO schemas. The weather dataset receives editable regional
defaults; other municipal datasets receive blank cells for the current inventory
year where their schema or the template declares a row layout. Example values
from the template are never copied. Editable municipal input
bindings point to those local dataset UUIDs; the `de/` method and reference
inputs stay shared. It
sets `ags_number` and `lau_code` from the municipality's AGS, enables account
management when BISKO enables it, and opens one draft submission for the most
recent historical year. A draft with empty municipal inputs is deliberately
incomplete; do not treat template demonstration values as the town's balance.

`instanceEditor.beginInventoryYear(year)` opens the next inventory year, adds
blank cells to editable instance-owned annual datasets, and advances the
instance's last historical year. Framework instances also receive a draft
inventory submission; standalone database-backed instances do not. Existing
values are retained. Required-combination validation reports missing values in
the newly represented year; optional inputs can remain blank.

For municipalities activated before local inputs were provisioned, reconcile
their existing instances without resetting account passwords:

```bash
python manage.py reconcile_bisko_municipalities \
  bisko-03151009 bisko-07131007 bisko-12060005 --dry-run
python manage.py reconcile_bisko_municipalities \
  bisko-03151009 bisko-07131007 bisko-12060005
```

The command creates missing datasets, binding overrides, and a draft submission;
it preserves existing municipal data points, local binding choices, and
submissions. The dry run rolls back its database writes. These test accounts
hold regional `OrganizationAccessGrant` roles. `Instance.users` lists direct
municipal memberships only, so it remains empty until a municipal admin adds
or invites one; regional grant holders do not consume municipal seats.
An instance admin can query `Instance.inheritedOrganizationGrants` for the
regional accounts that cover the municipality, including their email, role,
and grant root. `memberSeatLimit` and `memberSeatsInUse` describe the separate
municipal roster.

The GraphQL `framework.organizations` field returns only accessible
organizations, with `parentId`, `search`, `first`, and `offset` arguments.
`search` filters the direct children when `parentId` is supplied; without a
parent it searches the accessible subtree. `framework.organization(id: ...)`
retrieves one organization by UUID if the caller has access. Each organization
reports `municipalityCount`, `activatedMunicipalityCount`, and
`unactivatedMunicipalityCount` for its whole subtree (including itself if it
is a municipality). The AGS-bearing municipality counts once, including the
municipality child of a district-free city. Activated
means that a BISKO `FrameworkConfig` exists; it says nothing about whether the
municipality has entered data or finalized a submission. These counts are
aggregated in the database, so a UI need not page through thousands of towns.
Each row has `instanceIdentifier`, which is null until activation. A UI can
offer activation on an accessible municipality without an instance using
`activateFrameworkOrganization(frameworkId: "bisko", organizationId: ...)`.
The mutation requires edit access to that municipality, returns the instance
identifier and framework config UUID, and is repeatable. It rejects nonmunicipal
organizations, missing AGS identifiers, and activation before template
publication. Existing model instances without published-template inheritance
need explicit conversion; activation will not silently replace them.
`FrameworkConfig.organizationId` links back to the direct organization, and
`Framework.configs(organizationId: ...)` returns configs under that organization
or any descendant, filtered by the caller's access. Population remains a
separate data input; the instance's `lau_code` now selects its own LAU from the
shared population dataset.
The local `kommune/` datasets start empty. Until municipal data is entered or
imported, the balance has missing inputs and must be shown as incomplete; the
template's demonstration values are never used as town-specific totals.

```graphql
mutation Activate($organization: ID!) {
  activateFrameworkOrganization(frameworkId: "bisko", organizationId: $organization) {
    ... on ActivateOrganizationResult {
      frameworkConfigId
      instanceIdentifier
      created
    }
  }
}
```

The current Data Studio municipality picker still reads `framework.configs`,
which lists only provisioned model instances. Its loader must switch to the
organization query for browsing municipalities without model instances.

## Organization population

After importing the BKG tree, project the pinned `demography/population_lau`
dataset from the BISKO template onto its AGS-bearing organizations:

```bash
python manage.py migrate
python manage.py import_organization_population --dry-run
python manage.py import_organization_population
```

The projection stores one provider observation per municipality and year with
the template's dataset revision. Repeating the import replaces the projection
atomically; it does not create a `FrameworkConfig` for a Landkreis or Land.
When the pinned population dataset changes, rerun the command. The source
dataset remains authoritative, and the same `DE_<AGS>` key is used by the
municipal model's population node.

`FrameworkOrganization.population(year:)` returns `value` only when every
municipality in that organization's subtree has an observation for the year.
`partialValue`, `observedMunicipalityCount`, and `municipalityCount` show the
sum and coverage when some are missing. `sourceRevision` identifies the
provider snapshot. The request loads a year's observations once and reuses
their rollups across organization fields, including a page of districts.
`FrameworkConfig.population(year:)` delegates to the same organization value
for the municipality switcher.
Missing years are never filled with zero or copied from adjacent years.

Active organization grants also reach provisioned municipal instances and their
datasets under the grant's subtree. An instance membership suspension blocks
that instance even if a wider organization grant would otherwise allow it.
The GraphQL mutations `assignOrganizationRole`, `suspendOrganizationRole`,
and `reactivateOrganizationRole` manage grants. Framework and active subtree
admins may use them within their scope.
