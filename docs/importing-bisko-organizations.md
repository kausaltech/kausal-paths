# BISKO administrative organizations

The source parser lives in `kausal-importers`. Paths owns the database write,
because `Organization` is a treebeard tree and existing model instances may
already point at municipality nodes.

## Import

1. Apply the Paths migrations and run `python -m tools.setup_bisko`. That
   provisions the BISKO framework, German organization classes, and the `ars`
   and `ags` identifier namespaces. Repeating setup preserves existing names.
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
   ```

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

To provision 12 password-login test accounts after importing the BKG tree,
set `BISKO_TEST_ACCOUNT_PASSWORD` in the environment and run:

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
ARS. Repeating the command keeps the same accounts and grants and resets their
passwords to the supplied value. A changed, suspended, or extra grant requires
explicit resolution. The password is never printed. The command creates users
in the Paths database; a deployment using an external identity provider also
needs matching accounts there for password login through that provider.

The GraphQL `framework.organizations` field returns only accessible
organizations, with `parentId`, `search`, `first`, and `offset` arguments.

The current Data Studio municipality picker still reads `framework.configs`,
which lists only provisioned model instances. Its loader must switch to the
organization query for browsing municipalities without model instances.
Active organization grants also reach provisioned municipal instances and their
datasets under the grant's subtree. An instance membership suspension blocks
that instance even if a wider organization grant would otherwise allow it.
The GraphQL mutations `assignOrganizationRole`, `suspendOrganizationRole`,
and `reactivateOrganizationRole` manage grants. Framework and active subtree
admins may use them within their scope.
