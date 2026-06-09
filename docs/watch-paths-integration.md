# Kausal Watch – Kausal Paths Integration

This document describes how to connect a Kausal Paths model instance to a Kausal Watch plan so that Watch category pages display emission impact values from the Paths model.

## How it works

1. The Watch UI fetches the plan's `kausalPathsInstanceUuid` (id or UUID only, not URL) field from the Watch backend.
2. It queries the Paths production server API (`api.paths.kausal.dev/v1/graphql/`) using `x-paths-instance-identifier` header to load the Paths instance context (goals, scenarios, parameters).
3. For each Watch category that has a `kausalPathsNodeUuid` set, the Watch UI queries the Paths API for that node's impact metric and displays the result on the category card.

## Required configuration

### 1. Watch plan — Paths instance identifier

In the Watch admin, open the plan settings and set **Kausal Paths instance UUID** to the Paths instance identifier string (e.g. `muenchen-bisko`). This is a string identifier, not an actual UUID despite the field name.

### 2. Paths model — outcome node with goals

The Paths instance must have at least one outcome node with `is_outcome: true` and a `goals:` block. Without this, the Paths API returns `goals: []` and any `GetNodeContent` query will fail with `"Goal not found"`.

```yaml
- id: net_emissions
  type: simple.AdditiveNode
  quantity: emissions
  unit: kt_co2e/a
  is_outcome: true
  goals:
  - label_de: Emissionen
    label_en: Emissions
    default: true
    linear_interpolation: true
    is_main_goal: true
    values:
    - year: <reference_year>
      value: <historical_value>
    - year: <target_year>
      value: <target_value>
```

### 3. Watch categories — Paths node UUIDs

In the Watch admin, open each category and set **Kausal Paths node UUID** to the identifier of the corresponding node in the Paths model (e.g. `heat_source_development_for_heat_networks`). Only categories with a non-empty value will show impact data.

### 4. Sync DB-sourced instances after YAML changes

If the Paths instance uses `config_source: database`, YAML changes are not picked up automatically. After editing the YAML and deploying, run:

```bash
python manage.py sync_instance_to_db <instance-identifier>
```

This must be run on every server where the change needs to take effect (local, staging, production).

## Diagnostic queries

**Check Watch plan identifier:**
```graphql
# POST https://api.watch.de.kausal.tech/v1/graphql/
{ plansForHostname(hostname: "example.watch.de.kausal.tech") {
    ... on Plan { identifier kausalPathsInstanceUuid }
} }
```

**Check Paths instance goals:**
```graphql
# POST https://api.paths.kausal.dev/v1/graphql/
# Header: x-paths-instance-identifier: <instance-id>
{ instance { id goals { id label default } } }
```

**Test a node's impact metric:**
```graphql
# POST https://api.paths.kausal.dev/v1/graphql/
# Header: x-paths-instance-identifier: <instance-id>
query GetNodeContent($node: ID!, $goal: ID) {
  node(id: $node) {
    id name
    impactMetric(goalId: $goal) { name id }
  }
}
```
