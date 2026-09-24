# Explicit ports and hooks in YAML

Status: agreed with Juha 2026-09-23; not started.

Tags currently double as formula names, and they are overloaded (`non_additive`,
`splittee`, `partial_factor`, `factor`, …). This plan gives YAML node definitions
explicit input ports, output ports and hooks, alongside the legacy
`input_nodes`/`input_datasets` syntax. A node uses one syntax or the other,
never both.

```yaml
- id: knsv_dekarbonisierung_strommix
  type: formula.FormulaAction
  input_ports:
  - id: factor                          # the name in the formula
    unit: dimensionless
    dataset: knsv_dekarbonisierung_strommix/knsv_massnahmen_energietraeger
  hooks:
  - id: territorial                     # the contribution: an output port, assigned in the formula
    acts_on: end_energy_emission_factors   # or node.port
    base_as: territorial_ef             # optional: the target's un-hooked value, as an input
  - id: administration
    acts_on: municipal_building_energy_ef
    base_as: administration_ef
  formula: |
    # The factor on today's grid emission factor.
    path = interpolate(select_category(factor, measure_scenario=knsv_variant))
    territorial = territorial_ef * (path - 1)
    administration = administration_ef * (path - 1)
```

Rules:

- **Ports and hooks are lists with `id`**, not maps keyed by identifier, so
  validation rules can address them.
- **A port binding is `node:` or `dataset:`**: one reference, or a list for a
  `multi` port. The long form carries `metric`, `transformations`
  (`kind` strings or one-key maps, the stored vocabulary) and `tags`.
- **`binding_owner: instance`** on a template's input port marks a
  municipality's data slot. It replaces the `declare_local_data_slots`
  inference from `kommune/*` names.
- **`hooks:` is only valid on actions.** A hook's `id` names its contribution.
  The contribution's unit and dimensions are the target's, and an assignment
  to that id in the formula makes it an output. `base_as` is optional, because
  an absolute effect never reads the target.
- **One namespace for formula names.** Input ports, hook ids, `base_as` names
  and parameters share it.
- **`formula:` is a top-level key** that compiles to the stored pipeline
  (`docs/plans/formula-pipeline-round-trip.md`). Its comments become step
  descriptions.
- **Port UUIDs derive from the node UUID and the identifier.**

Needs, besides the parser: multi-output pipelines (an assignment to an output
identifier), and the export direction.

The hooks change makes `partial_factor` unnecessary. The inner join of
`territorial_ef * (path - 1)` limits the contribution to the carriers in the
dataset, and the hook's outer join leaves the other carriers unchanged.
