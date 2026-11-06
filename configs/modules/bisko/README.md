# Frozen certification model

`model.yaml`, `configs/bisko.yaml` and `configs/pruefstadt-bisko.yaml` are based on Paths
revision `d5ed34484e6a8d9510f430041933bcc111b45f7b`, for the ongoing BISKO review, with one
deliberate correction: municipal-facilities overlap
subtraction is disabled in `bisko` as it already is in `pruefstadt-bisko`.
The module includes the November 2 certification fix requiring a quality grade for every
entered stationary-energy and district-heating value. Tests pin its contents and the original
21 per-sector end-energy requirements.

The evolving framework template is `configs/bisko-base.yaml`, with its module under
`configs/modules/bisko-base/`. The framework identifier remains `bisko`; its
`template_instance` is `bisko-base`. Other municipal models use the evolving module.

The certification entrypoints remain independent instances, with their existing URLs, data,
schemas, validation rules and publications. Do not attach them to the framework, share their
catalogue with it, or use the framework setup command to refresh them.

The new parser compiles the old certification YAML syntax through
`nodes.bisko_certification.certification_yaml`: named category domains become native shapes,
and the original metric requirements retain their groups and enforcement tiers. This is
scoped to the two certification entrypoints; it does not relax the YAML contract for new models.
Municipal facilities are entered separately, so the overlap correction is disabled in both
certification entrypoints. The evolving module removes that subtraction altogether.

To create or refresh the separate framework template:

```sh
python -m tools.setup_bisko --sync-template --publish --dry-run
python -m tools.setup_bisko --sync-template --publish
```

The default template is `bisko-base`. The command bootstraps a missing instance and accepts
`--template` for another non-certification template. It refuses to replace an existing
framework's template; any such transition must preserve or migrate existing revision pins
explicitly. No automatic template reassignment is part of this split.

Both models still run on the same backend. Validate their calculations and editing behavior
after backend changes, including missing cells, quality grades, scenarios and Excel exports.
To retire the split after certification, migrate callers deliberately rather than copying
new calculations into the reviewed module while it is still in use.
