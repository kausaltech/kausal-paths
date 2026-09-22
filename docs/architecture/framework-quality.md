# Framework quality catalogue

Framework models live under `src/frameworks/models/`:

- `framework.py`: framework identity, defaults, and classification dimensions.
- `config.py`: instance membership, configuration, and measure-to-node resolution.
- `measures.py`: sections, measure templates, defaults, measures, and data points.
- `quality.py`: versioned quality schemes and their categorical levels.

`frameworks.models` re-exports existing names, including the defaults and callables
referenced by historical migrations. Moving the models does not change their app
labels, database tables, or content types.

## Quality schemes

`DataQualityScheme` belongs to a framework and is identified by
`(framework, identifier, version)`. Its `DataQualityLevel` rows have stable UUIDs,
an identifier, display name, description, display order, and a numeric score in
the inclusive range 0–1. Grade identifiers are unique within a scheme version.

These are framework vocabulary, not assessments of individual values. Later
evidence can refer to a particular level, and assessment profiles can use the
same catalogue without owning or duplicating it. Ungraded is absence of an
assessment, not another level and not a synonym for a zero score.

Model saves reject changes to scheme identity and to a level's scheme,
identifier, or score. Create a new scheme version when the scoring meaning
changes. Names and descriptions can be corrected. Direct bulk SQL/ORM updates
bypass those model guards; the provisioning service detects conflicting seed
scores and refuses to overwrite them. Database constraints enforce uniqueness
and score bounds independently of model validation.

The catalogue follows the existing read-only framework vocabulary permission
policy: readable publicly, writable through permissioned surfaces only by
superusers. No new API editing surface is introduced here.

## BISKO provisioning

After applying migrations, with an existing `bisko` template instance:

```bash
python -m tools.setup_bisko --dry-run
python -m tools.setup_bisko
```

This creates the `bisko` Framework, points its template at that instance, and seeds
the `bisko` quality scheme version `1`: A=1, B=0.5, C=0.25, D=0.
`1` versions this catalogue definition; it is not a certification protocol edition.
An alternative existing template can be selected with `--template IDENTIFIER`.
Missing instances or conflicting existing definitions fail the whole transaction.
Reruns preserve UUIDs and existing framework settings.

The script creates no Wagtail pages, hostnames, organizations, or calculation
graphs. Registration and instance creation remain disabled on a newly provisioned
framework. The template FK alone does not attach the template as a framework member.

Attach an existing database-backed dependent instance explicitly:

```bash
python -m tools.setup_bisko --instance example-bisko --dry-run
python -m tools.setup_bisko --instance example-bisko
```

Repeat `--instance` for more than one dependent instance. Membership changes instance
authorization: the existing framework permission policies apply. No users or
groups are granted access by this script.

YAML-backed membership is refused: `InstanceConfig.get_yaml_config_entrypoint()`
switches framework members to the framework's YAML file, which would replace a
dependent instance's own entrypoint. Migrate and verify its database-backed model first.

Shared graph publication and conversion are documented in
[template inheritance](template-inheritance.md).

## Next steps

Data-point evidence and quality assignment, derived numeric quality columns,
GraphQL quality fields, assessment profiles and their typed criteria, and the
certification evaluator remain separate work. This foundation does not yet change
how the data editor reads or writes quality.
