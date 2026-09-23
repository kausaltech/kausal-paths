# Formulas as a view of stored pipelines

Status: agreed with Juha 2026-09-23; not started.

Related plans and docs:

- [PipelineNode capability discovery](pipeline-node-capability-discovery.md)
- [Action hooks](../architecture/action-hooks.md): `formula.FormulaAction` is the first consumer
- Pipeline operations spec (Dec 2024): `docs/trailhead/kausal-paths-pipeline-operations-spec.md.pdf`

## Decisions

1. **The pipeline is the stored form.** A typed `PipelineConfig` built on the
   existing `nodes.pipeline.PipelineSpec` becomes the one canonical, persisted
   computation schema, replacing the placeholder `PipelineConfig.operations`
   and the loose `NodeSpec.pipeline` field. This answers the discovery plan's
   request for one stored schema.
2. **A formula is a lossless, canonically formatted text view of it.** Experts
   may edit either the structured pipeline or the formula text; both edit the
   same stored pipeline. This supersedes the Dec 2024 spec's "formulas are
   one-way compiled", and the discovery plan's "FormulaNode is separate".
3. **One implementation per operation.** Temporal operations (`interpolate`,
   `extend`, `backfill`) become canonical pipeline operations sharing the
   functions the binding transformations use (`nodes/transforms.py`). No
   second implementation under the same name: `PathsExt.linear_interpolate`
   becomes an alias of the canonical op and is deprecated later.
4. **Magic flags give way to explicit steps.** For computation, interpolation
   belongs in the pipeline/formula (`interpolate(factor)`) rather than in a
   dataset flag (`interpolate: true`): it creates values, and the spec's rule is
   that edges and bindings carry shape and operations carry values. The binding
   transformations remain for existing models until migrated.

## Text ↔ pipeline mapping

| Formula text | Stored pipeline |
|---|---|
| `name = expression` | a step whose result is named `name` |
| nested sub-expression | an unnamed intermediate step |
| comment on the line(s) above a statement, or at its end | that step's `description` |
| comment after the last statement | the pipeline's `description` |
| last bare expression, or assignment to an output-port identifier | the output |
| `a + b`, `a - b`, `a * b`, `a / b` | `add`, `subtract`, `multiply`, `divide` |
| any other op | `op_kind(input, keyword=value, …)`; conditions as keywords (`only_if=…`) |
| an input-port identifier | `PortInputRef` |
| for `FormulaAction`, the identifier of the node it acts on | a hook-base reference (the target's un-hooked value) |

Example (`FormulaAction` acting on `final_energy_use`):

```python
# Heat-plan path, stated per carrier as a factor of today's use.
path = interpolate(factor)
# The hook adds the change against the node's own value.
final_energy_use * (path - 1)
```

compiles to an `interpolate` step named `path` (first comment as its
description), an unnamed `subtract`, and the output `multiply` (second comment
as its description), and renders back to the same text.

## Round-trip rules

- **Preserved:** step names, comments (as descriptions), and whether an
  intermediate is named or inline. The renderer inlines an unnamed intermediate
  used once and writes named ones as assignments. Step identity follows the
  result name (`pipeline_compile._intermediate_value_id`), so it survives edits
  to other lines.
- **Normalised:** whitespace, redundant parentheses and line breaks, like `gofmt`
  or `black`. The saved text is the canonical rendering.
- **Comment attachment:** a comment inside an expression goes to the enclosing
  statement's step. Loose comments between statements go to the next statement;
  that is documented behaviour, since a pipeline has no place for free-floating
  text.
- **Every pipeline renders:** each operation kind has a canonical call form, so
  a pipeline built in the structured editor can always be edited as text.
- **Invariant, enforced by tests:** `render(compile(text))` equals the canonical
  form of `text`, and `compile(render(pipeline)) == pipeline`.

## Mechanics

- Parse with Python's `ast` in `exec` mode (assignments); collect comments with
  `tokenize` and attach them to statements by line number.
- Map compile, shape and unit errors back to line and column.
- A text save replaces the whole pipeline in one validated step: parse, compile,
  then shape/unit checks, and nothing is stored if any fails.
- Step descriptions feed node explanations: the public explanation can show the
  steps with the modeller's own comments.

## First implementation step

1. Typed stored `PipelineConfig` on `PipelineSpec`: result names, step
   descriptions, pipeline description.
2. Canonical `interpolate`, `extend`, `backfill` operations sharing the binding
   implementations.
3. Formula compiler and renderer for arithmetic, names/ports, scalars, the
   temporal ops and the hook base, with round-trip tests including comments.

Runtime evaluation of `FormulaNode`/`FormulaAction` stays unchanged in this
step. Moving it onto the pipeline executor comes later, gated by the existing
parity harness (`nodes/pipeline/compare.py`). Formulas using functions without
a canonical op yet (about 60 `PathsExt` and custom functions) keep evaluating
as today and are reported as not representable until their ops exist.

## Open questions

- Descriptions are single-language text. If public explanations must be
  translated, a description becomes an I18n string, and the text editor writes
  the instance's default language.
- Multi-output nodes: one pipeline per output port, or assignments to output-port
  identifiers in one pipeline. Assignments would also settle the discovery plan's
  first checkpoint question.
