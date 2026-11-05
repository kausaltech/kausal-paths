# Comparing Two Versions of a Paths Model

*Produced by Claude Opus 5.5 on 2026-11-05.*
*Responsible: Jouni Tuomisto.*

"Is the model in sync?" can mean several different things. A Paths model is
not one object. It is made of several layers, and each one has its own copies
that drift apart independently. This document gives an overview: which copies
exist, which one a given reader actually sees, and which strategy compares
which pair. The commands have their own manuals, linked at the end; here they
appear only to show which comparison each one makes.

## 1. The layers, and their copies

| Layer | Copies that can differ |
|-------|------------------------|
| **Structure** (nodes, edges, dimensions, params, scenarios, pages) | The YAML in `configs/` (in your checkout, and in the deployed image) · the **draft** in the DB (`InstanceConfig.spec`, `NodeConfig.spec`, port bindings) · each **published revision** (an `InstanceSnapshot` stored with the revision) |
| **Data** | Producer output in paths-data (`models/<city>/datasets/`, `modules/*/`) · DVC at the commit the config **pins** · DVC at other commits · the **DB row** (`Dataset` + `DataPoint`), one per instance scope |
| **Code** | The kausal-paths commit, including node classes and how they compute |
| **Environment** | Local DB, staging, and each production region (fi/de/us), each with its own DB |

A node's output is a function of all four. Two outputs can differ because of
any one layer, so a comparison has to name both sides on every layer, or at
least know which layers it is holding fixed.

## 2. Which copy does a reader see?

Before comparing anything, work out which copy is the real one for the
question you are asking. Usually that means "what do visitors see?".

**Structure**

| Instance state | Anonymous visitor | Model editor / `preview: DRAFT` |
|----------------|-------------------|-------------------------------|
| `config_source: yaml` | the deployed YAML (with the DB's admin overlay) | same — there is no revision |
| `config_source: database`, never published | the draft | the draft |
| `config_source: database`, published | the **live revision** | the draft |

The rules are in `_resolve_preview_source` (`src/paths/schema_context.py`).
Check the state like this:

```bash
python manage.py shell_plus --quiet-load -c "ic=InstanceConfig.objects.get(identifier='<instance>'); print(ic.config_source, ic.live_revision_id, ic.has_unpublished_changes)"
```

Compare `config_source` in every environment you care about; it is a DB
field, not part of the YAML, so a local DB can hold a yaml-sourced copy of an
instance that is database-sourced in production. The local copy then tells you
nothing about production's draft.

**Data.** With `use_datasets_from_db`, any dataset identifier that has a DB
row in the instance's scope is read from that row, whatever the
`config_source`. Identifiers without a row are read from DVC at the pin. So
the "real" data is the DB row where one exists, and DVC everywhere else. A
published revision pins the dataset rows it computed from
(`InstanceRevisionDatasetPin`).

## 3. Two strategies: compare outputs, or compare inputs

- **Compare outputs** to find out whether a difference *matters*: compute both
  versions and diff node values. This is robust to representational noise
  (a field serialized differently, a reordered list) and catches differences
  in layers you did not think to inspect. It does not tell you *why*.
- **Compare inputs** (specs, dataset rows, pins) to find out *what* differs
  and so where to fix it. This can report differences that change no output.

A practical order: compare outputs first. If they are identical, stop. If they
differ, compare the inputs of the nodes that changed, nearest the leaves first.

## 4. Pairwise comparisons

### 4.1 YAML vs DB draft (structure)

- **One node:** `python -m tools.debug_instance -i <instance> --diff-node <node>`
  parses the YAML in the current checkout and diffs it against the stored
  `NodeConfig.spec`.
- **Node metadata on a yaml-sourced instance:**
  `load_nodes.py -i <instance> --update-nodes --overwrite --dry-run` prints a
  JSON diff of what would change in `NodeConfig` (names, descriptions and
  other admin fields).
- **The whole instance:** there is no command yet. `sync_instance_to_db --dry-run`
  prints totals only, not a diff. Running `--diff-node`'s comparison over every
  node and over `InstanceConfig.spec` is a short script; a `--diff-instance`
  option on `debug_instance` would be the natural home for it.

Two cautions:

- **The YAML side is whichever file the process reads.** Run in production,
  that is the deployed YAML, not your working tree.
- **A yaml-sourced instance's DB spec is the minimal one** `ensure_spec()`
  derives, with no dimension catalogue, and its node specs may be empty. A
  diff against it reports almost everything as different and means nothing.
  See [`data-management.md`](data-management.md), Step 9.

**Syncing is not a comparison.** After `sync_instance_to_db`, the draft equals
the YAML by construction, so the before state is gone. That is fine when the
YAML is the only place the model is maintained. When anyone edits the
instance in the model editor, those edits exist only in the DB, and the sync
overwrites them. Diff first in that case.

### 4.2 Draft vs published

```bash
python manage.py compare_draft_published <instance>
```

This computes both and compares every node's output, cell by cell. It is the
only way to see what a publish will change, because the public UI never asks
for the draft. `has_unpublished_changes` is a cheaper first signal but a weak
one: every sync of a published instance sets it, even a sync that changed
nothing.

This also serves as an after-the-fact diff for a sync: if the live revision
matched the old draft, `compare_draft_published` after the sync shows what the
sync changed, at the output level.

### 4.3 Two published revisions

There is no command for this. A revision stores an `InstanceSnapshot` and its
dataset pins, so a comparison would load both through the published path and
diff outputs as `compare_draft_published` does.

### 4.4 DB row vs DVC (data)

| Question | Command |
|----------|---------|
| Which datasets exist where, with counts and dates on both sides? | `dataset_inventory <instance>` (`--order drift`, `--csv`) |
| Which ones need importing, renaming or nothing, and in what order? | `dataset_status <instance> --stale-only` |
| What exactly differs in one dataset, row by row? | `compare_dataset <instance> <dataset>` |
| What would an import change? | `load_dvc_dataset <instance> <dataset> --plan` |

Things that make these numbers mislead:

- **Point counts include empty cells** on both sides, deliberately. Drift in
  counts is a difference in shape, not in how full the data is.
- **A template** (DVC holds the shape, the DB row holds values typed in the
  admin) has zero count drift. It is finished, not stale, and re-importing it
  blanks the row. `dataset_status` grades it `template`.
- **DVC means DVC at a pin, and there are two pins.** A DB-sourced instance's
  spec carries one, the YAML carries another, and they agree only after a
  sync. `--repo-from yaml|db` chooses; the commands print which they used.
- **Rows are per scope.** Each city's copy of a shared identifier goes stale
  on its own.

### 4.5 paths-data producer output vs DVC vs DB

Producer scripts in paths-data write the CSV that `upload_new_dataset` pushes
to DVC. To check whether what reached DVC, or the DB, is still what the
producer produces, bring both into the same format and diff them:

```bash
python manage.py export_dataset <instance> <dataset> --from dvc --out DIR_A
python manage.py export_dataset <instance> <dataset> --from db  --out DIR_B
```

Exporting `--from dvc` twice, once with `--repo-from yaml` and once with
`--repo-from db`, compares the data at the two pins in the same way.

`export_dataset` writes the upload format, with its sources registry, so it
compares directly against the producer's output CSV. It reports what it cannot
carry rather than dropping it silently. When all you have is a hash,
`python -m tools.fetch_dataset <ekey> <out.csv>` reads the object with no pin,
DB or checkout involved.

Remember that an upload is not an import: after a new DVC push, the DB row
still holds the old data until `load_dvc_dataset` runs.

### 4.6 Before vs after a change (code or config)

```bash
python manage.py test_instance <instance> --state-dir DIR --store     # before
# ... change code, YAML or data ...
python manage.py test_instance <instance> --state-dir DIR --compare   # after
```

`--all-nodes` compares every node rather than only outcome nodes. For a single
node, `load_nodes.py -i <instance> --node <node>` (with `--scenario`) prints
its output on each side for a manual diff.

For changes to one node's inputs, the cheapest test of whether something is
load-bearing is the one in
[`trailhead/tools.md`](trailhead/tools.md) (*Retiring a dataset*): capture the
outputs, make the change, recompute and diff.

### 4.7 Local vs production

No tool compares across environments; each command sees one database. Run the
same read-only command in both and diff the text, for example
`dataset_inventory <instance> --csv` or a node's output from `load_nodes.py`.
Before trusting the result, check that the two sides are actually comparable:
the same `config_source` (§2), the same kausal-paths commit, the same YAML,
and the same pins.

## 5. Traps that apply everywhere

- **A forced source shows a state that exists nowhere.** `debug_instance
  --source yaml` on a database-sourced instance computes something no visitor
  or editor ever sees. Judge an instance from its own `config_source` unless
  that hypothetical is exactly the question.
- **The cache can hide a change.** When outputs that should differ do not,
  compare again with the cache off (`load_nodes.py --skip-cache`,
  `debug_instance --no-cache`) before concluding they are equal.
- **Comparing in one process.** `debug_instance` holds one context, so
  computing two scenarios one after the other in the same process is not the
  same as two requests.
- **Write the "before" down first.** Every fixing command (`sync_instance_to_db`,
  `load_dvc_dataset --force`, publishing) destroys one side of the comparison.

## 6. Where the manuals are

| Topic | Document |
|-------|----------|
| `debug_instance`, `sync_instance_to_db`, `compare_draft_published`, `dataset_status`, `dataset_inventory`, `load_dvc_dataset`, `test_instance` | [`trailhead/tools.md`](trailhead/tools.md) |
| Stale spec vs stale dataset, refreshing a DB row | [`data-management.md`](data-management.md), Step 9 |
| `upload_new_dataset`, `export_dataset`, `fetch_dataset` | [`dataset-management-commands.md`](dataset-management-commands.md) |
| What survives a DB → CSV → DVC → DB round trip | [`dataset-round-trip.md`](dataset-round-trip.md) |
| How bindings refer to datasets | [`architecture/port-bindings-and-dataset-identity.md`](architecture/port-bindings-and-dataset-identity.md) |
| What a scenario sets and omits | [`architecture/scenarios.md`](architecture/scenarios.md) |
