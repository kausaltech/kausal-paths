# Merging main into feat/data-studio-backend after the port-declaration change

*Produced by Claude Opus 5.5 on 2026-10-09.*
*Responsible: Jouni Tuomisto.*

This note is for whoever next merges `main` into `feat/data-studio-backend`. Two
changes on `main` touch code the branch has also changed: the argument-node and
reference-edge work (PR #288, merged) and the constraint-solver unit fix (PR #289,
merging). Most of what follows is ordinary conflict resolution. Three items are not,
because a merge without conflicts would still leave them wrong, and only one of
those fails loudly.

The findings come from a trial merge of `main` at `6d4ffa71` into the branch at
`80e2b6d8`, plus `git merge-tree` against PR #289's head `00bfe3cf`. Re-check the
lists if either side has moved much since.

## 1. What changed on main

A node class used to declare its input ports by assigning `input_port_declarations`
in its class body. Since #288, it assigns **`declared_input_ports`** instead, and
`Node.__init_subclass__` computes `input_port_declarations` from it, adding the
universal `Node.reference_port` at the end:

```python
cls.input_port_declarations = (*cls.declared_input_ports, Node.reference_port)
```

Two consequences matter for the merge:

- **An assignment to `input_port_declarations` in a class body is now refused.**
  `__init_subclass__` would otherwise overwrite it on the next line, and the class
  would silently lose every port it meant to declare. It raises `TypeError: <Class>
  assigns input_port_declarations; declare its ports in declared_input_ports`.
- **`input_port_declarations` is never empty.** Every class has at least the
  reference port. Code that tests it for emptiness to mean "this class has not
  migrated to ports yet" now gets the wrong answer for every class. Ask
  `declared_input_ports` for that question.

Read `input_port_declarations` when you mean "every role this class accepts", and
`declared_input_ports` when you mean "the roles this class itself declares".

## 2. Seven classes the branch migrated to ports (loud)

These classes exist on both sides, but only the branch gave them declared ports,
so those lines are new to the merge and #288's rename never reached them. After
the merge each still assigns the old attribute:

| File | Class |
|---|---|
| `src/nodes/actions/linear.py` | `DatasetDifferenceAction` |
| `src/nodes/buildings.py` | `FloorAreaNode` |
| `src/nodes/buildings.py` | `CfNode` |
| `src/nodes/costs.py` | `SelectiveNode` |
| `src/nodes/costs.py` | `DilutionNode` |
| `src/nodes/simple.py` | `DataAvailabilityNode` |
| `src/nodes/simple.py` | `AnnuityNode` |

**Fix:** rename the assignment to `declared_input_ports` in each, keeping the
`ClassVar[...]` annotation where there is one.

**If missed:** importing the module raises the `TypeError` above, so test collection
fails at once. This is the failure the guard exists to make visible.

Find any others with:

```bash
grep -rnE '^\s+input_port_declarations\s*[:=]' src | grep -v src/nodes/node.py
```

The only legitimate hit is the guard's own test in `src/nodes/tests/test_operands.py`.

## 3. `SelectiveNode.compute()` (silent)

`SelectiveNode.compute()` in `src/nodes/costs.py` exists only on the branch. It
gathers what to sum like this:

```python
for port in self.input_port_declarations:
    ...
    selected.extend(self.iter_input_bindings(port))
```

After the merge, that loop also visits the reference port, so an input tagged
`reference` would be **added to the sum**. No error, and no test catches it unless a
`SelectiveNode` has a reference edge.

**Fix:** loop over `self.declared_input_ports`.

In the trial merge, this and the loader check in §4 are the only reads of
`input_port_declarations` that exist only on the branch outside tests. Re-run this
after resolving, and judge each hit by which meaning in §1 it needs:

```bash
grep -rn 'input_port_declarations' src | grep -v '/tests/'
```

## 4. Conflicts with main as it is now (#288)

Four files conflict.

**`src/nodes/formula.py`, imports.** Keep both: the branch's
`from nodes.defs.node_defs import ActionConfig, FormulaConfig`, and main's
constants line, which adds `REFERENCE_TAG`.

**`src/nodes/instance_loader.py`, the unmigrated-class check** (around line 1446).
The branch extended the condition with a validation clause, and main switched it to
`declared_input_ports`. Combine both. The branch's version must not be taken as is,
because its `not target.input_port_declarations` is now always false (§1):

```python
if target is None or (
    not target.declared_input_ports and not any(p.validation for p in target_meta.spec.input_ports)
):
```

The comment under it already explains why it asks `declared_input_ports`.

**`src/nodes/instance_loader.py`, the role fallback** (around line 1467). Keep both
blocks, the branch's first: its validation-dependency fallback, then main's
`if REFERENCE_TAG in definition.tags: role = REFERENCE_ROLE`. A reference tag must
win over any other role, so it goes last.

**`src/nodes/tests/test_add_multiply_semantics.py`, imports.** Keep the branch's
`from datasets.runtime import Dataset` (the class moved there on the branch) and
main's constants line with `REFERENCE_ROLE, REFERENCE_TAG`. Drop main's
`from nodes.datasets import Dataset`.

**`src/nodes/tests/test_runtime_input.py`, imports.** Keep main's
`from nodes.defs.binding_def import EdgeBindingDef`. Drop main's
`from nodes.datasets import FixedDataset`, because the branch already imports it
from `datasets.runtime`.

## 5. Further conflicts once #289 is on main

#289 moves the solver's tag classification into a new module,
`src/nodes/constraints/tags.py`, and changes how a dataset port gets its unit. Five
files conflict with the branch.

**`src/nodes/constraints/solver.py`.**
- **Imports:** keep the branch's `shape_check` and `steps` imports, and add
  `from nodes.constraints.tags import tag_operation_is_opaque`.
- **Tag classification:** delete the branch's `NEUTRAL_TAG_OPERATIONS` and
  `_tag_is_opaque` from `solver.py`; they now live in `tags.py`. The two lists were
  identical at the time of the trial, so nothing is lost; diff them again before
  deleting.
- **Callers:** replace the two calls `_tag_is_opaque(...)` with
  `tag_operation_is_opaque(...)`. Nothing else on either side imports the old names.

**`src/nodes/instance_graph.py`, `INSTANCE_GRAPH_FORMAT_VERSION`.** The branch is at
10 and #289 sets 7. **Use 11**, keep all the existing comment lines, and add one:

```python
# v11: a dataset input port takes the dataset's authored unit before the node's own.
INSTANCE_GRAPH_FORMAT_VERSION = 11
```

The number is part of the graph cache key. Taking either side's number would let
graphs cached under the other side's meaning be read back.

**`src/nodes/instance_parser.py`, the dataset `InputPortDef`.** Keep the branch's
`binding_owner=ds_def.binding_owner,` and take #289's `unit=` line with its
comment:

```python
binding_owner=ds_def.binding_owner,
# An authored unit outranks the node's own output metric: on a
# formula node the dataset is one term, not the result.
unit=ds_def.unit if ds_def.unit is not None else (metric.unit if metric is not None else None),
```

**`src/nodes/tests/test_constraint_solver.py` and
`src/nodes/tests/test_instance_parser.py`.** Both sides appended tests at the same
place. Keep both sets.

## 6. Checks after resolving

```bash
# No class body may still assign the old attribute (only the guard's test should match)
grep -rnE '^\s+input_port_declarations\s*[:=]' src | grep -v src/nodes/node.py

# Every module in nodes imports; this is where a missed rename from §2 shows up
python -c "
from kausal_common.development.django import init_django; init_django()
import importlib, pkgutil, nodes
for m in pkgutil.walk_packages(nodes.__path__, 'nodes.'):
    if '.tests' not in m.name and '.migrations' not in m.name:
        importlib.import_module(m.name)
print('ok')"

python -m pytest --reuse-db src/nodes/tests
```

Then run the pre-push checks as usual. Nothing needs re-syncing for computation:
#288 and #289 left the outputs of every instance on main unchanged, apart from the
early years and forecast flags the retired `ignore_content` tag used to add. A
database-sourced instance whose nodes have an authored dataset unit will show a
spec diff in `debug_instance --diff-node` until its next `sync_instance_to_db`.
