# Runtime vs structure-based validation and warnings

## Structure-based validation (config / graph time)

**Where:** `nodes/explanations.py`

- **ValidationRule** subclasses (e.g. `NodeClassRule`, `DatasetRule`, `EdgeRule`, `BasketRule`, `FormulaDimensionRule`, `FormulaUnitRule`) implement `validate(node_config, context) -> list[ValidationResult]`.
- **ValidationResult:** `method`, `is_valid`, `level` (`'error' | 'warning' | 'info'`), `message`.
- **When:** `NodeExplanationSystem.generate_validations()` runs all rules at instance load; results are stored in `nes.validations` (keyed by node id) and shown in the UI.
- **Scope:** Rules see only **config** (node config, graph, context). They do **not** see computed outputs or runtime data.

So things like “missing input”, “unknown dimension”, “formula unit mismatch” are caught here.

---

## Runtime warnings and explanations

Runtime behaviour is less unified. Two mechanisms exist:

### 1. Log-based warnings

- **Node.warning(msg)** (`nodes/node.py`) → **context.warning()** (`nodes/context.py`) → **instance.warning()** (`nodes/instance.py`) → **log.opt().warning()** (loguru).
- Used in a few places (e.g. normalizer, generic node “node not found”, instance_loader).
- **Visibility:** Logs only. Not attached to a specific node’s explanation in the UI.

### 2. DataFrame runtime explanations (UI-visible)

- **PathsDataFrame._explanation: list[str]** (`common/polars.py`). Initialized to `[]`; copied when creating new frames from a source (e.g. `_from_pydf(..., source_df=...)`).
- **Who writes:** In `common/polars_ext.py`, operations such as `join_over_index`, `add_df`, `multiply_df`, `compare_df` append **category mismatch** (and similar) messages to `out._explanation`.
- **Who reads:** **Node.get_explanation()** (`nodes/node.py`): when `instance.features.show_category_warnings` is True, it appends `get_output_pl()._explanation` as a “Category warnings” section to the node’s HTML explanation.
- **Visibility:** These messages show up in the node’s explanation in the UI (when “show category warnings” is on).

So: **runtime warnings that should appear in the node explanation** need to be attached to the **result PathsDataFrame** of that node (e.g. by appending to `_explanation`). Log-based `node.warning()` is for logs only.

---

---

## Implementing and() / or() with “non-binary” runtime warnings

Goal: wrapper functions **and(a, b)** and **or(a, b)** that:

- Delegate to **min(a, b)** and **max(a, b)** respectively.
- Emit a **runtime warning** if any value deviates from 0 or 1 by more than a tolerance (e.g. 1e-6).

Two ways to surface that warning:

1. **Log only:** In the formula handler for `and`/`or`, after computing the result, check the evaluated `left`/`right` (scalars or PDF value columns). If any value is outside `[0, 1]` beyond tolerance, call **`self.warning("and()/or() received non-binary value ...")`**. Simple; visible in logs only.
2. **UI-visible:** When the result is a **PDF**, append a message to **`result._explanation`** (e.g. “Logical and() received values outside {0, 1}; interpreted as fuzzy logic.”). Then it appears in **Node.get_explanation()** when `show_category_warnings` is True, consistent with other runtime explanations.

You can do both: log for debugging and `_explanation` for user-facing “category warnings” on the node.

Implementation sketch:

- In **FormulaNode._handle_custom_function**, add branches for `func in ('and', 'or')` that:
  - Evaluate both arguments (same as for `max`/`min`).
  - Call `_apply_max_min(left, right, 'min')` or `'max'` to get the result.
  - Call a helper **`_warn_if_non_binary(left, right, result, func_name)`** that:
    - For **Quantity:** check `abs(x - 0) > tol and abs(x - 1) > tol`.
    - For **PDF:** check the value column (e.g. any row outside `[0, 1]` beyond tol).
  - If non-binary detected: `self.warning(...)` and, if result is a PDF, `result._explanation.append(...)`.
  - Return the same result as min/max.

This keeps and()/or() as thin wrappers over min/max and reuses the existing runtime-warning mechanisms (log + optional `_explanation`) without adding a new system.

**Implemented:** FormulaNode supports `and(a, b)` and `or(a, b)`; they delegate to min/max and append to `result._explanation` when inputs deviate from 0 or 1 by more than `LOGICAL_TOLERANCE` (1e-6), so the warning appears in the node explanation when `show_category_warnings` is True.
