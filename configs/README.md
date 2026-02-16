# Kausal Paths — general model design instructions

Instructions for designing model YAML files and datasets for participatory climate action modeling.

---

## Model philosophy

### Core purpose

The models are built to increase understanding among decision makers, civil servants, citizens, and stakeholders. Many models focus on city-level topics, but the platform itself is more generic.

Importantly, the purpose not primarily to convince stakeholders or force them to adopt a consensus. Instead, it offers a platform for stakeholders to bring their ideas, concerns, and objections to the table and try to convince "us"—an imaginary group sharing common values.

### Common values

- Less suffering is good, irrespective of whose suffering we can reduce.
- Future generations should have opportunity for living good lives.
- Impartial consideration of all people.
- Intergenerational equity.

### Value framework approach

The model uses common values as the baseline scenario. Alternative value frameworks (libertarian, localist, etc.) can be explicitly modeled as variants. Model outputs show: "Under framework X, outcome Y violates value Z." This makes transparent that policy disagreements are sometimes about values, sometimes about facts.

### Stakeholder representation

Different stakeholders have different beliefs about causal relationships. These are encoded as categorical dimensions (e.g. `hypotheses`) on nodes. Stakeholders can select their belief categories and see resulting outcomes. The same computational model accommodates different worldviews.

### Objections handling

**Empirical disputes** are claims about how the world works that can be investigated with evidence. Treatment: model them as causal chain parameters or uncertainties; explore with sensitivity analysis.

**Value conflicts** are claims rooted in value frameworks that conflict with or are orthogonal to common values. Treatment: flag as alternative value framework; run scenarios with different value weights; output shows what values were violated.

**Interests as values** are claims that appear principled but primarily protect material interests. Treatment: separate "How much will X lose?" (empirical) from "Should we care?" (normative). Within the common-values framework, weigh economic loss against suffering reduction.

---

## Architecture

### Structure

The model is a directed acyclic graph (DAG) of nodes.

### Node types

**Actions** are policy levers that we control externally. They are typically `simple.GenericAction`. They are set by scenario definitions, usually have `input_datasets` and rarely `input_nodes`, and have no formulas—values come from scenarios or datasets.

**Computational nodes** are all other nodes, calculated from inputs. Types include: `generic.GenericNode` (dataset-driven, may have operations), `formula.FormulaNode` (explicit formulas), and `simple.AdditiveNode` (sum inputs; used for outcome aggregation).

**Value nodes** are technically actions (`simple.ValueAction`), but their main purpose is to let users express their values about the moral and other topics in the model by setting parameters (often shown as sliders on the pages). They do not affect the model about the physical world; instead, they affect the utilities and other valuation nodes.

**Argument nodes** are metadata or documentation nodes. They are pure documentation and not used in calculations. Their timeseries are filled with zeros. They connect *to* computational nodes via `output_nodes`. Use `quantity: argument`, `unit: dimensionless`, and a dataset with zeros.

### Dimensions

Nodes can have multiple dimensions (categorical variables). The most common dimensions are `energy_carrier` and `sector`. Other examples: `time_of_day`, `vehicle_type`, `stakeholder`, `hypotheses`, `ghg` (greenhouse gas).

**Dimension arithmetic:** Multiplication (and division) results in the union of dimensions from both inputs. Addition (and subtracton) requires dimensions to match; incompatible dimensions must be aggregated first. Also categories are treated differently: multiplication and division merge with inner join resulting in dropout of categories that do not exist in both sides. In contrast, addition and subtraction use outer join and the emerging null values are treated as zeros.

**Flattening:** Use `from_dimensions` with `flatten: true` at edges to sum over a dimension. Use `sum_dim(expr, dimension)` in `FormulaNode` to aggregate within a formula.

**Hypotheses dimension:** Its purpose is to encode competing hypotheses or beliefs about causal relationships. Categories describe the hypothesis itself, not who believes it. Good category examples: `high_displacement`, `net_reduction`, `empirical_estimate`. Bad examples: `business_belief`, `planner_belief`. Use the hypotheses dimension sparingly—only when showing multiple hypotheses simultaneously. An alternative is to use different scenarios with different parameter values. Aggregate hypotheses as soon as there are no interactions with other dimensions; ideally calculate expectation (weighted mean).

**Stakeholder dimension:** Its purpose is to let different stakeholders use the same model with different parameters. It is often used on value weight nodes for multi-criteria decision making (MCDM). Note that in some cases, stakeholder dimension is stackable, e.g. when distinct stakeholders see different costs and it is possible to calculate total costs by summing over stakeholder dimension. In contrast, stakeholder categories may be overlapping (young people, women) and they may share same utilities (e.g. the utility of reaching a national climate target); in these cases summing up over the dimension is not meaningful.

### Causal chains

Causality lives in the **formulas**, not in edge tags. Edge tags such as `increase` and `decrease` are only for documentation when the formula is not yet known. However, some edge tags (such as `non_additive`) inform the formula about what to calculate.

Include all intermediate nodes and be explicit about mechanisms.

### Outcome nodes

Outcome nodes are for UI display and reporting. They are typically `simple.AdditiveNode` and are flagged with `is_outcome: true`. Use one outcome node per outcome type.

---

## YAML structure

### Top-level sections

Top-level sections include `actions` and `nodes`.

### Action schema

**Required fields:** `id` (snake_case identifier), `name` (human-readable, sentence case), `description` (HTML allowed), `type` (usually `simple.GenericNode` or `formula.FormulaNode`), `quantity`, `unit` (Pint-compatible), `output_dimensions` (list of dimension ids).

**Optional fields:** `input_datasets`, `params` (including multipliers for unit fixes).

### Node schema

**Required fields:** `id` (snake_case), `name` (sentence case), `type` (e.g. `generic.GenericNode`, `formula.FormulaNode`, `simple.AdditiveNode`), `quantity`, `unit`, `input_dimensions` (legacy—dimensions going into formula), `output_dimensions` (dimensions coming out).

**Optional fields:** `description` (HTML allowed), `is_outcome` (boolean), `input_datasets`, `input_nodes`, `output_nodes` (for argument nodes), `params`.

**Input nodes structure:** Each entry has `id`, optionally `to_dimensions` (list with `id`), `from_dimensions` (e.g. with `flatten: true`), and `tags` (reference names for formulas).

**Params for formula node:** `params` with `formula` (expression using tag names as variables).

**Params for generic node:** `params` with e.g. `operations` (comma-separated names), `multiplier`, `value`, and units as needed.

### Formula syntax

**Allowed:** Basic arithmetic (`+`, `-`, `*`, `/`, parentheses), functions such as `exp()`, `log()`, `abs()` `min()`, `max()`, aggregation `sum_dim(expression, dimension_name)`, helper `complement(x)`, and logical comparisons (`==`, `!=`, `>=`, `<=`, `<`, `>`, which return 1.0 or 0.0). You can also use `select_port(condition, a, b)` to select between two data frames based on a boolean condition.

**Not allowed:** if–then clauses (use logical expressions instead); unit conversion constants (Pint handles these).

**Conditional logic:** Use expressions that return 1.0 (true) or 0.0 (false), e.g. `formula: (a > b) * outcome_if_true + (a <= b) * outcome_if_false`.

**Dimension aggregation:** Use `sum_dim()` inside a formula, e.g. `formula: baseline + sum_dim(displaced * spillover, hypotheses)`. Alternatively, use `from_dimensions` with `flatten` at the edge level.

### Tags usage

Tags serve multiple functions. For **semantic documentation** (when there is no formula), use tags like `increase`, `decrease`, `ignore_content` to document direction of effect. For **variable naming**, use tags like `price`, `elasticity`, `baseline` to reference input nodes in formulas. You can combine both on the same input (e.g. `tags: [increase, price]`), but if there is a formula, semantic tags are just documentation.

### Unit handling

Pint does automatic conversion; never put conversion constants in formulas.

**Common units:** Monetary (e.g. EUR/trip, EUR/vehicle, EUR/a), traffic (vehicles/hour, trips/hour), distance (km, km/trip, vehicle_km/hour, vehicle_km/a or vkm/hour, vkm/a), emissions (e.g. g_co2e/km, g_co2e/vkm, t_co2e/a, kt_co2e/a), dimensionless (for fractions, ratios). Custom units include `t_co2e` (tonne CO2 equivalent), `kt` (kilotonne), `vkm` (vehicle_km alias). Pint converts between e.g. /hour and /a automatically. Use `dimensionless` or `%` for percentages. When dataset units do not match node units, use a multiplier parameter (this is a workaround; the proper fix is correcting dataset units).

### Operations system

GenericNode runs a sequence of operations. Default is `get_single_dataset`, `multiply`, `add`. If the node deviates, specify explicitly, e.g. `params: - id: operations, value: get_single_dataset,apply_multiplier`. Common operations include `get_single_dataset`, `multiply`, `add`, `apply_multiplier`. Many other operations exist but are not detailed here.

---

## Dataset structure

Datasets are typically CSV with a specific column structure.

**Columns:** Dataset (full descriptive name; snake_cased to create id), Metric (usually Value when one metric per dataset), Quantity (from node definition), Unit (from node definition), one column per dimension (empty when not relevant), optional Description on first row, then year columns (e.g. 2010, 2011, …, 2025) with actual values.

Create one column for each dimension that exists in any dataset; leave cells empty where a dimension does not apply. Each row is a unique timeseries for a combination of dimension categories; each year has its own column. Years 2010–2023 are typically historical, 2024–2025 future. For actions, historical data should be zeros (no policy yet); actual action values are set when running scenarios. When the model processes a dataset, it becomes a PathsDataFrame with a metric column (float with unit), zero or more dimension columns, year column, and forecast boolean column.

Datasets are organized hierarchically (e.g. `transportation/dataset_id`). Format can be CSV, DVC, JSON, or internal database; the model sees PathsDataFrame and does not depend on storage format.

---

## Design patterns

**Always specify:** Both `input_dimensions` and `output_dimensions` (even if identical); `to_dimensions` for each input_node in multi-row format with explicit `id`; explicit dimensions after operations.

**Prefer:** `complement(x)` over `(1 - x)`; `sum_dim(expr, dim)` for in-formula aggregation; `vkm` as alias for vehicle_km; explicit intermediate nodes over complex formulas.

**Dimension management:** Keep dimensions as long as they might interact with others (e.g. keep hypotheses through vehicle_type calculations if reduction rate differs by vehicle type under different hypotheses). Aggregate dimensions as soon as there are no more interactions, using `from_dimensions` flatten at edges or `sum_dim()` in formulas; aggregate hypotheses when there are no more hypothesis-specific interactions. For addition, dimensions must match; aggregate incompatible dimensions first. For multiplication, the result has the union of dimensions.

**Formula clarity:** Use descriptive tags (e.g. `[baseline]`, `[reduction]`, not `[x]`, `[y]`). Break complex calculations into multiple nodes. Document units in node names when ambiguous.

**Argument nodes:** Connect them to the most directly relevant computational node(s). Use HTML in descriptions for rich documentation. List related nodes, stakeholders, and hypotheses in the description. They can connect to multiple nodes if the objection relates to several.

**Naming conventions:** IDs in snake_case; names in sentence case; use single quotes for strings; only quote strings that contain special characters (e.g. `:`, `#`).

---

## Style guidelines

**Indentation:** When list items are on separate rows starting with `-`, indent the `-` at the same level as the list title. Correct: `my_list:` then `- item1` / `- item2` at the same indent as `my_list`. Incorrect: extra indent under `my_list`.

**Dimension references:** Use multi-row format with explicit `id` for `to_dimensions` and `from_dimensions`. Correct: `to_dimensions:` then `- id: dimension1` etc. Incorrect: `to_dimensions: [dimension1, dimension2]`.

**String quoting:** Quote only when necessary (special characters). No quotes needed for e.g. `name: Traffic volume`, `unit: vehicles/hour`, `formula: baseline * (1 - reduction)`. Quotes needed for e.g. `name: 'OBJ-CC13: Traffic displacement'`, `color: '#FF0000'`, or descriptions containing apostrophes. Prefer single quotes unless double quotes are needed (e.g. for an apostrophe).

---

## Best practices

Start simple: begin with a minimal viable model and expand incrementally. Test early: run the model as soon as possible to find issues. Document decisions: record rationale for design choices. Mark workarounds: use FIXME for temporary solutions. Keep structure explicit: include all intermediate nodes. Refine progressively: start with qualitative edge tags and add formulas later; start with placeholder data and refine values later; start with simple scenarios and add complexity later.

---

## Key concepts

**Participatory modeling:** The model is designed for stakeholder engagement, not expert prediction. The goal is to help stakeholders understand the implications of their beliefs and values. The goal is not to convince anyone or find an "optimal" policy.

**Hypothesis dimensions:** Encode competing empirical beliefs as categories in a dimension. Different stakeholders select different categories to see "their" outcomes. The model structure stays the same; parameters vary.

**Value weights:** MCDM converts outcomes to utilities. Different stakeholders weight outcomes differently.

**Argument nodes:** They are documentation nodes that do not affect calculations. They link stakeholder concerns to relevant computational nodes and enable traceability (e.g. "This objection relates to these parameters").

**Outcome nodes:** Special nodes marked `is_outcome: true` for UI presentation. Use AdditiveNode to sum components.

---

## Technical notes

**Pint integration:** The Pint library handles all unit conversions automatically. Never put conversion constants in formulas. Trust Pint to convert e.g. /hour to /a, g to t.

**Dimension broadcasting:** When multiplying nodes with different dimensions, the result has both; the system broadcasts appropriately (similar to NumPy). Example: [time, vehicle] × [vehicle, fuel] → [time, vehicle, fuel].

**PathsDataFrame:** Internal data structure wrapping the dataframe; it has built-in unit awareness via Pint and handles dimension alignment automatically.

**DAG execution:** Nodes are calculated in topological order (dependencies first). Each node is calculated once per run. Comparisons are done by running the model twice (e.g. baseline vs scenario).

**input_dimensions (legacy):** The `input_dimensions` field is legacy and mostly repetitive. It is required and usually matches `output_dimensions`. It may be removed or simplified in future versions; include it for now.
