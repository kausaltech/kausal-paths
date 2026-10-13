# Kausal Paths — general model design instructions

Instructions for designing model YAML files and datasets for participatory climate action modeling.

---

## Model philosophy

### Core purpose

The models are built to increase understanding among decision makers, civil servants, citizens, and stakeholders. Many models focus on city-level topics, but the platform itself is more generic.

Importantly, the purpose is not primarily to convince stakeholders or force them to adopt a consensus. Instead, it offers a platform for stakeholders to bring their ideas, concerns, and objections to the table and try to convince "us"—an imaginary group sharing common values.

The platform operationalizes the logic of **Theory of Change**: every model makes explicit the causal chain from action to outcome, together with the assumptions that must hold for change to occur. Unlike conventional Theory of Change diagrams, Kausal Paths encodes these causal chains computationally, making them testable, comparable across scenarios, and sensitive to stakeholder input.

### Common values

- Less suffering is good, irrespective of whose suffering we can reduce.
- Future generations should have opportunity for living good lives.
- Impartial consideration of all people.
- Intergenerational equity.

> **Note:** The common values are a working proposal and a starting point for deliberation — not a discovered or politically neutral foundation. They reflect a broadly cosmopolitan-consequentialist orientation. Alternative value frameworks are explicitly accommodated as model variants (see Value framework approach below).

### Value framework approach

The model uses common values as the baseline scenario. Alternative value frameworks (libertarian, localist, etc.) can be explicitly modeled as variants. Model outputs show: "Under framework X, outcome Y violates value Z." This makes transparent that policy disagreements are sometimes about values, sometimes about facts.

### Stakeholder representation

Different stakeholders have different beliefs about causal relationships. These are encoded as categorical dimensions (e.g. `hypotheses`) on nodes. Stakeholders can select their belief categories and see resulting outcomes. The same computational model accommodates different worldviews.

### Objections handling

Making the type of objection visible is as important as handling it correctly. When a stakeholder objects to a policy recommendation, the nature of the objection determines the appropriate response — and conflating the three types leads to confused deliberation.

**Empirical disputes** are claims about how the world works that can be investigated with evidence. Treatment: model them as causal chain parameters or uncertainties; explore with sensitivity analysis.

**Value conflicts** are claims rooted in value frameworks that conflict with or are orthogonal to common values. Treatment: flag as alternative value framework; run scenarios with different value weights; output shows what values were violated.

**Interests as values** are claims that appear principled but primarily protect material interests. Treatment: separate "How much will X lose?" (empirical) from "Should we care?" (normative). Within the common-values framework, weigh economic loss against suffering reduction.

> **Note for facilitators:** Identifying an objection as primarily interest-based is a sensitive move that requires care. The model helps by quantifying impacts separately from normative weighting — making it easier to acknowledge losses honestly while keeping the value question distinct.

### Methodological context

Kausal Paths builds on and connects several established methodological traditions:

- **Theory of Change** (Weiss, 1995; Funnell & Rogers, 2011): The platform makes causal chains and underlying assumptions explicit and testable, extending standard Theory of Change methodology into a computational and participatory setting.
- **Value of Information** (Howard, 1966; Claxton, 1999): Sensitivity analysis in the model identifies which uncertain parameters most affect outcomes — equivalent to identifying parameters with high Expected Value of Perfect Partial Information (EVPPI). This guides where better evidence or deeper deliberation would have the most impact.
- **Structured Decision Making** (Gregory et al., 2012): The platform shares the SDM emphasis on separating facts from values, making objectives explicit, and structuring stakeholder engagement around a transparent analytical framework.

### Generic models and scaling

A key challenge in participatory modeling is the barrier to entry: building a rigorous causal model from scratch requires expertise and time that most cities do not have. Kausal Paths addresses this through a library of **generic models** covering common city-level climate actions — transport, buildings, energy, land use, and others.

Each generic model provides:
- A pre-built causal chain with documented assumptions
- Uncertainty ranges informed by current research literature
- Common value framings and typical stakeholder objections encoded as argument nodes
- Standard outcome nodes and scenario definitions

Cities use generic models as a **scaffold**: a well-researched starting point that they adapt to local circumstances, stakeholder input, and political priorities. The scaffold makes building possible — it does not determine what gets built.

**Implications for model designers:**
- Design generic models to foreground their own assumptions, making it easy for city users to identify and challenge them.
- Use argument nodes generously to document known empirical disputes, value conflicts, and stakeholder concerns from the research literature.
- Prefer modular structures that cities can extend or adjust without restructuring the core model.
- Generic models improve over time as cities contribute local adaptations — design for reusability from the start.

**For outside experts:** Generic models provide a concrete and meaningful way for researchers, sector specialists, and NGOs to contribute to the platform without being tied to a single city project. Contributing to a generic model means contributing to every city that uses it.

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

**Dimension arithmetic:** Multiplication (and division) results in the union of dimensions from both inputs. Addition (and subtraction) requires dimensions to match; incompatible dimensions must be aggregated first. Also categories are treated differently: multiplication and division merge with inner join resulting in dropout of categories that do not exist in both sides. In contrast, addition and subtraction use outer join and the emerging null values are treated as zeros.

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

**Edge dimension mappings - completeness rule:** When specifying `to_dimensions` in an edge (either in `input_nodes` or `output_nodes`), you must specify ALL dimensions that the target node expects. If you specify any dimension in `to_dimensions`, you cannot omit others—the mapping must be complete. Example: If target node has `output_dimensions: [ghg, scope, emission_sectors]` and you need to map `scope`, you must also explicitly map `ghg` and `emission_sectors` in the `to_dimensions` list, even if they pass through unchanged.

**Dimensions in node definitions vs edge mappings:** Only include dimensions in a node's `output_dimensions` if those dimensions exist in the node's `input_dimensions` or come from its input nodes. If a dimension needs to be added when connecting to a downstream node (e.g., assigning a `scope` category), add it via the edge mapping (`to_dimensions` in the target node's `input_nodes` or the source node's `output_nodes`), not in the source node's `output_dimensions`. This keeps node definitions accurate to their actual data structure.

**Edge definition uniqueness:** Edges must be defined in exactly one place—either in the source node's `output_nodes` section OR in the target node's `input_nodes` section, but never both. If you define an edge in both places, you'll get a "Duplicate edge definition" error.

**Category assignment in edge `to_dimensions` — use `categories:`, not `assign_category:`:** When an edge needs to assign a fixed category to a dimension (e.g. routing a value with `cost_type = rent`), use `categories: [rent]` inside the `to_dimensions` entry. The `assign_category:` field only works in dataset filter context (`DimensionDatasetFilterDef`); in edge definitions it is silently ignored, causing "Dimensions do not match" errors at runtime.

```yaml
# Correct — use categories:
to_dimensions:
- id: cost_type
  categories: [rent]
- id: stakeholder
  categories: [tenant]

# Wrong — assign_category is silently ignored in edge context
to_dimensions:
- id: cost_type
  assign_category: rent   # does nothing here
```

**Multi-metric actions backed by DVC datasets:** A single action can output two or more metrics with different quantities and units. Declare the metrics under `output_metrics:` and route each to a different downstream node via `metrics: [metric_id]` on the `output_nodes` entry. Back the action with a DVC dataset that has one row per metric per dimension combination (wide year columns). Set `interpolate: true` on the `input_datasets` entry if the dataset only contains sparse key years.

```yaml
- id: my_action
  type: simple.AdditiveAction
  output_metrics:
  - id: energy_reduction
    unit: kWh/m**2/a
    quantity: energy
  - id: additional_cost
    unit: EUR/m**2/month
    quantity: unit_price
  input_datasets:
  - id: my_namespace/my_dataset
    forecast_from: 2025
    interpolate: true
  output_nodes:
  - id: heat_demand_node
    metrics: [energy_reduction]
  - id: cost_node
    metrics: [additional_cost]
```

**`interpolate: true` in `input_datasets` must be set explicitly:** Datasets with only a few key years (e.g. 2024, 2030, 2040, 2050) will show zero effect in all intermediate years unless `interpolate: true` is set on the `input_datasets` entry. This triggers linear interpolation across all model years. It is not inferred from the data sparsity — you must set it deliberately.

**Preserving dimensions through chains:** When removing dimension flattening to preserve dimensions through a chain of nodes: (1) Remove `flatten: true` from edges where dimensions were being flattened, (2) Add the dimensions to all intermediate nodes' `input_dimensions` and `output_dimensions`, (3) Ensure edge mappings (`to_dimensions`) properly pass through all dimensions at each step, (4) Remember the completeness rule—if you specify any `to_dimensions`, specify all of them.

**Formula clarity:** Use descriptive tags (e.g. `[baseline]`, `[reduction]`, not `[x]`, `[y]`). Break complex calculations into multiple nodes. Document units in node names when ambiguous.

**Argument nodes:** Connect them to the most directly relevant computational node(s). Use HTML in descriptions for rich documentation. List related nodes, stakeholders, and hypotheses in the description. They can connect to multiple nodes if the objection relates to several. In generic models, use argument nodes proactively to document known empirical disputes and stakeholder concerns from the research literature — this is part of the value generic models provide.

**Naming conventions:** IDs in snake_case; names in sentence case; use single quotes for strings; only quote strings that contain special characters (e.g. `:`, `#`).

---

## Style guidelines

**Indentation:** When list items are on separate rows starting with `-`, indent the `-` at the same level as the list title. Correct: `my_list:` then `- item1` / `- item2` at the same indent as `my_list`. Incorrect: extra indent under `my_list`.

**Dimension references:** Use multi-row format with explicit `id` for `to_dimensions` and `from_dimensions`. Correct: `to_dimensions:` then `- id: dimension1` etc. Incorrect: `to_dimensions: [dimension1, dimension2]`.

**String quoting:** Quote only when necessary (special characters). No quotes needed for e.g. `name: Traffic volume`, `unit: vehicles/hour`, `formula: baseline * (1 - reduction)`. Quotes needed for e.g. `name: 'OBJ-CC13: Traffic displacement'`, `color: '#FF0000'`, or descriptions containing apostrophes. Prefer single quotes unless double quotes are needed (e.g. for an apostrophe).

---

## Best practices

**Start simple:** begin with a minimal viable model and expand incrementally. Test early: run the model as soon as possible to find issues. Document decisions: record rationale for design choices. Mark workarounds: use FIXME for temporary solutions. Keep structure explicit: include all intermediate nodes. Refine progressively: start with qualitative edge tags and add formulas later; start with placeholder data and refine values later; start with simple scenarios and add complexity later.

**Generic model design:** When building a generic model intended for reuse across cities, design it to surface its own assumptions — cities should find it easy to identify what to challenge and customize. Use argument nodes generously to document known empirical disputes, value conflicts, and stakeholder concerns drawn from research literature. Write node descriptions as if explaining to a city expert who did not build the model. Prefer modular structures that can be extended locally without restructuring the core. Mark parameters that are most likely to vary by local context with clear documentation.

**Sensitivity as priority signal:** Parameters where model outputs are highly sensitive to input values deserve explicit attention — in documentation, in argument nodes, and in facilitation. These are the parameters where better local data or deeper stakeholder deliberation will have the most impact. Identifying them is equivalent to Value of Information analysis: invest attention where uncertainty matters most.

---

## Key concepts

**Participatory modeling:** The model is designed for stakeholder engagement, not expert prediction. The goal is to help stakeholders understand the implications of their beliefs and values. The goal is not to convince anyone or find an "optimal" policy.

**Destructive policies:** However, one goal is to identify policies that appear tempting but are actually, based on current scientific knowledge and commonly shared values, performing much worse than people think. These are called "destructive policies" and the models should highlight the discrepancy between perceived and actual performance.

**Theory of Change:** The platform implements Theory of Change computationally. Causal chains from actions to outcomes are explicit, underlying assumptions are documented (including in argument nodes), and the model can be run under different causal hypotheses. This extends conventional Theory of Change by making assumptions testable and quantitatively comparable across stakeholder worldviews.

**Value of Information:** Sensitivity analysis identifies which parameters most affect model outcomes — equivalent to estimating Expected Value of Perfect Partial Information (EVPPI). Parameters where conclusions are highly sensitive deserve more attention: better data, deeper deliberation, or explicit flagging as key uncertainties for decision makers.

**Hypothesis dimensions:** Encode competing empirical beliefs as categories in a dimension. Different stakeholders select different categories to see "their" outcomes. The model structure stays the same; parameters vary.

**Value weights:** MCDM converts outcomes to utilities. Different stakeholders weight outcomes differently.

**Argument nodes:** They are documentation nodes that do not affect calculations. They link stakeholder concerns to relevant computational nodes and enable traceability (e.g. "This objection relates to these parameters"). In generic models, they also capture known objections and disputes from the research literature, making the model's epistemic grounding visible.

**Outcome nodes:** Special nodes marked `is_outcome: true` for UI presentation. Use AdditiveNode to sum components.

**Generic models:** Pre-built causal models covering common city-level climate actions. They serve as a scaffold — a well-researched starting point that cities adapt to local circumstances, stakeholder input, and political priorities. The scaffold makes participation possible; it does not determine what gets built. Generic models should be designed for reusability, with assumptions foregrounded and argument nodes documenting known debates.

---

## Technical notes

**Pint integration:** The Pint library handles all unit conversions automatically. Never put conversion constants in formulas. Trust Pint to convert e.g. /hour to /a, g to t.

**Dimension broadcasting:** When multiplying nodes with different dimensions, the result has both; the system broadcasts appropriately (similar to NumPy). Example: [time, vehicle] × [vehicle, fuel] → [time, vehicle, fuel].

**PathsDataFrame:** Internal data structure wrapping the dataframe; it has built-in unit awareness via Pint and handles dimension alignment automatically.

**DAG execution:** Nodes are calculated in topological order (dependencies first). Each node is calculated once per run. Comparisons are done by running the model twice (e.g. baseline vs scenario).

**input_dimensions (legacy):** The `input_dimensions` field is legacy and mostly repetitive. It is required and usually matches `output_dimensions`. It may be removed or simplified in future versions; include it for now.
