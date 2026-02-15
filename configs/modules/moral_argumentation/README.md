# Moral argumentation in local climate policy (generic model)

This module specifies a **generic model for moral argumentation** in local climate policy: global values that motivate local action via the claim that *climate change is a global system affected by everyone, so nobody can exempt themselves*. It defines the **values**, the **duties** they imply, and how to implement them in the kausal-paths modelling platform.

**Alignment with kausal_paths.yaml (model design):** (1) Physical world is the basis — duty/contribution nodes (`fair_share_met`, `emissions_acceptable`) depend on physical outcomes only. (2) Moral layer is on top — values and arguments interpret those outcomes as duties and justify priorities. (3) This value module is separate and reusable across instances. (4) Text: values and arguments in node descriptions and argument nodes. (5) Calculations: duty evaluations plus value weights and utility nodes assess duty fulfilment and moral utility per scenario. (6) User-adjustable parameters: each value has a weight param; changing weights changes duty/utility outputs. (7) Priorities: impact_overview with effect_node = moral_value_profile ranks actions by moral value impact.

---

## 1. Background: platform representation

In kausal-paths:

- **Argument nodes**: `type: generic.GenericNode`, `quantity: argument`. They do not compute numbers; they link via `output_nodes` to **evaluation nodes** that operationalize whether a duty is met.
- **Evaluation nodes**: e.g. `formula.FormulaNode` or `generic.ConstantNode` with `is_outcome: true`. They represent *whether a duty is fulfilled* (e.g. fair share, acceptable emissions).
- **Values** can be represented implicitly (in argument descriptions) or via a dimension tagging arguments.

The causal/narrative flow is: **Value (why we care) → Argument (claim) → Duty evaluation (operationalization)**.

---

## 2. Global values motivating local action

These are **moral values** that support the “no exemption” idea and thus motivate local climate action:

| Value | Description | Role in “no exemption” |
|-------|-------------|--------------------------|
| **Fairness** | No one who contributes to the problem may opt out of the solution. | Direct: fair share is owed regardless of size. |
| **Reciprocity** | We expect others to act; we must act ourselves. | “Do unto others” → we cannot demand action and refuse it. |
| **Stewardship / intergenerational** | We hold the planet in trust for future generations. | We cannot exempt ourselves from preserving a livable world. |
| **Non-harm** | Contributing to avoidable harm is wrong. | Every jurisdiction’s emissions contribute to harm; no exemption. |
| **Common heritage** | Climate is a shared commons. | No right to overuse; duty to stay within one’s share. |

In the **calculation layer** (`value_weights.yaml`), each value is a ValueAction with a *constant* (weight) parameter and a utility node (e.g. `utility_fairness` = weight × duty). Setting a value's weight to 0 removes that value from priorities.

### 2.1 Value node naming and abbreviations

**Abbreviations:** Value nodes are not arguments or objections, so ARG-n / OBJ-n are not used. Use **VAL-n** (e.g. VAL-1, VAL-2) as a neutral short reference for value nodes in descriptions, lists, and traceability. Numbering can be per instance or follow a shared convention (e.g. VAL-1 = fairness/reciprocity, VAL-2 = conditional cooperation, VAL-3 = stewardship).

**Names:** The display name of a value node should be **a sentence that describes the sentiment of a person who holds that value**. That way:
- The name stays neutral (no “problem” or “objection” framing).
- A high weight clearly means “I strongly hold this sentiment.”
- Different worldviews can recognise themselves without feeling judged.

Examples:
- Instead of “Free rider problem” → “Whether others act matters to me when I decide.” (VAL-2)
- Instead of “Universal responsibility” → “We should do our fair share even if others do not.” (VAL-1)
- Instead of “Intergenerational responsibility” → “We should leave a habitable world for future generations.” (VAL-3)

Optional format for the name field: **VAL-n: &lt;sentence&gt;** so both the abbreviation and the sentiment appear in lists and the UI.

---

## 3. Duties implied by these values

**Duties** are what we owe; they can be operationalized as **evaluation nodes** in the model.

| Duty | Meaning | Typical operationalization in model |
|------|--------|-------------------------------------|
| **Fair-share duty** | Do one’s proportional part of global mitigation. | `fair_share_met`: achieved reduction ≥ required (e.g. global_reduction_needed). |
| **No-exemption duty** | Act even if our impact is small. | Same as universal responsibility: act regardless of others; `emissions_acceptable` and `fair_share_met` show we are doing our part. |
| **Intergenerational duty** | Do not exceed a sustainable carbon budget. | `emissions_acceptable` (per capita ≤ threshold); full version: cumulative budget / carbon_budget_remaining. |
| **Non-free-riding** | Do not condition our action on others acting first (unless one explicitly adopts a conditional cooperation threshold). | `our_city_threshold`, `should_we_cooperate`; deontological view: duty exists regardless → ARG-1. |

---

## 4. Implementation in the platform

### 4.1 Existing pattern (Equalia)

- **Evaluation nodes** (duties): `emissions_acceptable`, `fair_share_met`, `commons_survives`, `should_we_cooperate`, `cooperation_gap`.
- **Value nodes** (e.g. VAL-1, VAL-2, VAL-3) are ValueActions with `output_nodes` pointing to these evaluations. Names are sentences expressing the sentiment of someone who holds that value. Descriptions state the **value** (e.g. fairness, reciprocity) and the **claim** (e.g. “moral duty exists independent of impact”).
- **Parameters** users can debate: `acceptable_emissions_per_person`, `global_reduction_needed`, `our_city_threshold`, and each value node's weight (constant).

So: **values** are **value nodes** (name = sentiment sentence, param = weight); **duties** are the **evaluation nodes**; value nodes connect to them in the graph.

### 4.2 Optional: explicit values dimension

To tag arguments by value, add a dimension, e.g.:

```yaml
dimensions:
- id: moral_value
  label_en: Moral value
  categories:
  - id: fairness
    label_en: Fairness
  - id: reciprocity
    label_en: Reciprocity
  - id: stewardship
    label_en: Stewardship / intergenerational
  - id: non_harm
    label_en: Non-harm
  - id: common_heritage
    label_en: Common heritage
```

Then give argument nodes `input_dimensions: [moral_value]` and tag each argument with the appropriate value(s) (e.g. via dataset or constant).

### 4.3 Optional: duty in value profile (greentransition-style)

To aggregate “duty met” into a multi-criteria value profile:

- Keep duty evaluations as above.
- Add utility nodes, e.g. `utility_fair_share` with `input_nodes: [fair_share_met]` and a weight parameter.
- Feed into a `value_profile` (or similar) node so that duty is one dimension of local motivation alongside prosperity, legality, etc.

### 4.3 Operationalizing different value structures (weight-based)

**Problem:** If a value is only text (in argument descriptions), a user who disagrees with that value cannot see priorities that reflect "I give this value zero weight." Priorities must respond to value weights.

**Solution: dual representation of values**

1. **Text layer (unchanged):** Values appear in **argument node** descriptions and in `output_nodes` links to duty evaluations. This provides narrative, traceability, and explanation ("this duty is justified by fairness").
2. **Calculation layer (new):** Each value has a **user-adjustable weight parameter** and a **utility node** that multiplies that weight by a **contribution** (the outcome of a duty evaluation or other physical outcome). A **moral value profile** node sums these utilities. Priorities (e.g. impact_overview with effect_node = moral_value_profile) are then driven by this aggregate. When the user sets a value's weight to 0, that value no longer contributes to the profile, so priorities change.

So the same value appears in two places:

| Layer        | Role                                                                 |
|--------------|----------------------------------------------------------------------|
| **Text**     | Each ValueAction has a name (sentiment sentence) and description. |
| **Calculation** | ValueAction outputs its weight (constant param); utility node = weight × duty → moral_value_profile (drives priorities). |

**Required convention:** The instance must define **duty/contribution nodes** (`fair_share_met`, `emissions_acceptable`). The module defines ValueActions and utility nodes. Mapping:

- **Fairness** → `value_fairness` × `fair_share_met`.
- **Reciprocity** → `value_reciprocity` × `fair_share_met`.
- **Stewardship** → `value_stewardship` × `emissions_acceptable`.
- **Non-harm** → `value_non_harm` × `emissions_acceptable`.
- **Common heritage** → `value_common_heritage` × `emissions_acceptable`.

**User interaction:** User adjusts each value's slider (the ValueAction's *constant* parameter). No global weight params; each value node carries its own weight. Setting a value's weight to 0 removes that value from the moral profile and thus from priorities.

### 4.4 Optional: moral profile in combined value profile (greentransition-style)

To combine moral priorities with economic and other criteria: build the moral value profile as in 4.3, then feed it as one input into the instance's main `value_profile`, or use a dedicated impact_overview with effect_node = moral_value_profile for "priorities by moral values only."

---

## 5. Summary

- **Global values** (fairness, reciprocity, stewardship, non-harm, common heritage) motivate the claim that **no one can exempt themselves** from climate action.
- **Duties** are: fair share, no exemption, intergenerational, non–free-riding. They are implemented as **evaluation nodes** (e.g. `fair_share_met`, `emissions_acceptable`).
- **Implementation**: (1) **ValueActions** (one per value topic) with name = sentiment sentence and *constant* param = weight. (2) **Utility nodes** (formula: weight × contribution) fed by each ValueAction and the duty node. (3) **moral_value_profile** = sum of utilities. Setting a value's weight to 0 removes that value from priorities.

**Files in this module:**

- `value_weights.yaml` — ValueActions (VAL-1–VAL-5), utility nodes, and `moral_value_profile`. No global weight params; each value's weight is the ValueAction's slider.

### Including the module

In your instance config, add an action_groups entry for moral_values, then include the module:

```yaml
action_groups:
- id: moral_values
  name_en: Moral values
  color: '#9B59B6'

include:
- file: modules/moral_argumentation/value_weights.yaml
  allow_override: true
```

**Requirements:**

- The instance must define duty nodes `emissions_acceptable` and `fair_share_met`, and action_groups with `id: moral_values`. If your instance uses different duty node IDs, override the utility nodes’ `input_nodes`.

**Priorities:** Add an impact_overview with effect_node = moral_value_profile. Users change each value's slider (ValueAction constant param); the profile and ranking update automatically.
