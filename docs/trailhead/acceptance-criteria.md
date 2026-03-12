🪤 Steps in creating a Paths model
==================================

Engineering-level user journey for the Paths admin UI. Based on discovery sessions by Stefan Cave and Michael Mechenich, 23 Feb 2026 and 26 Feb 2026. Steps are ordered to reflect the recommended build sequence: establish the baseline with formula nodes first, then layer in action nodes.

MVP – Required for initial release

Future – Post-MVP enhancement

Offline – Happens outside the admin UI

0. Pre-work Offline
----------------

Before opening the admin UI, the user (typically with help from a customer success manager) prepares their emissions inventory and makes structural decisions about the model. This thinking must be complete before step 1.

| ID | Phase | Acceptance criterion |
| --- | --- | --- |
| AC-0.1 | Offline | User has prepared their emissions inventory (typically in Excel), covering historical years at minimum. |
| AC-0.2 | Offline | User has decided how to divide the model into datasets (e.g., one per GPC sector: stationary, transport). |
| AC-0.3 | Offline | User has decided on dimensions and their categories (e.g., building type: residential / commercial / industry; energy carrier: natural gas / electricity). |
| AC-0.4 | Offline | User knows which metrics they will track (e.g., emissions only; or activity data and emission factors). |
| AC-0.5 | Offline | User knows the model's key year parameters: reference year, min/max historical years, target year, model end year. |

1. Create model instance
---------------------

User creates a new Paths model via an onboarding wizard. This defines global model metadata and year parameters. The resulting instance appears in the admin UI and becomes the container for all subsequent objects.

| ID | Phase | Acceptance criterion |
| --- | --- | --- |
| AC-1.1 | MVP | A "Create new model" action in the admin UI launches an onboarding wizard (modal or dedicated page). |
| AC-1.2 | MVP | User must provide a model/instance name (free text, required). |
| AC-1.3 | MVP | User must select a default display language and can optionally add one or more additional supported languages. |
| AC-1.4 | MVP | User must define five year parameters: **reference year** (standalone baseline bar), **min historical year**, **max historical year**, **target year**, and **model end year**. |
| AC-1.5 | MVP | System validates chronological ordering: reference year ≤ min historical year ≤ max historical year < target year ≤ model end year. Saving is blocked if the constraint is violated, with a clear error message. |
| AC-1.6 | MVP | On save, the new model instance appears in the admin UI instance list and can be selected to begin configuration. |
| AC-1.7 | Future | User can select a visual theme (colour palette, logo) for the public-facing UI during setup. |

2. Create datasets
---------------

User creates one or more datasets, the foundational data containers. Each dataset has metrics (what is measured), dimensions (how the data is categorised), a first forecast year, and the actual data values. Both dimensions and metrics are global to the model and can be created inline during dataset creation, then reused across all subsequent datasets. A model cannot have any nodes without at least one dataset.

| ID | Phase | Acceptance criterion |
| --- | --- | --- |
| AC-2.1 | MVP | User can create a new dataset and must provide a name. |
| AC-2.2 | MVP | User must define at least one metric per dataset. Each metric requires: a label (free text, e.g., "emissions", "emission factor natural gas") and a unit (free text, e.g., "tCO₂e/year"). |
| AC-2.3 | MVP | User can define multiple metrics per dataset (e.g., "emission factor electricity" and "emission factor natural gas" where units differ). |
| AC-2.4 | MVP | Metrics are global to the model instance. When defining metrics for a dataset, user can select from previously created metrics or create a new one inline. A newly created metric is saved globally and immediately available to all subsequent datasets. |
| AC-2.5 | MVP | Dimensions are global to the model instance. When assigning dimensions to a dataset, user selects from the global dimensions list. |
| AC-2.6 | MVP | User can create a new dimension inline while building a dataset (in-context addition): user provides a dimension name and one or more category values. The new dimension is saved globally and immediately available to other datasets. |
| AC-2.7 | MVP | Best-practice guidance: 1 to 5 dimensions per dataset is recommended. System may surface a soft warning (non-blocking) if more than 5 are assigned. |
| AC-2.8 | MVP | User defines the **first forecast year** for the dataset. This value propagates automatically through the node graph: a node's first forecast year is the minimum first forecast year across all its input datasets. If metrics within the same dataset require different forecast start years, they should be placed in separate datasets. |
| AC-2.9 | MVP | User can enter numeric data values for each year x dimension category combination. System rejects non-numeric input in value cells. |
| AC-2.10 | MVP | A dimension or metric defined in one dataset is immediately reusable when creating a subsequent dataset; no re-entry required. This prevents typo-based inconsistency across the model. |
| AC-2.11 | MVP | System does not enforce specific unit strings (free-form); there is no blocking validation on metric labels or units at this stage. |
| AC-2.12 | Future | System offers a recognised unit picker with conversion support, rather than free-text unit entry. |

3. Create formula node(s)
----------------------

Formula nodes form the calculation backbone of the model. Build them first to establish the baseline before adding action nodes. Each formula node works with a single quantity, takes one or more labelled input ports (datasets or other nodes), and applies an Excel-style formula. Outcome nodes (a sub-type) are formula nodes marked as model outcomes; they are referenced by impact overviews (step 6) and used for goal tracking.

| ID | Phase | Acceptance criterion |
| --- | --- | --- |
| AC-3.1 | MVP | User enters a node name (free text, required). System auto-generates a unique node ID; user does not set or see this. |
| AC-3.2 | MVP | User selects node type: **Formula**. This is for all non-action calculation nodes. |
| AC-3.3 | MVP | User can enter a description in all supported languages (same as step 1). |
| AC-3.4 | MVP | User selects a **quantity** from a pre-populated dropdown. The full list of recognised quantities (e.g., "emissions", "fuel consumption", "emission factor", "energy") is hardcoded in the Paths system configuration. No user setup is required and no new quantities can be created via the UI. |
| AC-3.5 | MVP | User enters a **unit** (free text, e.g., "tCO₂e/year", "kWh/year"). |
| AC-3.6 | MVP | User assigns **dimensionality** from the global dimensions list. |
| AC-3.7 | MVP | User defines one or more **input ports**. Each port requires: a user-defined label (used in the formula as a variable name) and a source, either a dataset or another node. |
| AC-3.8 | MVP | When the source is a dataset, user selects which metric to pull. System automatically applies `drop_na` for multi-metric datasets. |
| AC-3.9 | MVP | User writes a **formula** referencing input port labels as variables (Excel-style syntax, e.g., `energy_use * emission_factor`). The formula field only allows referencing labels already defined in the node's input ports. |
| AC-3.10 | MVP | User can tick an **"outcome" checkbox** to designate this node as a model outcome. Outcome nodes are referenced by impact overviews (step 6) and used for goal tracking. |
| AC-3.11 | MVP | User can optionally define **goals** on a node: a target value for the target year. Goals are set at the individual node level and multiple target years can be specified. |
| AC-3.12 | MVP | User connects the formula node to one or more **output nodes** from the model's node list. At minimum, a chain must terminate at a net-emissions outcome node. |
| AC-3.13 | MVP | Graphical node editor: user can drag nodes onto a canvas and draw connections by clicking on an output port and dragging to an input port. Input port labels are shown on hover. The graphical editor covers all node types (formula nodes, action nodes, datasets). |
| AC-3.14 | Future | Formula editor validates syntax on save and highlights any port labels referenced in the formula that are not defined as input ports, with a clear error message. |
| AC-3.15 | Future | System warns if input dimensionality and output dimensionality differ in an unexpected way (non-standard transformation). |

4. Define action groups (global)
-----------------------------

Action groups are model-level categories applied to action nodes (e.g., "System retrofits", "Renewable energy production"). They appear as the "type" column in the public action list and support colour coding. They can be defined here upfront or created inline when the first action node is built.

| ID | Phase | Acceptance criterion |
| --- | --- | --- |
| AC-4.1 | MVP | Action groups are global to the model instance and shared across all action nodes. |
| AC-4.2 | MVP | User can create an action group by providing a name and selecting a colour. |
| AC-4.3 | MVP | Action groups can be created inline when creating an action node (in-context addition). The new group is immediately saved globally and available to all subsequent action nodes. |
| AC-4.4 | Future | User can manage action groups (rename, recolour, reorder, delete) from a dedicated global settings section within the admin UI. |

5. Create action node(s)
---------------------

Action nodes represent policy interventions or measures. They appear in the public UI action list. Action nodes support one or more output quantities. The impact overview columns and graphs for these actions are configured in step 6. Impact calculations compare the model output with the action on versus off.

| ID | Phase | Acceptance criterion |
| --- | --- | --- |
| AC-5.1 | MVP | User enters a node name (free text, required). System automatically generates a unique node ID; the user does not need to set or see this. |
| AC-5.2 | MVP | User selects node type from a dropdown: **Action** or **Formula**. Selecting "Action" surfaces action-specific fields. |
| AC-5.3 | MVP | User can enter a description for the node. Description fields are provided for each supported language defined in step 1. |
| AC-5.4 | MVP | User selects an action group from the global list. If no group exists, or a new one is needed, user can create one inline (see AC-4.3). |
| AC-5.5 | MVP | User defines one or more **input ports**. Each port requires: a user-defined label (e.g., "energy_reduction", "cost") and a source, either a dataset or another node. |
| AC-5.6 | MVP | When the source is a dataset, user selects which metric to pull from that dataset. System automatically applies `drop_na` for multi-metric datasets (no user action required). |
| AC-5.7 | MVP | User assigns **dimensionality** from the global dimensions list. The same dimensions apply to both input and output (identical in ~95% of cases; advanced divergence is a future concern). |
| AC-5.8 | MVP | User defines **output quantities**. Action nodes support one or more output quantities; there is no fixed minimum above one. The appropriate number depends on the nature of the action being modelled. |
| AC-5.9 | MVP | User connects the action node to one or more **output nodes** (the nodes that receive this node's data). The output node dropdown shows all existing nodes in the model. |
| AC-5.10 | MVP | Action nodes automatically appear in the public UI action list once the model runs. No separate registration step is required. |
| AC-5.11 | MVP | Impact overview columns in the public UI action list are driven by the impact overviews configured in step 6. The Paths engine calculates each overview automatically by comparing the model output with the action on versus off. |
| AC-5.12 | Future | User can configure custom impact overview columns: choose which outcome nodes to compare, define a label, set units, and define an efficiency formula (e.g., cost per tonne reduced). |

6. Define impact overviews
-----------------------

Impact overviews define the columns displayed in the public UI action list and the graphs shown alongside it (e.g., the MAC curve). Each overview links to specific nodes in the model and specifies a graph type. For MVP, three graph types are supported and the available types are hardcoded in the Paths system. A model can have more than one impact overview.

| ID | Phase | Acceptance criterion |
| --- | --- | --- |
| AC-6.1 | MVP | User can define one or more impact overviews for the model instance. A model without any impact overviews will not display cost-related columns or MAC curve graphs in the public UI. |
| AC-6.2 | MVP | Each impact overview has a graph type. The supported types for MVP are: **cost efficiency**, **cost benefit**, and **return on investment**. These types are hardcoded in the Paths system; no additional types can be created via the UI for MVP. |
| AC-6.3 | MVP | For a **cost efficiency** overview, user selects an effect node (e.g., a node representing emissions reduction) and a cost node, and provides a units label for the visualisation (e.g., "euros per tonne CO₂e"). |
| AC-6.4 | MVP | For a **cost benefit** overview, user selects the relevant cost node and provides a units label. |
| AC-6.5 | MVP | For a **return on investment** overview, user selects the relevant nodes and provides a units label. |
| AC-6.6 | MVP | Impact overviews drive the columns displayed in the public UI action list and the MAC (marginal abatement cost) curve graph. |
| AC-6.7 | MVP | A model instance can have more than one impact overview of the same or different types (e.g., one cost efficiency overview and one cost benefit overview, or multiple cost efficiency overviews for different stakeholder groups). |
| AC-6.8 | Future | User can define additional graph types beyond the three MVP types, with configurable axes, node mappings, and labels. |

7. Define scenarios
----------------

Scenarios let users define named combinations of actions being on or off. They appear as the scenario selector in the public UI, allowing end users to compare different action implementation plans (e.g., baseline with no actions, full action plan, committed measures only). For MVP, scenarios cover action on/off toggling, which addresses approximately 90% of city modelling use cases.

| ID | Phase | Acceptance criterion |
| --- | --- | --- |
| AC-7.1 | MVP | User can define one or more named scenarios for the model instance. |
| AC-7.2 | MVP | Each scenario has a name (required) and an optional description. |
| AC-7.3 | MVP | For each scenario, user specifies which action nodes are active (on) and which are inactive (off). Action nodes not explicitly set to on are treated as off by default. |
| AC-7.4 | MVP | Multiple scenarios can be defined per model instance (e.g., baseline with all actions off, full action plan with all on, a "committed" subset with only confirmed measures active). |
| AC-7.5 | MVP | User can edit and delete existing scenarios. |
| AC-7.6 | Future | Scenarios can include per-scenario values for global parameters and node parameters (for models where named parameters must vary between scenarios). |
| AC-7.7 | Future | Scenarios can include slider configurations for action nodes that support interactive adjustment in the public UI. |

8. Validate model
--------------

Validation happens at two levels: real-time structural checks as the user connects nodes in the graphical editor, and a full model run for catching calculation errors. The goal is that dimensional mismatches and formula errors are caught immediately as the user builds the graph, rather than discovered only after a complete model run. The data frame inspector allows intermediate outputs to be verified at any node.

| ID | Phase | Acceptance criterion |
| --- | --- | --- |
| AC-8.1 | MVP | The graphical node editor performs real-time validation as the user connects nodes. When two nodes are connected, the system immediately checks dimensional compatibility and flags any mismatch before the user proceeds. |
| AC-8.2 | MVP | If a dimensional mismatch is detected at the point of connection, the affected connection is highlighted and an explanatory error message is shown. The user does not need to wait for a full model run to discover the issue. |
| AC-8.3 | MVP | When a formula is saved on a formula node, the system checks that the formula is syntactically valid and can produce an output. The user is shown an error if the formula will not work. |
| AC-8.4 | MVP | If a node produces a runtime error during model execution, the error is surfaced in the editor with a clear reference to the specific node that caused it, so the user can locate and fix the issue without reading a raw error log. |
| AC-8.5 | MVP | A **"Test model"** button is available to trigger a full model run on demand. This serves as a fallback for catching errors that require a complete calculation pass and may be too computationally expensive to run continuously during editing. |
| AC-8.6 | MVP | For any node, user can view the computed **data frame** (a table of values indexed by year x dimension category) directly in the admin UI, without needing terminal access. |
| AC-8.7 | MVP | The data frame view shows: year, dimension category values, numeric value, and whether each row is historical or forecast. |
| AC-8.8 | MVP | User can navigate from an output node to any upstream (input) node to trace calculation errors through the tree. |
| AC-8.9 | Future | Nodes with calculation errors, null values, or broken input connections are visually flagged in the node list/editor (e.g., red border, warning icon). |
| AC-8.10 | Future | A model-wide validation panel surfaces a summary of issues: missing connections, unresolved quantities, year-range mismatches, or null-value propagation. |
| AC-8.11 | Future | In the graphical node editor, each node card has an inline table icon; clicking it opens the data frame viewer without leaving the canvas. |
| AC-8.12 | Future | A module library provides a browsable collection of pre-built node configurations. Users can view a module as a reference example to inform how they set up their own nodes. |
| AC-8.13 | Future | Users can copy a module from the library and insert it into their model instance. The module node structure and connections are placed on the canvas; the user then configures the dimension assignments and dataset mappings for the imported nodes. |

**Engineering note:** Michael confirmed that the graphical editor should give immediate feedback when two nodes are connected with mismatched dimensions. Waiting for a full model run to discover errors was described as "going back to the 90s". The MVP real-time validation (AC-8.1 to AC-8.5) should therefore be treated as high-priority alongside the data-frame viewer (AC-8.6 to AC-8.8). Exact implementation approach (lightweight structural check vs. continuous model run) to be confirmed with Jouni during development.
