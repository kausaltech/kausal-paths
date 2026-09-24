# Variant choices and the forecast boundary

Two changes towards letting an action carry its own effect table instead of
reading its row out of a wide shared dataset; datasets owned by a node are the
third, still being designed. The motivating case is the Mainz KNSV measures
(the city administration's own climate-neutrality measures):

- **Wide tables.** Five `mainz/knsv_massnahmen_*` datasets each hold several
  actions. A `measure` dimension mirrors the action list, and every action
  filters out its own row.
- **Repeated forecast year.** `forecast_from: 2025` is repeated on 31 bindings.
- **Variant switch.** 17 hidden `selected_number` parameters index into a
  `categories` string. Every scenario has to set all of them, and all of them
  must agree.

## The forecast boundary

A binding that says nothing about where its forecast begins gets the year from
the first of these that gives one:

1. the binding's own `set_forecast_from`;
2. the dataset's `Dataset.spec['forecast_from']`;
3. with the instance feature `forecast_after_maximum_historical_year`, the year
   after `maximum_historical_year`, because the model's history ends there.

A `Forecast` column in the data still wins over all three. The instance
default is opt-in, since some instances rely on every dataset year counting as
historical. It enters the pipeline as a `set_forecast_from` operation
(`DatasetWithFilters.apply_forecast_defaults`), so dataset hashes and the
prepared store follow it.

## Choice parameters

- `ChoiceParameter` (`type: choice`) holds the id of one of its `choices`.
- `DimensionCategoryParameter` (`type: dimension_category`) takes its choices
  from a dimension's categories, so labels stay live, and it validates its
  value against them once the context is known.

An instance can now define global parameters of its own by giving their
`type`. Before this, only parameters defined in code could be global.

```yaml
params:
- id: knsv_variant
  type: dimension_category
  dimension: measure_scenario
  value: szenario_1
scenarios:
- id: knsv_szenario_2
  params:
  - id: knsv_variant
    value: szenario_2
```

GraphQL exposes them as `ChoiceParameterType` (with `choices` and
`dimensionId`), and they are set with `stringValue`.

## Choosing a category: `select_category`

`select_category(x, measure_scenario=knsv_variant)` keeps one category and drops
the dimension. The category is either a quoted id or a parameter. It is a
pipeline operation (`SelectCategoryOperationSpec`), a `FormulaNode` function,
and one shared implementation (`nodes.transforms.select_category`).

The selection belongs to the computation, not to the binding. Dataset frames
are scenario-independent and cached as such, and a parameter-dependent filter
on a binding would break that. A formula node that selects by a global
parameter depends on it: `Node.referenced_global_parameters()` lists the
parameter, and the loader subscribes the node to it.

## Not built yet

- **The editor side.** The create-parameter mutation for the new types.
- **`GenericAction`.** It still selects variants with
  `select_variant`/`selected_number`. Moving the KNSV actions to
  `FormulaAction` plus `select_category` is the migration that would remove
  them.
- **`FrameworkMeasureDVCDataset2`.** It never applied a dataset-level
  `forecast_from`, and it still doesn't.
