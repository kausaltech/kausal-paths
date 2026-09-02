"""Tests for mapping MeasureTemplate UUIDs onto the model nodes that hold their values."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import polars as pl
import pytest

from common.polars import DataFrameMeta, to_ppdf
from frameworks.datasets import FrameworkMeasureDVCDataset2
from frameworks.models import FrameworkConfig, NodeDimensionSelection
from frameworks.schema import MeasureType
from nodes.constants import VALUE_COLUMN, YEAR_COLUMN
from nodes.units import unit_registry

if TYPE_CHECKING:
    from common.polars import PathsDataFrame
    from nodes.context import Context

pytestmark = pytest.mark.django_db

UUID_A = '11111111-1111-1111-1111-111111111111'
UUID_B = '22222222-2222-2222-2222-222222222222'
UUID_C = '33333333-3333-3333-3333-333333333333'


def _frame(rows: list[dict[str, Any]], dims: list[str]) -> PathsDataFrame:
    df = pl.DataFrame(rows)
    meta = DataFrameMeta(
        units={VALUE_COLUMN: unit_registry.parse_units('kWh')},
        primary_keys=[YEAR_COLUMN, 'uuid', *dims],
    )
    return to_ppdf(df, meta)


def _binding(df: PathsDataFrame | None, tags: list[str] | None = None) -> Any:
    return SimpleNamespace(id='ds', tags=tags or [], get_uuid_frame=lambda: df)


def _uuids_from_binding(ds: Any) -> list[tuple[str, dict[str, str] | None]]:
    return FrameworkConfig._get_measure_template_uuids_from_binding(ds)


def test_binding_without_uuid_column_claims_nothing():
    assert _uuids_from_binding(_binding(None)) == []


def test_single_uuid_binding_needs_no_dimensions():
    df = _frame(
        [
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, VALUE_COLUMN: 1.0},
            {YEAR_COLUMN: 2021, 'uuid': UUID_A, VALUE_COLUMN: 2.0},
        ],
        dims=[],
    )
    assert _uuids_from_binding(_binding(df)) == [(UUID_A, None)]


def test_uuids_are_separated_by_dimension_category():
    df = _frame(
        [
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, 'energy_carrier': 'electricity', VALUE_COLUMN: 1.0},
            {YEAR_COLUMN: 2021, 'uuid': UUID_A, 'energy_carrier': 'electricity', VALUE_COLUMN: 2.0},
            {YEAR_COLUMN: 2020, 'uuid': UUID_B, 'energy_carrier': 'heat', VALUE_COLUMN: 3.0},
        ],
        dims=['energy_carrier'],
    )
    assert dict(_uuids_from_binding(_binding(df))) == {
        UUID_A: {'energy_carrier': 'electricity'},
        UUID_B: {'energy_carrier': 'heat'},
    }


def test_uuids_sharing_a_dimension_category_are_both_dropped():
    """
    Two measures under one category cannot be told apart, so neither is claimed.

    Values come from the node's output, which carries no uuid -- the binding is read to
    find the measure, not to serve it -- so categories are the only selector, and a shared
    one would hand both measures the same series.
    """
    df = _frame(
        [
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, 'energy_carrier': 'electricity', VALUE_COLUMN: 1.0},
            {YEAR_COLUMN: 2020, 'uuid': UUID_B, 'energy_carrier': 'electricity', VALUE_COLUMN: 3.0},
        ],
        dims=['energy_carrier'],
    )
    assert _uuids_from_binding(_binding(df)) == []


def test_only_the_colliding_uuids_are_dropped():
    """A collision is not a reason to discard the measures around it."""
    df = _frame(
        [
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, 'energy_carrier': 'electricity', VALUE_COLUMN: 1.0},
            {YEAR_COLUMN: 2020, 'uuid': UUID_B, 'energy_carrier': 'electricity', VALUE_COLUMN: 3.0},
            {YEAR_COLUMN: 2020, 'uuid': UUID_C, 'energy_carrier': 'heat', VALUE_COLUMN: 5.0},
        ],
        dims=['energy_carrier'],
    )
    assert dict(_uuids_from_binding(_binding(df))) == {UUID_C: {'energy_carrier': 'heat'}}


def test_null_categories_are_dropped_from_the_selector():
    """
    A null category means the column does not apply, not that it should be matched.

    ``filter(energy_carrier=None)`` against the node's output matches nothing, so leaving
    it in the selector would blank the cell.
    """
    df = _frame(
        [
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, 'transport_mode': 'cars', 'energy_carrier': None, VALUE_COLUMN: 1.0},
            {YEAR_COLUMN: 2020, 'uuid': UUID_B, 'transport_mode': 'buses', 'energy_carrier': None, VALUE_COLUMN: 2.0},
        ],
        dims=['transport_mode', 'energy_carrier'],
    )
    assert dict(_uuids_from_binding(_binding(df))) == {
        UUID_A: {'transport_mode': 'cars'},
        UUID_B: {'transport_mode': 'buses'},
    }


def test_uuid_spanning_several_categories_is_skipped():
    """
    A cell holds one number, so a measure spread over categories has no single answer.

    Mapping it with no categories to narrow by would emit one point per category per
    year, and the client would keep whichever arrived last. Its neighbours in the same
    binding are unaffected.
    """
    df = _frame(
        [
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, 'energy_carrier': 'electricity', VALUE_COLUMN: 1.0},
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, 'energy_carrier': 'heat', VALUE_COLUMN: 2.0},
            {YEAR_COLUMN: 2020, 'uuid': UUID_B, 'energy_carrier': 'heat', VALUE_COLUMN: 3.0},
        ],
        dims=['energy_carrier'],
    )
    assert dict(_uuids_from_binding(_binding(df))) == {UUID_B: {'energy_carrier': 'heat'}}


def test_a_selector_contained_in_another_is_dropped_but_the_narrower_one_stays():
    """
    A broader selector sweeps up the narrower one's rows, so it pins nothing.

    UUID_A's null energy_carrier leaves it selecting on transport_mode alone, which also
    matches UUID_B's rows. UUID_B keeps its extra category, and nothing outside it
    satisfies that, so it is still unambiguous.
    """
    df = _frame(
        [
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, 'transport_mode': 'cars', 'energy_carrier': None, VALUE_COLUMN: 1.0},
            {
                YEAR_COLUMN: 2020,
                'uuid': UUID_B,
                'transport_mode': 'cars',
                'energy_carrier': 'electricity',
                VALUE_COLUMN: 2.0,
            },
        ],
        dims=['transport_mode', 'energy_carrier'],
    )
    assert dict(_uuids_from_binding(_binding(df))) == {
        UUID_B: {'transport_mode': 'cars', 'energy_carrier': 'electricity'},
    }


def test_an_empty_selector_beside_any_other_is_dropped():
    """The empty selector filters nothing, so it matches every other measure's rows too."""
    df = _frame(
        [
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, 'energy_carrier': None, VALUE_COLUMN: 1.0},
            {YEAR_COLUMN: 2020, 'uuid': UUID_B, 'energy_carrier': 'heat', VALUE_COLUMN: 2.0},
        ],
        dims=['energy_carrier'],
    )
    assert dict(_uuids_from_binding(_binding(df))) == {UUID_B: {'energy_carrier': 'heat'}}


def test_two_uuids_in_a_binding_with_no_dimensions_are_both_dropped():
    """
    An empty selector is as ambiguous as a shared one, and collides the same way.

    Both measures would resolve to the same node, metric and (absent) categories, so the
    resolver would hand them the identical output series.
    """
    df = _frame(
        [
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, VALUE_COLUMN: 1.0},
            {YEAR_COLUMN: 2020, 'uuid': UUID_B, VALUE_COLUMN: 2.0},
        ],
        dims=[],
    )
    assert _uuids_from_binding(_binding(df)) == []


def test_uuid_spanning_several_years_is_claimed_once():
    df = _frame(
        [
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, VALUE_COLUMN: 1.0},
            {YEAR_COLUMN: 2021, 'uuid': UUID_A, VALUE_COLUMN: 2.0},
            {YEAR_COLUMN: 2022, 'uuid': UUID_A, VALUE_COLUMN: 3.0},
        ],
        dims=[],
    )
    assert _uuids_from_binding(_binding(df)) == [(UUID_A, None)]


def _instance_with_bindings(node_id: str, tag_sets: list[list[str]]) -> Any:
    node = SimpleNamespace(input_dataset_instances=[_binding(None, tags) for tags in tag_sets])
    return SimpleNamespace(context=SimpleNamespace(nodes={node_id: node}))


def test_historical_binding_wins_over_goal_on_the_same_node():
    """One dataset, one uuid column, a historical and a goal value column: take historical."""
    instance = _instance_with_bindings('renovation', [['city_data', 'historical'], ['city_data', 'goal']])
    values = [
        NodeDimensionSelection(node_id='renovation', dimensions=None, dataset_index=0, binding_role='historical'),
        NodeDimensionSelection(node_id='renovation', dimensions=None, dataset_index=1, binding_role='goal'),
    ]

    assert FrameworkConfig._prefer_historical_bindings(instance, values) == [values[0]]


def test_untagged_binding_wins_over_goal():
    instance = _instance_with_bindings('renovation', [['city_data'], ['city_data', 'goal']])
    values = [
        NodeDimensionSelection(node_id='renovation', dimensions=None, dataset_index=0),
        NodeDimensionSelection(node_id='renovation', dimensions=None, dataset_index=1, binding_role='goal'),
    ]

    assert FrameworkConfig._prefer_historical_bindings(instance, values) == [values[0]]


def _instance_with_nodes_bindings(spec: dict[str, list[list[str]]], deltas: set[str] | None = None) -> Any:
    """Return an instance whose nodes carry the given binding tag-sets, optionally delta-valued."""
    deltas = deltas or set()
    nodes = {
        node_id: SimpleNamespace(
            input_dataset_instances=[_binding(None, tags) for tags in tag_sets],
            output_is_baseline_delta=node_id in deltas,
        )
        for node_id, tag_sets in spec.items()
    }
    return SimpleNamespace(context=SimpleNamespace(nodes=nodes))


def test_tag_ranking_does_not_reach_across_nodes():
    """
    The tag tie-break settles which column of one dataset to read, nothing more.

    An action binds a column as ``historical`` while the level node downstream binds the
    same column untagged. Ranking them together lets the action win and the ``*_observed``
    node never reaches the preference that exists for it -- and the action reports a
    baseline delta, so the cell ends up blank.
    """
    instance = _instance_with_nodes_bindings({
        'a31_renovation_improvements': [['city_data', 'historical']],
        'old_building_renovation_rate_observed': [['city_data']],
    })
    action = NodeDimensionSelection(
        node_id='a31_renovation_improvements', dimensions=None, dataset_index=0, binding_role='historical'
    )
    observed = NodeDimensionSelection(node_id='old_building_renovation_rate_observed', dimensions=None, dataset_index=0)

    assert FrameworkConfig._prefer_historical_bindings(instance, [action, observed]) == [action, observed]


def test_a_level_node_beats_a_delta_node():
    """
    A cell asks what the plan is, and a delta node answers how far it moves.

    The node-id heuristics cannot separate these: neither ``new_building_shares`` nor
    ``a32_new_building_improvements`` carries a suffix they recognise.
    """
    instance = _instance_with_nodes_bindings(
        {'new_building_shares': [['city_data']], 'a32_new_building_improvements': [['city_data', 'historical']]},
        deltas={'a32_new_building_improvements'},
    )
    level = NodeDimensionSelection(node_id='new_building_shares', dimensions=None, dataset_index=0)
    delta = NodeDimensionSelection(
        node_id='a32_new_building_improvements', dimensions=None, dataset_index=0, binding_role='historical'
    )

    assert FrameworkConfig._prefer_a_level_over_a_delta(instance, [level, delta]) == [level]


def test_all_delta_candidates_are_left_for_the_resolver_to_withhold():
    """Nothing to prefer, so the choice stands and the resolver declines to show a delta."""
    instance = _instance_with_nodes_bindings({'a1': [['city_data']], 'a2': [['city_data']]}, deltas={'a1', 'a2'})
    first = NodeDimensionSelection(node_id='a1', dimensions=None, dataset_index=0)
    second = NodeDimensionSelection(node_id='a2', dimensions=None, dataset_index=0)

    assert FrameworkConfig._prefer_a_level_over_a_delta(instance, [first, second]) == [first, second]


def test_legacy_selection_survives_alongside_a_goal_binding():
    """A DatasetNode carries no binding to rank, so it must reach the node-id heuristics."""
    instance = _instance_with_bindings('renovation', [['city_data', 'goal']])
    legacy = NodeDimensionSelection(node_id='population_observed', dimensions=None)
    binding = NodeDimensionSelection(node_id='renovation', dimensions=None, dataset_index=0)

    assert FrameworkConfig._prefer_historical_bindings(instance, [legacy, binding]) == [legacy, binding]


def test_legacy_selection_survives_alongside_a_historical_binding():
    """
    A historical binding must not evict a legacy candidate.

    Ranking a binding-less selection against a binding would silently resolve mixed
    models -- a uuid still present on a DatasetNode *and* on a new city_data binding --
    without the node-id heuristics getting a say.
    """
    instance = _instance_with_bindings('renovation', [['city_data', 'historical']])
    legacy = NodeDimensionSelection(node_id='population_observed', dimensions=None)
    binding = NodeDimensionSelection(node_id='renovation', dimensions=None, dataset_index=0, binding_role='goal')

    assert FrameworkConfig._prefer_historical_bindings(instance, [legacy, binding]) == [legacy, binding]


def test_legacy_selection_survives_a_tie_between_bindings():
    """The tag tiebreak narrows the bindings but must still leave the legacy candidate."""
    instance = _instance_with_bindings('renovation', [['city_data', 'historical'], ['city_data', 'goal']])
    legacy = NodeDimensionSelection(node_id='population_observed', dimensions=None)
    historical = NodeDimensionSelection(node_id='renovation', dimensions=None, dataset_index=0, binding_role='historical')
    goal = NodeDimensionSelection(node_id='renovation', dimensions=None, dataset_index=1, binding_role='goal')

    survivors = FrameworkConfig._prefer_historical_bindings(instance, [legacy, historical, goal])

    assert survivors == [legacy, historical]


def _real_binding(df: PathsDataFrame, tags: list[str]) -> FrameworkMeasureDVCDataset2:
    """
    Build a binding the node walk will accept, with its frame already in place.

    The walk tests the concrete dataset type, so a stand-in object is skipped. Seeding
    the memo means ``get_uuid_frame`` answers without reaching for DVC or the database.
    """
    ds = object.__new__(FrameworkMeasureDVCDataset2)
    ds.id = 'nzc/buildings_stock_renovation'
    ds.tags = tags
    ds._uuid_frame = df
    ds._uuid_frame_loaded = True
    return ds


def _node_with_metrics(bindings: list[Any], metrics: dict[str, str]) -> Any:
    """Return a non-DatasetNode whose output_metrics map metric id to output column."""
    return SimpleNamespace(
        id='a31_renovation_improvements',
        input_dataset_instances=bindings,
        output_metrics={mid: SimpleNamespace(column_id=col) for mid, col in metrics.items()},
    )


def test_binding_tag_names_the_output_metric():
    """
    A multi-metric node renames Value to the metric column, so the binding must say which.

    The config tags each binding with the metric id it feeds, alongside historical/goal.
    Without this the resolver finds no Value column and blanks the cell.
    """
    df = _frame([{YEAR_COLUMN: 2020, 'uuid': UUID_A, VALUE_COLUMN: 1.0}], dims=[])
    node = _node_with_metrics(
        [_real_binding(df, ['historical', 'renovation_rate', 'city_data'])],
        {'renovation_rate': 'renovation_rate', 'renovation_shares': 'renovation_shares'},
    )
    ((_uuid, sel),) = FrameworkConfig()._get_node_dimension_selections('a31_renovation_improvements', node)

    assert _uuid == UUID_A
    assert sel.metric_col == 'renovation_rate'


def test_single_metric_node_leaves_metric_col_unset():
    """Nothing was renamed, so Value is the series and there is no column to record."""
    df = _frame([{YEAR_COLUMN: 2020, 'uuid': UUID_A, VALUE_COLUMN: 1.0}], dims=[])
    node = _node_with_metrics([_real_binding(df, ['historical', 'city_data'])], {})
    ((_uuid, sel),) = FrameworkConfig()._get_node_dimension_selections('population', node)

    assert sel.metric_col is None


def _points(df: PathsDataFrame, selection: NodeDimensionSelection) -> list[tuple[Any, Any]]:
    pts = MeasureType._narrow_to_placeholders(df, selection, baseline_year=2019, last_year=2026, label='test measure')
    return [(p.year, p.value) for p in pts]


_BINDING_SEL = NodeDimensionSelection(node_id='n', dimensions=None, dataset_index=0)


def test_years_outside_the_sheet_window_are_dropped():
    df = _frame(
        [{YEAR_COLUMN: y, 'uuid': UUID_A, VALUE_COLUMN: float(y)} for y in (2018, 2019, 2020, 2026, 2027)],
        dims=[],
    )
    assert _points(df, _BINDING_SEL) == [(2020, 2020.0), (2026, 2026.0)]


def test_null_values_from_a_sibling_metric_are_dropped():
    """
    A multi-metric node keeps rows where the other metrics have nothing to say.

    After selecting one metric those read as null. They are not this measure's series,
    and counting them would trip the one-row-per-year guard and blank a cell that has a
    perfectly good value.
    """
    df = _frame(
        [
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, VALUE_COLUMN: 1.0},
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, VALUE_COLUMN: None},
            {YEAR_COLUMN: 2021, 'uuid': UUID_A, VALUE_COLUMN: 2.0},
        ],
        dims=[],
    )
    assert _points(df, _BINDING_SEL) == [(2020, 1.0), (2021, 2.0)]


def test_genuinely_duplicated_years_yield_nothing():
    """Two real values for one year means the selection failed; do not pick one."""
    df = _frame(
        [
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, VALUE_COLUMN: 1.0},
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, VALUE_COLUMN: 9.0},
        ],
        dims=[],
    )
    assert _points(df, _BINDING_SEL) == []


def test_categories_narrow_to_one_series():
    df = _frame(
        [
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, 'energy_carrier': 'electricity', VALUE_COLUMN: 1.0},
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, 'energy_carrier': 'heat', VALUE_COLUMN: 9.0},
        ],
        dims=['energy_carrier'],
    )
    sel = NodeDimensionSelection(node_id='n', dimensions={'energy_carrier': 'electricity'}, dataset_index=0)
    assert _points(df, sel) == [(2020, 1.0)]


_LEGACY_SEL = NodeDimensionSelection(node_id='n', dimensions=None, dataset_index=None)


def test_legacy_path_keeps_emitting_duplicate_years():
    """
    The one-row-per-year guard is deliberately binding-only.

    A DatasetNode wraps a single dataset, so its output was always taken as the measure's
    series whatever shape it had. No NZC config reaches this branch any more -- the model
    swap left none of those nodes -- so nothing but this test holds the behaviour still.
    """
    df = _frame(
        [
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, VALUE_COLUMN: 1.0},
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, VALUE_COLUMN: 9.0},
        ],
        dims=[],
    )
    assert _points(df, _LEGACY_SEL) == [(2020, 1.0), (2020, 9.0)]


def test_legacy_path_drops_null_values():
    """
    Null filtering is shared, and this pins that it is safe to share.

    A null placeholder is nothing the client can draw, and leaving one in could displace
    a real value for that year, so dropping it is an improvement on the legacy path too --
    but it is a behaviour change to code CADS would run, hence the test.
    """
    df = _frame(
        [
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, VALUE_COLUMN: None},
            {YEAR_COLUMN: 2021, 'uuid': UUID_A, VALUE_COLUMN: 2.0},
        ],
        dims=[],
    )
    assert _points(df, _LEGACY_SEL) == [(2021, 2.0)]


def test_legacy_path_still_honours_categories_and_the_window():
    df = _frame(
        [
            {YEAR_COLUMN: 2019, 'uuid': UUID_A, 'energy_carrier': 'heat', VALUE_COLUMN: 1.0},
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, 'energy_carrier': 'heat', VALUE_COLUMN: 2.0},
            {YEAR_COLUMN: 2020, 'uuid': UUID_A, 'energy_carrier': 'electricity', VALUE_COLUMN: 3.0},
        ],
        dims=['energy_carrier'],
    )
    sel = NodeDimensionSelection(node_id='n', dimensions={'energy_carrier': 'heat'}, dataset_index=None)
    assert _points(df, sel) == [(2020, 2.0)]


def _instance_with_nodes(*nodes: Any) -> Any:
    return SimpleNamespace(context=SimpleNamespace(nodes={n.id: n for n in nodes}))


def _node(node_id: str, unit: str | None, dims: list[str], metrics: dict[str, str] | None = None) -> Any:
    return SimpleNamespace(
        id=node_id,
        unit=unit_registry.parse_units(unit) if unit else None,
        output_nodes=[],
        output_dimensions=dict.fromkeys(dims),
        output_metrics={mid: SimpleNamespace(column_id=col) for mid, col in (metrics or {}).items()},
    )


def _action(action_id: str, targets: list[Any]) -> Any:
    """
    Return a real ActionNode, so the isinstance check in the traversal holds.

    ``output_nodes`` is a read-only property derived from the graph's edges, so a subclass
    supplies the targets rather than a plain instance being mutated.
    """
    from nodes.actions.simple import GenericAction

    class _StubAction(GenericAction):
        @property
        def output_nodes(self) -> list[Any]:  # type: ignore[override]
            return list(targets)

    action = object.__new__(_StubAction)
    action.id = action_id
    return action


def _goal_chain(goal_unit: str = '%', target_unit: str = '%', dims: list[str] | None = None) -> tuple[Any, Any]:
    """Return ``goal -> action -> trajectory``, the shape NZC uses."""
    dims = dims if dims is not None else ['transport_mode']
    goal = _node('utilisation_goal', goal_unit, dims)
    trajectory = _node('utilisation', target_unit, dims)
    goal.output_nodes = [_action('a21_optimised_logistics', [trajectory])]
    return goal, trajectory


def test_a_goal_node_defers_to_the_trajectory_the_action_emits():
    """
    Follow the graph, not the name.

    The goal holds only the target end; the action combines it with history and emits the
    whole series, so the relationship is the graph edge.
    """
    goal, trajectory = _goal_chain()
    instance = _instance_with_nodes(goal, trajectory)
    sel = NodeDimensionSelection(node_id='utilisation_goal', dimensions={'transport_mode': 'light_trucks'}, dataset_index=0)

    assert FrameworkConfig._prefer_the_full_trajectory(instance, sel).node_id == 'utilisation'


def test_a_node_feeding_no_action_is_left_alone():
    """Nothing better to point at, so the measure keeps its node and shows nothing."""
    lone = _node('fossil_electricity_goal', '%', [])
    instance = _instance_with_nodes(lone)
    sel = NodeDimensionSelection(node_id='fossil_electricity_goal', dimensions=None, dataset_index=0)

    assert FrameworkConfig._prefer_the_full_trajectory(instance, sel).node_id == 'fossil_electricity_goal'


def test_the_action_output_measuring_the_same_quantity_is_chosen():
    """
    Pick the output measuring the goal's own quantity.

    An action feeds several nodes: ``a21_optimised_logistics`` emits a utilisation
    percentage and vehicle kilometres, and only the former is this measure's quantity.
    """
    goal = _node('utilisation_goal', '%', ['transport_mode'])
    trajectory = _node('utilisation', '%', ['transport_mode'])
    other = _node('freight_transport_vehicle_kilometres', 'Mvehicle_km/a', ['transport_mode'])
    goal.output_nodes = [_action('a21_optimised_logistics', [trajectory, other])]
    instance = _instance_with_nodes(goal, trajectory, other)
    sel = NodeDimensionSelection(node_id='utilisation_goal', dimensions=None, dataset_index=0)

    assert FrameworkConfig._prefer_the_full_trajectory(instance, sel).node_id == 'utilisation'


def test_an_ambiguous_action_output_is_left_alone():
    """Two candidates of the same quantity, so the graph does not answer; do not guess."""
    goal = _node('utilisation_goal', '%', ['transport_mode'])
    trajectory = _node('utilisation', '%', ['transport_mode'])
    twin = _node('utilisation_twin', '%', ['transport_mode'])
    goal.output_nodes = [_action('a21_optimised_logistics', [trajectory, twin])]
    instance = _instance_with_nodes(goal, trajectory, twin)
    sel = NodeDimensionSelection(node_id='utilisation_goal', dimensions=None, dataset_index=0)

    assert FrameworkConfig._prefer_the_full_trajectory(instance, sel).node_id == 'utilisation_goal'


def test_a_target_that_cannot_serve_the_selection_is_not_used():
    """The redirect must not hand the resolver a node it cannot filter the same way."""
    goal, trajectory = _goal_chain(dims=['transport_mode'])
    trajectory.output_dimensions = dict.fromkeys(['other_dim'])
    instance = _instance_with_nodes(goal, trajectory)
    sel = NodeDimensionSelection(node_id='utilisation_goal', dimensions={'transport_mode': 'light_trucks'}, dataset_index=0)

    assert FrameworkConfig._prefer_the_full_trajectory(instance, sel).node_id == 'utilisation_goal'


def test_a_target_without_the_chosen_metric_is_not_used():
    goal, trajectory = _goal_chain(dims=[])
    trajectory.output_metrics = {'other': SimpleNamespace(column_id='other')}
    instance = _instance_with_nodes(goal, trajectory)
    sel = NodeDimensionSelection(node_id='utilisation_goal', dimensions=None, dataset_index=0, metric_col='renovation_rate')

    assert FrameworkConfig._prefer_the_full_trajectory(instance, sel).node_id == 'utilisation_goal'


def test_an_ordinary_node_is_untouched():
    plain = _node('population', 'cap', [])
    instance = _instance_with_nodes(plain)
    sel = NodeDimensionSelection(node_id='population', dimensions=None, dataset_index=0)

    assert FrameworkConfig._prefer_the_full_trajectory(instance, sel) is sel


def test_delta_producing_classes_declare_it():
    """
    The resolver withholds these values, and relies on the class to say so.

    Both end on ``col - pl.first(col)``. Checking the declaration rather than the class
    keeps the arithmetic and the claim about it together: a producer that starts or stops
    subtracting the baseline updates one line, and no consumer has to be edited to match.
    """
    from nodes.actions.gpc import DatasetDifferenceAction2, SCurveAction as GPCSCurveAction
    from nodes.actions.linear import DatasetDifferenceAction, DatasetReduceAction, DatasetReduceAction2
    from nodes.actions.simple import SCurveAction

    # Each subtracts a baseline or last-historical anchor before returning.
    assert DatasetReduceAction.output_is_baseline_delta
    assert DatasetReduceAction2.output_is_baseline_delta
    assert DatasetDifferenceAction.output_is_baseline_delta
    assert DatasetDifferenceAction2.output_is_baseline_delta
    assert SCurveAction.output_is_baseline_delta
    assert GPCSCurveAction.output_is_baseline_delta


def test_levels_are_the_default():
    """
    The claim is about the arithmetic, not about being an action.

    GenericAction returns its own quantity, and the one NZC measure sitting on one is a
    reduction percentage that reads correctly as it stands.
    """
    from nodes.actions.simple import GenericAction
    from nodes.generic import GenericNode
    from nodes.node import Node

    assert not Node.output_is_baseline_delta
    assert not GenericNode.output_is_baseline_delta
    assert not GenericAction.output_is_baseline_delta


def _exploding_binding(tags: list[str]) -> FrameworkMeasureDVCDataset2:
    """Return a binding whose source cannot be read, as a missing dataset would be."""
    from nodes.exceptions import NodeError

    ds = object.__new__(FrameworkMeasureDVCDataset2)
    ds.id = 'nzc/vanished'
    ds.tags = tags

    def boom() -> Any:
        raise NodeError(SimpleNamespace(id='n'), 'dataset is gone')  # type: ignore[arg-type]

    ds.get_uuid_frame = boom  # type: ignore[method-assign]
    return ds


def test_one_unreadable_binding_does_not_blank_the_others():
    """
    The mapping is built once for the whole config and every measure waits on it.

    An exception escaping here fails correspondingNode and placeholderDataPoints for every
    measure on the tab -- and being a cached_property it is not cached either, so each
    measure retries the same broken load.
    """
    good = _frame([{YEAR_COLUMN: 2020, 'uuid': UUID_A, VALUE_COLUMN: 1.0}], dims=[])
    node = _node_with_metrics(
        [_exploding_binding(['city_data']), _real_binding(good, ['city_data'])],
        {},
    )
    selections = FrameworkConfig()._get_node_dimension_selections('population', node)

    assert [u for u, _ in selections] == [UUID_A]


def test_an_unreadable_binding_claims_nothing():
    node = _node_with_metrics([_exploding_binding(['city_data'])], {})

    assert FrameworkConfig()._get_node_dimension_selections('population', node) == []


def _dataset_with_source(df: PathsDataFrame) -> FrameworkMeasureDVCDataset2:
    """
    Build a binding whose frame has to travel the real pipeline to be read.

    Unlike ``_real_binding`` this seeds no memo, so ``get_uuid_frame`` runs its body:
    the source load, the binding's ``column`` selection and its pre-temporal
    transformations. That is the part the mapping's correctness rests on, and the part
    no other test in this file exercises.
    """
    # perf_context is only needed so that the instrumented `_filter_and_process_df` can
    # run at all: the second test below calls it deliberately, and without this the call
    # dies on the stub rather than on the column it drops.
    context = cast(
        'Context',
        SimpleNamespace(
            dimensions={},
            framework_config_data=None,
            get_parameter_value=lambda *_args, **_kwargs: False,
            instance=SimpleNamespace(reference_year=2020, maximum_historical_year=2020),
            unit_registry=unit_registry,
            perf_context=SimpleNamespace(exec_named=lambda **_kwargs: nullcontext(None)),
        ),
    )
    ds = FrameworkMeasureDVCDataset2(id='nzc/buildings_stock_renovation', context=context)
    ds.payload_ref = cast('Any', object())
    ds.payload_store = cast('Any', SimpleNamespace(get_dataframe=lambda _ref: df))
    return ds


def test_get_uuid_frame_reaches_the_caller_with_uuid_intact():
    """
    The measure linkage has to survive the read, or every measure resolves to nothing.

    This is the whole premise of the binding path: ``uuid`` is the only column saying
    which measure a row belongs to, and the node's output carries none.
    """
    df = _frame([{YEAR_COLUMN: 2020, 'uuid': UUID_A, VALUE_COLUMN: 1.0}], dims=[])

    got = _dataset_with_source(df).get_uuid_frame()

    assert got is not None
    assert 'uuid' in got.columns
    assert got['uuid'].to_list() == [UUID_A]


def test_get_uuid_frame_must_not_go_through_filter_and_process_df():
    """
    ``_filter_and_process_df`` calls ``before_temporal_fill``, which drops ``uuid``.

    The overlay there folds in the city's MeasureDataPoints and then rebuilds the frame
    without ``uuid`` (see ``_override_with_measure_datapoints``), so a ``get_uuid_frame``
    routed through it returns ``None`` for every binding and the placeholder mapping
    comes out empty -- silently, because a blank cell is what a missing placeholder
    looks like. Pin both halves: the overlay drops the column, and this method does not
    reach the overlay.
    """
    df = _frame([{YEAR_COLUMN: 2020, 'uuid': UUID_A, VALUE_COLUMN: 1.0}], dims=[])
    ds = _dataset_with_source(df)

    assert 'uuid' not in ds.before_temporal_fill(df).columns
    assert 'uuid' in cast('PathsDataFrame', ds.get_uuid_frame()).columns


def test_a_goal_only_measure_gets_no_placeholders():
    """
    A measure carried only by a goal-tagged binding is a target-year input.

    "Share of heating as district heating, the target year" is not a question about the
    years up to today, and the action it sits on emits movement from the baseline rather
    than either the target or a level. On budapest-cap-2030 this is 21 of 159 mapped
    measures, every one a 2030 assumption.
    """
    node = SimpleNamespace(id='a34_decarbonising_heat_generation')
    measure = SimpleNamespace(measure_template=SimpleNamespace(uuid=UUID_A, unit='%'))
    sel = NodeDimensionSelection(node_id=node.id, dimensions=None, dataset_index=0, binding_role='goal')

    assert MeasureType._get_placeholder_df(cast('Any', measure), cast('Any', node), sel) is None


def test_a_historical_measure_is_not_withheld():
    """The counterpart: an untagged or historical binding is exactly what the tab asks for."""
    df = _frame([{YEAR_COLUMN: 2020, 'uuid': UUID_A, VALUE_COLUMN: 1.0}], dims=[])
    node = SimpleNamespace(id='heating_fuel_share_district', get_output_pl=lambda: df)
    measure = SimpleNamespace(measure_template=SimpleNamespace(uuid=UUID_A, unit='kWh'))
    sel = NodeDimensionSelection(node_id=node.id, dimensions=None, dataset_index=0)

    got = MeasureType._get_placeholder_df(cast('Any', measure), cast('Any', node), sel)

    assert got is not None
    assert got[VALUE_COLUMN].to_list() == [1.0]
