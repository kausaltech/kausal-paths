"""
A parameter panel lists parameters the editor chose, for the visitor to change.

What these pin down: the admin offers only parameters a visitor may change, global and
node-level alike; the page returns them in the editor's order from the instance it is served
for; and a parameter the served instance lacks, or has since made fixed, is left out instead of
failing the page.
"""

import json
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from django.http import Http404
from django.test import RequestFactory
from wagtail.models import Locale, Page

import pytest

from paths.context import RealmContext, realm_context

from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.defs.node_defs import NodeSpec, SimpleConfig
from nodes.defs.port_def import OutputPortDef
from nodes.dimensions import Dimension
from nodes.models import NodeConfig, test_instance_registry
from nodes.scenario import Scenario, ScenarioKind
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory
from nodes.units import unit_registry
from pages.admin_views import node_graph_options
from pages.blocks import (
    ParameterPanelBlock,
    ParameterPanelGraphBlock,
    admin_instance_category_choices,
    admin_instance_dimension_choices,
    admin_instance_parameter_choices,
)
from pages.models import InstanceRootPage, OutcomePage
from params.param import BoolParameter, NumberParameter

if TYPE_CHECKING:
    from django.test import Client

    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db

PAGES_QUERY = """
query Pages {
    pages {
        ... on OutcomePage {
            body {
                ... on ParameterPanelBlock {
                    title
                    showReset
                    nodeGraphs {
                        nodeId
                        groupBy
                        filters {
                            dimension
                            categories
                        }
                    }
                    parameters {
                        id
                        label
                        isCustomizable
                    }
                }
            }
        }
    }
}
"""


@pytest.fixture
def panel_config() -> InstanceConfig:
    """Return a database-backed instance with customizable and fixed parameters, global and on a node."""
    instance = InstanceFactory.create()
    spec = InstanceModelSpec(
        years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030),
        params=[
            NumberParameter(local_id='warming_limit', label='Warming limit', value=1.5, is_customizable=True),
            BoolParameter(local_id='locked', label='Locked', value=False, is_customizable=False),
        ],
        scenarios=[Scenario(id='default', name='Default', kind=ScenarioKind.DEFAULT)],
        dimensions=[
            {
                'id': 'meal',
                'label': 'Meal',
                'categories': [{'id': 'meat', 'label': 'Meat'}, {'id': 'vegetables', 'label': 'Vegetables'}],
            },
            {'id': 'site', 'label': 'Site', 'categories': [{'id': 'school', 'label': 'School'}]},
        ],
    )
    ic = InstanceConfigFactory.create(
        identifier=instance.id,
        instance=instance,
        config_source='database',
        owner='Test Owner',
        spec=spec,
    )
    for dim_dict in spec.dimensions:
        ic.sync_dimension(Dimension.model_validate(dim_dict))
    NodeConfigFactory.create(
        instance=ic,
        identifier='population',
        name='Population',
        spec=NodeSpec(
            type_config=SimpleConfig(node_class='nodes.simple.FixedMultiplierNode'),
            output_ports=[OutputPortDef(id=uuid4(), unit=unit_registry.parse_units('cap'), quantity='population', dimensions=[])],
            params=[NumberParameter(local_id='multiplier', label='Multiplier', value=1.0, is_customizable=True)],
        ),
    )
    NodeConfigFactory.create(
        instance=ic,
        identifier='meal_emissions',
        name='Meal emissions',
        spec=NodeSpec(
            type_config=SimpleConfig(node_class='nodes.simple.AdditiveNode'),
            output_ports=[
                OutputPortDef(id=uuid4(), unit=unit_registry.parse_units('t/a'), quantity='emissions', dimensions=['meal'])
            ],
        ),
    )
    # Build the runtime from the database rather than the shared test instance.
    test_instance_registry.pop(ic.identifier, None)
    return ic


def _outcome_page(ic: InstanceConfig, panel: dict[str, Any]) -> OutcomePage:
    locale, _ = Locale.objects.get_or_create(language_code='en')
    wagtail_root = Page.get_first_root_node()
    assert wagtail_root is not None
    root = cast(
        'InstanceRootPage',
        wagtail_root.add_child(
            instance=InstanceRootPage(locale=locale, title='Root', slug=ic.identifier, url_path='', body='[]'),
        ),
    )
    ic.root_page = root
    ic.save(update_fields=['root_page'])
    body = json.dumps([{'type': 'parameter_panel', 'value': panel}])
    node = NodeConfig.objects.get(instance=ic, identifier='population')
    page = OutcomePage(locale=locale, title='Issue', slug='issue', outcome_node=node, body=body)
    return cast('OutcomePage', root.add_child(instance=page))


def test_choices_are_the_customizable_parameters_globals_first(panel_config: InstanceConfig) -> None:
    with realm_context.activate(RealmContext(realm=panel_config, user=None)):
        choices = admin_instance_parameter_choices()

    assert choices == [('warming_limit', 'Warming limit'), ('population.multiplier', 'Population: Multiplier')]


def test_choices_are_empty_outside_an_admin_request() -> None:
    assert admin_instance_parameter_choices() == []


def test_the_page_returns_the_chosen_parameters_in_order(client: Client, panel_config: InstanceConfig) -> None:
    from paths.tests.graphql import PathsTestClient

    meals = NodeConfig.objects.get(instance=panel_config, identifier='meal_emissions')
    _outcome_page(
        panel_config,
        {
            'title': 'Your premises',
            'parameter_ids': ['population.multiplier', 'warming_limit', 'population.multiplier'],
            'show_reset': True,
            'graphs': [
                {'node': meals.pk, 'group_by': 'meal', 'filters': ['meal:vegetables', 'meal:meat']},
                {'node': meals.pk, 'group_by': None, 'filters': []},
            ],
        },
    )
    gql_client = PathsTestClient(client)
    gql_client.set_instance(panel_config)
    data = gql_client.query_data(PAGES_QUERY)

    panels = [block for page in data['pages'] for block in page.get('body') or [] if block]
    assert len(panels) == 1
    panel = panels[0]
    assert panel['title'] == 'Your premises'
    assert panel['showReset'] is True
    # Duplicates are dropped; the editor's order is kept.
    assert [(p['id'], p['label']) for p in panel['parameters']] == [
        ('population.multiplier', 'Multiplier'),
        ('warming_limit', 'Warming limit'),
    ]
    assert all(p['isCustomizable'] for p in panel['parameters'])
    # The same node may be graphed twice, with different defaults.
    assert panel['nodeGraphs'] == [
        {
            'nodeId': 'meal_emissions',
            'groupBy': 'meal',
            'filters': [{'dimension': 'meal', 'categories': ['vegetables', 'meat']}],
        },
        {'nodeId': 'meal_emissions', 'groupBy': None, 'filters': []},
    ]


def test_parameters_the_served_instance_lacks_or_fixes_are_left_out(panel_config: InstanceConfig) -> None:
    # As when the published revision differs from the draft the editor chose from.
    info = _info(panel_config)

    params = ParameterPanelBlock().parameters(
        info, {'parameter_ids': ['removed', 'locked', 'nonexistent_node.multiplier', 'warming_limit']}
    )

    assert [param.global_id for param in params] == ['warming_limit']


def _info(panel_config: InstanceConfig) -> Any:
    context = panel_config.get_instance().context
    return type('Info', (), {'context': type('Ctx', (), {'require_instance': lambda _: context.instance})()})()


def test_graph_defaults_the_served_node_lacks_are_left_out(panel_config: InstanceConfig) -> None:
    # As when the published revision differs from the draft the editor chose from: a node row the
    # runtime does not have, and dimensions or categories the node does not have.
    info = _info(panel_config)
    meals = NodeConfig.objects.get(instance=panel_config, identifier='meal_emissions')
    newer = NodeConfigFactory.create(instance=panel_config, identifier='newer', name='Newer')

    graphs = ParameterPanelBlock().node_graphs(
        info,
        {
            'graphs': [
                {'node': newer, 'group_by': None, 'filters': []},
                {'node': None, 'group_by': None, 'filters': []},
                {
                    'node': meals,
                    'group_by': 'site',
                    # A dict is an item of the list the field used to be; it is ignored.
                    'filters': ['site:school', 'meal:fish', 'meal:meat', 'meal:meat', 'garbled', {'value': 'meal:meat'}],
                },
            ]
        },
    )

    assert len(graphs) == 1
    assert graphs[0].node_id == 'meal_emissions'
    assert graphs[0].group_by is None
    assert [(f.dimension, f.categories) for f in graphs[0].filters] == [('meal', ['meat'])]


def test_a_panel_saved_without_graphs_has_none(panel_config: InstanceConfig) -> None:
    assert ParameterPanelBlock().node_graphs(_info(panel_config), {'parameter_ids': ['warming_limit']}) == []


def test_the_admin_offers_dimensions_and_their_categories(panel_config: InstanceConfig) -> None:
    with realm_context.activate(RealmContext(realm=panel_config, user=None)):
        assert admin_instance_dimension_choices() == [('meal', 'Meal (meal)'), ('site', 'Site (site)')]
        assert admin_instance_category_choices() == [
            ('Meal', [('meal:meat', 'Meat'), ('meal:vegetables', 'Vegetables')]),
            ('Site', [('site:school', 'School')]),
        ]


def test_the_admin_form_refuses_dimensions_the_node_lacks(panel_config: InstanceConfig) -> None:
    from wagtail.blocks import StructBlockValidationError

    meals = NodeConfig.objects.get(instance=panel_config, identifier='meal_emissions')
    block = ParameterPanelGraphBlock()
    with realm_context.activate(RealmContext(realm=panel_config, user=None)):
        block.clean(block.to_python({'node': meals.pk, 'group_by': 'meal', 'filters': ['meal:meat']}))

        for bad in (
            {'group_by': 'site', 'filters': []},
            {'group_by': '', 'filters': ['site:school']},
        ):
            with pytest.raises(StructBlockValidationError):
                block.clean(block.to_python({'node': meals.pk, **bad}))


def test_the_admin_form_accepts_only_customizable_parameters(panel_config: InstanceConfig) -> None:
    from wagtail.blocks import StreamBlockValidationError, StructBlockValidationError

    block = ParameterPanelBlock()
    with realm_context.activate(RealmContext(realm=panel_config, user=None)):
        value = block.to_python({'title': '', 'parameter_ids': ['warming_limit', 'population.multiplier'], 'show_reset': True})
        block.clean(value)

        fixed = block.to_python({'title': '', 'parameter_ids': ['locked'], 'show_reset': True})
        with pytest.raises((StructBlockValidationError, StreamBlockValidationError)):
            block.clean(fixed)


def _options(panel_config: InstanceConfig, node_pk: object) -> Any:
    request = RequestFactory().get('/admin/pages/node-graph-options/', {'node': str(node_pk)})
    with realm_context.activate(RealmContext(realm=panel_config, user=None)):
        return node_graph_options(cast('Any', request))


def test_the_admin_is_told_the_dimensions_and_categories_of_the_node(panel_config: InstanceConfig) -> None:
    meals = NodeConfig.objects.get(instance=panel_config, identifier='meal_emissions')
    population = NodeConfig.objects.get(instance=panel_config, identifier='population')

    response = _options(panel_config, meals.pk)
    # The node has no input to compute from, so every category of its dimension is offered.
    assert json.loads(response.content) == {
        'dimensions': [
            {
                'id': 'meal',
                'label': 'Meal',
                'categories': [{'id': 'meat', 'label': 'Meat'}, {'id': 'vegetables', 'label': 'Vegetables'}],
            }
        ]
    }
    assert json.loads(_options(panel_config, population.pk).content) == {'dimensions': []}


def test_the_admin_is_told_only_the_categories_the_node_has_data_for(
    panel_config: InstanceConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pages import admin_views

    meals = NodeConfig.objects.get(instance=panel_config, identifier='meal_emissions')
    monkeypatch.setattr(admin_views, '_categories_with_data', lambda _node: {'meal': {'vegetables'}})

    data = json.loads(_options(panel_config, meals.pk).content)
    assert data['dimensions'][0]['categories'] == [{'id': 'vegetables', 'label': 'Vegetables'}]


def test_the_admin_is_not_told_about_other_instances_nodes(panel_config: InstanceConfig) -> None:
    other = NodeConfigFactory.create()
    for node_pk in (other.pk, 'abc', ''):
        with pytest.raises(Http404):
            _options(panel_config, node_pk)
