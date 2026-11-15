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

from wagtail.models import Locale, Page

import pytest

from paths.context import RealmContext, realm_context

from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.defs.node_defs import NodeSpec, SimpleConfig
from nodes.defs.port_def import OutputPortDef
from nodes.models import NodeConfig, test_instance_registry
from nodes.scenario import Scenario, ScenarioKind
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory
from nodes.units import unit_registry
from pages.blocks import ParameterPanelBlock, admin_instance_parameter_choices
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
    )
    ic = InstanceConfigFactory.create(
        identifier=instance.id,
        instance=instance,
        config_source='database',
        owner='Test Owner',
        spec=spec,
    )
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

    _outcome_page(
        panel_config,
        {
            'title': 'Your premises',
            'parameter_ids': ['population.multiplier', 'warming_limit', 'population.multiplier'],
            'show_reset': True,
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


def test_parameters_the_served_instance_lacks_or_fixes_are_left_out(panel_config: InstanceConfig) -> None:
    # As when the published revision differs from the draft the editor chose from.
    context = panel_config.get_instance().context
    info: Any = type('Info', (), {'context': type('Ctx', (), {'require_instance': lambda _: context.instance})()})()

    params = ParameterPanelBlock().parameters(
        info, {'parameter_ids': ['removed', 'locked', 'nonexistent_node.multiplier', 'warming_limit']}
    )

    assert [param.global_id for param in params] == ['warming_limit']


def test_the_admin_form_accepts_only_customizable_parameters(panel_config: InstanceConfig) -> None:
    from wagtail.blocks import StreamBlockValidationError, StructBlockValidationError

    block = ParameterPanelBlock()
    with realm_context.activate(RealmContext(realm=panel_config, user=None)):
        value = block.to_python({'title': '', 'parameter_ids': ['warming_limit', 'population.multiplier'], 'show_reset': True})
        block.clean(value)

        fixed = block.to_python({'title': '', 'parameter_ids': ['locked'], 'show_reset': True})
        with pytest.raises((StructBlockValidationError, StreamBlockValidationError)):
            block.clean(fixed)
