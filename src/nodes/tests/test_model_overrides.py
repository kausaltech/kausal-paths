"""
`InstanceType.model(...)` with arguments: a runtime of its own, next to the request's default one.

What these pin down is that the two never mix: fields under an overridden `model` read
the overridden runtime, fields elsewhere read the plain one, and nothing reaches the
visitor's session.
"""

from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest

from nodes.defs.instance_defs import InstanceModelSpec, NormalizationSpec, YearsSpec
from nodes.defs.node_defs import NodeSpec, SimpleConfig
from nodes.defs.port_def import OutputPortDef
from nodes.models import test_instance_registry
from nodes.scenario import Scenario, ScenarioKind
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory, NodeConfigFactory
from nodes.units import unit_registry
from params.overrides import InvalidModelOverrideError, ModelOverrides, ParameterOverride
from params.param import BoolParameter, NumberParameter
from params.storage import InstanceData

if TYPE_CHECKING:
    from paths.tests.graphql import PathsTestClient

    from nodes.context import Context
    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db

gql = str


@pytest.fixture
def overridable_config() -> InstanceConfig:
    """Return a database-backed instance with a customizable parameter, a second scenario and a normalization."""
    instance = InstanceFactory.create()
    spec = InstanceModelSpec(
        years=YearsSpec(reference=2020, min_historical=2010, max_historical=2022, target=2030),
        params=[
            NumberParameter(local_id='factor', value=2.0, is_customizable=True),
            BoolParameter(local_id='locked', value=False, is_customizable=False),
        ],
        scenarios=[
            Scenario(id='default', name='Default', kind=ScenarioKind.DEFAULT),
            Scenario(id='other', name='Other', kind=ScenarioKind.BASELINE, param_values={'factor': 7.0}),
        ],
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
            type_config=SimpleConfig(node_class='nodes.simple.SimpleNode'),
            output_ports=[OutputPortDef(id=uuid4(), unit=unit_registry.parse_units('cap'), quantity='population')],
        ),
    )
    assert ic.spec is not None
    ic.spec.normalizations = [
        NormalizationSpec.model_validate({
            'normalizer_node_id': 'population',
            'quantities': [{'id': 'emissions', 'unit': 't/cap/a'}],
        })
    ]
    ic.save(update_fields=['spec'])
    # Build real runtimes from the database rather than the shared test instance, so
    # that each `model(...)` gets one of its own, as it does in production.
    test_instance_registry.pop(ic.identifier, None)
    return ic


@pytest.fixture
def client_for(client, overridable_config: InstanceConfig) -> PathsTestClient:
    from paths.tests.graphql import PathsTestClient

    from users.tests.factories import UserFactory

    client.force_login(UserFactory.create(is_superuser=True))
    tc = PathsTestClient(client)
    tc.set_instance(overridable_config)
    return tc


STATE = gql("""
    fragment State on InstanceModel {
        activeScenario { id }
        activeNormalization { id }
        parameters {
            id
            ... on NumberParameterType { value }
        }
    }
""")


def _factor(model: dict[str, Any]) -> float:
    return next(param['value'] for param in model['parameters'] if param['id'] == 'factor')


def test_overridden_and_plain_models_resolve_side_by_side(client_for: PathsTestClient):
    data = client_for.query_data(
        STATE
        + gql("""
        query SideBySide {
            instance {
                plain: model { ...State }
                changed: model(parameters: [{ id: "factor", numberValue: 5 }], normalizer: "population") { ...State }
                other: model(scenario: "other") { ...State }
            }
            parameters { id ... on NumberParameterType { value } }
            activeScenario { id }
        }
    """)
    )
    plain = data['instance']['plain']
    changed = data['instance']['changed']
    other = data['instance']['other']

    assert _factor(plain) == 2.0
    assert plain['activeScenario'] == {'id': 'default'}
    assert plain['activeNormalization'] is None

    # A parameter is applied the way `setParameter` applies it: as the custom scenario.
    assert _factor(changed) == 5.0
    assert changed['activeScenario'] == {'id': 'custom'}
    assert changed['activeNormalization'] == {'id': 'population'}

    assert _factor(other) == 7.0
    assert other['activeScenario'] == {'id': 'other'}

    # The top-level fields read the operation's plain runtime, untouched by the overrides.
    assert _factor({'parameters': data['parameters']}) == 2.0
    assert data['activeScenario'] == {'id': 'default'}


def test_overrides_are_not_stored_in_the_session(client_for: PathsTestClient):
    client_for.query_data(
        gql("""
        query Overridden {
            instance { model(parameters: [{ id: "factor", numberValue: 5 }], scenario: "other") { activeScenario { id } } }
        }
    """)
    )
    data = client_for.query_data(
        gql("""
        query Plain { activeScenario { id } parameters { id ... on NumberParameterType { value isCustomized } } }
    """)
    )
    assert data['activeScenario'] == {'id': 'default'}
    factor = next(param for param in data['parameters'] if param['id'] == 'factor')
    assert factor == {'id': 'factor', 'value': 2.0, 'isCustomized': False}


def test_parameters_override_on_top_of_the_scenario_given(client_for: PathsTestClient):
    data = client_for.query_data(
        STATE
        + gql("""
        query OnTopOfScenario {
            instance { model(scenario: "other", parameters: [{ id: "factor", numberValue: 5 }]) { ...State
                scenarios { id baseScenario { id } customizedParameters }
            } }
        }
    """)
    )
    model = data['instance']['model']
    assert _factor(model) == 5.0
    custom = next(scenario for scenario in model['scenarios'] if scenario['id'] == 'custom')
    assert custom['baseScenario'] == {'id': 'other'}
    assert custom['customizedParameters'] == ['factor']


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ('parameters: [{ id: "nope", numberValue: 1 }]', 'Parameter nope does not exist'),
        ('parameters: [{ id: "locked", boolValue: true }]', 'Parameter locked is not customizable'),
        ('parameters: [{ id: "factor", boolValue: true }]', "You must specify 'numberValue'"),
        ('scenario: "nope"', "Scenario 'nope' not found"),
        ('normalizer: "nope"', "Normalization 'nope' not found"),
    ],
)
def test_invalid_overrides_are_refused(client_for: PathsTestClient, arguments: str, message: str):
    client_for.query_errors(
        'query Invalid { instance { model(%s) { activeScenario { id } } } }' % arguments,
        assert_error_message=message,
    )


def test_null_normalizer_turns_normalization_off(client_for: PathsTestClient, overridable_config: InstanceConfig):
    assert overridable_config.spec is not None
    overridable_config.spec.normalizations[0].default = True
    overridable_config.save(update_fields=['spec'])

    data = client_for.query_data(
        gql("""
        query NormalizationOff {
            instance {
                plain: model { activeNormalization { id } }
                off: model(normalizer: null) { activeNormalization { id } }
            }
        }
    """)
    )
    assert data['instance']['plain']['activeNormalization'] == {'id': 'population'}
    assert data['instance']['off']['activeNormalization'] is None


# -- The branch rule, without a request ---------------------------------------------


def test_overriding_in_a_named_scenario_branches_from_it(context: Context, custom_scenario, baseline_scenario):
    param = NumberParameter(local_id='factor', value=1.0, is_customizable=True)
    context.add_global_parameter(param)
    session = InstanceData(params={'stale': 1}, active_scenario=baseline_scenario.id)

    storage = ModelOverrides(parameters=(ParameterOverride(id='factor', number_value=3.0),)).storage_for(context, session)

    assert storage.get_active_scenario() == custom_scenario.id
    assert storage.get_custom_base() == baseline_scenario.id
    assert storage.get_customized_param_values() == {'factor': 3.0}, 'the stored edits of another branch survived'
    assert session.params == {'stale': 1}, 'the session data was modified'


def test_overriding_in_the_custom_scenario_adds_to_its_edits(context: Context, custom_scenario):
    param = NumberParameter(local_id='factor', value=1.0, is_customizable=True)
    context.add_global_parameter(param)
    session = InstanceData(params={'earlier': 1}, active_scenario=custom_scenario.id, custom_base='baseline')

    storage = ModelOverrides(parameters=(ParameterOverride(id='factor', number_value=3.0),)).storage_for(context, session)

    assert storage.get_custom_base() == 'baseline'
    assert storage.get_customized_param_values() == {'earlier': 1, 'factor': 3.0}


def test_a_parameter_may_be_overridden_once():
    with pytest.raises(InvalidModelOverrideError, match='only once'):
        ModelOverrides(parameters=(ParameterOverride(id='a', number_value=1.0), ParameterOverride(id='a', number_value=2.0)))
