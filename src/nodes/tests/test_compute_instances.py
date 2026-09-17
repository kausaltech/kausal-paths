from contextlib import contextmanager
from io import StringIO
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from django.core.management import call_command
from django.core.management.base import CommandError

import pytest

from nodes.management.commands.compute_instances import Command
from nodes.models import PreferredInstanceSource
from nodes.tests.factories import InstanceConfigFactory

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.django_db


def test_customer_selection_is_active_and_opt_in() -> None:
    customer = InstanceConfigFactory.create(name='customer', in_customer_use=True)
    InstanceConfigFactory.create(name='unused', in_customer_use=False)
    InstanceConfigFactory.create(name='inactive-customer', in_customer_use=True, is_active=False)
    with patch.object(Command, 'compute_instance') as compute:
        call_command('compute_instances', in_customer_use=True, stdout=StringIO())
    assert [args.args[0].pk for args in compute.call_args_list] == [customer.pk]


def test_selection_required_and_inactive_identifiers_rejected() -> None:
    inactive = InstanceConfigFactory.create(name='inactive', is_active=False)
    with pytest.raises(CommandError, match='Provide instance identifiers'):
        call_command('compute_instances')
    with pytest.raises(CommandError, match='No selected active instance'):
        call_command('compute_instances', inactive.identifier)


def test_failure_continues_to_next_instance() -> None:
    configs = [InstanceConfigFactory.create(name=name, identifier=name) for name in ['first', 'second']]
    with (
        patch.object(Command, 'compute_instance', side_effect=[ValueError('broken'), None]) as compute,
        pytest.raises(CommandError, match='Failed instances: first'),
    ):
        call_command('compute_instances', *[config.identifier for config in configs], stdout=StringIO(), stderr=StringIO())
    assert [args.args[0].identifier for args in compute.call_args_list] == ['first', 'second']


@pytest.mark.parametrize('fail', [True, False])
def test_compute_outcomes_in_default_and_baseline_and_clean(fail: bool) -> None:
    config = MagicMock()
    instance = config.enter_instance_context.return_value.__enter__.return_value
    context = instance.context
    default, baseline = MagicMock(), MagicMock()
    context.get_default_scenario.return_value = default
    context.scenarios = {'baseline': baseline}
    current = []
    computed = []

    @contextmanager
    def scenario_override(name: str, *, set_active: bool) -> Iterator[None]:
        assert set_active
        current.append(name)
        try:
            yield
        finally:
            current.pop()

    default.override.side_effect = lambda **kwargs: scenario_override('default', **kwargs)
    baseline.override.side_effect = lambda **kwargs: scenario_override('baseline', **kwargs)
    node = MagicMock()
    context.get_outcome_nodes.return_value = [node]

    def output() -> None:
        computed.append(current[-1])
        if fail:
            raise ValueError('failed output')

    node.get_output_pl.side_effect = output
    if fail:
        with pytest.raises(ValueError, match='failed output'):
            Command().compute_instance(config)
    else:
        Command().compute_instance(config)
    assert computed == (['default'] if fail else ['default', 'baseline'])
    assert current == []
    config.enter_instance_context.assert_called_once_with(source=PreferredInstanceSource.PUBLISHED)
    instance.clean.assert_called_once()


def test_no_selected_customers_is_successful() -> None:
    output = StringIO()
    with patch.object(Command, 'compute_instance') as compute:
        call_command('compute_instances', in_customer_use=True, stdout=output)
    compute.assert_not_called()
    assert 'Computed 0 instances; 0 failed' in output.getvalue()
