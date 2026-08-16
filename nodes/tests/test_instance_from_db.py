from __future__ import annotations

import pytest

from nodes.instance_from_db import _serialize_node_config
from nodes.instance_serialization import NodeSnapshot
from nodes.tests.factories import NodeConfigFactory


@pytest.mark.django_db
def test_serialize_node_config_round_trips_node_config_short_name():
    nc = NodeConfigFactory.create(short_name='Short label', i18n={'short_name_fi': 'Lyhyt nimi'})
    snapshot = NodeSnapshot.from_model(nc)

    config = _serialize_node_config(snapshot, input_nodes=[], dataset_ports=[])

    assert config['short_name_en'] == 'Short label'
    assert config['short_name_fi'] == 'Lyhyt nimi'


def _node_snapshot_stub(input_dims: list[str] | None = None, output_dims: list[str] | None = None):
    """Stand-in for a NodeSnapshot: the guard only reads the two dimension lists off `.spec`."""
    from types import SimpleNamespace

    return SimpleNamespace(spec=SimpleNamespace(input_dimensions=input_dims, output_dimensions=output_dims))


@pytest.mark.django_db
def test_yaml_minimal_spec_is_refused_when_nodes_declare_dimensions():
    """A `database` -> `yaml` flip empties the spec; the DB path must say so, not 'Dimension x not found'."""
    from nodes.instance_serialization import _check_spec_is_not_yaml_minimal
    from nodes.models import InstanceConfig, make_minimal_instance_spec
    from nodes.tests.factories import InstanceConfigFactory, InstanceFactory

    instance = InstanceFactory.create()
    ic: InstanceConfig = InstanceConfigFactory.create(identifier=instance.id, instance=instance)
    ic.spec = make_minimal_instance_spec(instance)
    assert not ic.spec.dimensions

    with pytest.raises(ValueError, match='sync_instance_to_db'):
        _check_spec_is_not_yaml_minimal(ic, [_node_snapshot_stub(output_dims=['sector'])])


@pytest.mark.django_db
def test_a_dimensionless_model_is_not_refused():
    """An empty catalogue is correct when nothing declares a dimension; the guard must not fire."""
    from nodes.instance_serialization import _check_spec_is_not_yaml_minimal
    from nodes.models import InstanceConfig, make_minimal_instance_spec
    from nodes.tests.factories import InstanceConfigFactory, InstanceFactory

    instance = InstanceFactory.create()
    ic: InstanceConfig = InstanceConfigFactory.create(identifier=instance.id, instance=instance)
    ic.spec = make_minimal_instance_spec(instance)

    _check_spec_is_not_yaml_minimal(ic, [_node_snapshot_stub(), _node_snapshot_stub(input_dims=[])])
