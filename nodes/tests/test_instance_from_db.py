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
