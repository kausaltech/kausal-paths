from __future__ import annotations

import pytest

from kausal_common.i18n.pydantic import TranslatedString

from nodes.instance_from_db import _serialize_node_config
from nodes.tests.factories import NodeConfigFactory


@pytest.mark.django_db
def test_serialize_node_config_round_trips_node_spec_short_name():
    nc = NodeConfigFactory.create()
    assert nc.spec is not None
    nc.spec.short_name = TranslatedString(en='Short label', fi='Lyhyt nimi')

    config = _serialize_node_config(nc, input_nodes=[], dataset_ports=[])

    assert config['short_name_en'] == 'Short label'
    assert config['short_name_fi'] == 'Lyhyt nimi'
