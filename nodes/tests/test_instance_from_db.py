from __future__ import annotations

import pytest

from nodes.defs.transform_def import (
    AssignDimensionOp,
    EdgeTransformOp,
    FilterDimensionOp,
    FlattenTransformation,
)
from nodes.dimensions import Dimension, DimensionCategory
from nodes.edges import Edge
from nodes.instance_from_db import _serialize_node_config, _transforms_to_config
from nodes.instance_serialization import NodeSnapshot
from nodes.tests.factories import ContextFactory, NodeConfigFactory, NodeFactory


@pytest.mark.django_db
def test_serialize_node_config_round_trips_node_config_short_name():
    nc = NodeConfigFactory.create(short_name='Short label', i18n={'short_name_fi': 'Lyhyt nimi'})
    snapshot = NodeSnapshot.from_model(nc)

    config = _serialize_node_config(snapshot, input_nodes=[], dataset_ports=[])

    assert config['short_name_en'] == 'Short label'
    assert config['short_name_fi'] == 'Lyhyt nimi'


@pytest.mark.django_db
def test_edge_transforms_round_trip_through_config_dict():
    """
    The shim's ops→dict conversion and the edge's dict→ops conversion are inverses.

    ``_transforms_to_config()`` serializes a binding's transformations into the
    ``from_dimensions``/``to_dimensions`` dict format, ``Edge.from_config()``
    parses that back into ``EdgeDimension`` objects, and ``Edge.to_transforms()``
    recovers the pipeline. Until ``_get_output_for_target()`` executes the ops
    directly, this loop is what guarantees the DB-stored pipeline and the
    legacy edge runtime describe the same computation.
    """
    context = ContextFactory.create()
    context.dimensions['sector'] = Dimension(
        id='sector',
        label='Sector',
        categories=[DimensionCategory(id='buildings', label='Buildings'), DimensionCategory(id='transport', label='Transport')],
    )
    context.dimensions['scope'] = Dimension(
        id='scope', label='Scope', categories=[DimensionCategory(id='scope1', label='Scope 1')]
    )
    context.dimensions['ghg'] = Dimension(id='ghg', label='GHG', categories=[DimensionCategory(id='co2', label='CO2')])
    context.dimensions['energy_carrier'] = Dimension(
        id='energy_carrier', label='Energy carrier', categories=[DimensionCategory(id='electricity', label='Electricity')]
    )
    source = NodeFactory.create(context=context)
    target = NodeFactory.create(context=context)

    ops: list[EdgeTransformOp] = [
        FilterDimensionOp(dimension='sector', categories=['buildings'], flatten=True, exclude=False),
        FilterDimensionOp(dimension='scope', categories=['scope1'], flatten=False, exclude=True),
        AssignDimensionOp(dimension='ghg', category='co2'),
        FlattenTransformation(dimension='energy_carrier'),
    ]

    config = {
        'id': target.id,
        **_transforms_to_config(ops, required_dimensions=['energy_carrier']),
    }
    edge = Edge.from_config(config, node=source, is_output=True, context=context)

    assert edge.to_transforms() == ops[:-1]
    assert set(edge.to_dimensions or ()) == {'ghg', 'energy_carrier'}
