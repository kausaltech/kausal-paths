"""The frame built from a dataset payload is the frame built from the live rows."""

from datetime import date
from decimal import Decimal

from django.contrib.contenttypes.models import ContentType

import pytest

from kausal_common.datasets.models import DimensionScope
from kausal_common.datasets.tests.factories import (
    DataPointFactory,
    DatasetFactory,
    DatasetMetricFactory,
    DatasetSchemaDimensionFactory,
    DimensionCategoryFactory,
    DimensionFactory,
)

from datasets.runtime import DBDataset
from datasets.snapshot import DatasetSnapshot
from nodes.defs.graph import DimensionCategoryMeta, DimensionMeta
from nodes.tests.factories import InstanceConfigFactory

pytestmark = pytest.mark.django_db


def test_payload_frame_equals_the_live_frame() -> None:
    """
    `DatasetSnapshot.to_frame` and `DBDataset.deserialize_df` must name and fill a frame alike.

    The runtime reads the payload (the materialization, or a pinned revision); the editor
    and the import read the rows. A difference between them is a model that computes
    differently depending on where its data was read from.
    """
    ic = InstanceConfigFactory.create(name='frames', config_source='database')
    ct = ContentType.objects.get_for_model(ic)
    catalog: dict = {}
    dataset = DatasetFactory.create(identifier='test/frames', scope=ic)
    categories = {}
    for identifier, column, labels in (('sector', None, ('homes', 'shops')), ('carrier', 'fuel', ('gas', 'oil'))):
        dimension = DimensionFactory.create(name=identifier)
        DimensionScope.objects.create(dimension=dimension, scope_content_type=ct, scope_id=ic.pk, identifier=identifier)
        DatasetSchemaDimensionFactory.create(schema=dataset.schema, dimension=dimension, column_name=column)
        cats = [DimensionCategoryFactory.create(dimension=dimension, identifier=label, label=label) for label in labels]
        categories[identifier] = cats
        catalog[dimension.uuid] = DimensionMeta(
            id=dimension.uuid,
            identifier=identifier,
            categories=tuple(DimensionCategoryMeta(id=c.uuid, identifier=c.identifier) for c in cats),
        )
    energy = DatasetMetricFactory.create(schema=dataset.schema, name='energy', unit='MWh')
    DatasetMetricFactory.create(schema=dataset.schema, name='cost', unit='EUR')  # no data points: a typed, empty column
    homes, shops = categories['sector']
    gas, oil = categories['carrier']
    for year, sector, carrier, value in (
        (2020, homes, gas, Decimal(5)),
        (2020, shops, oil, None),
        (2021, homes, oil, Decimal(2)),
    ):
        DataPointFactory.create(
            dataset=dataset, metric=energy, date=date(year, 1, 1), value=value, dimension_categories=[sector, carrier]
        )

    live = DBDataset.deserialize_df(dataset)
    payload = DatasetSnapshot.from_model(dataset, ic).to_frame(catalog)

    assert payload.primary_keys == live.primary_keys == ['Year', 'sector', 'fuel']
    assert payload.get_meta().units == live.get_meta().units
    assert payload.sort(payload.primary_keys).equals(live.sort(live.primary_keys).select(payload.columns))
