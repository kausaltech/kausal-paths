"""An import into a published instance is a draft change the model editor must offer to publish."""

from typing import TYPE_CHECKING

import pytest

from datasets.management.commands.load_dvc_dataset import Command
from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
from nodes.tests.factories import InstanceConfigFactory, InstanceFactory

if TYPE_CHECKING:
    from nodes.models import InstanceConfig

pytestmark = pytest.mark.django_db


def _db_instance() -> InstanceConfig:
    instance = InstanceFactory.create()
    return InstanceConfigFactory.create(
        identifier=instance.id,
        instance=instance,
        config_source='database',
        spec=InstanceModelSpec(years=YearsSpec(target=2030)),
    )


def test_an_import_into_a_published_instance_marks_unpublished_changes() -> None:
    ic = _db_instance()
    ic.save_revision().publish()
    ic.refresh_from_db()
    assert ic.live_revision_id is not None
    assert not ic.has_unpublished_changes

    Command.mark_unpublished_changes(ic)

    ic.refresh_from_db()
    assert ic.has_unpublished_changes


def test_an_import_into_a_never_published_instance_leaves_it_unflagged() -> None:
    ic = _db_instance()
    assert ic.live_revision_id is None

    Command.mark_unpublished_changes(ic)

    ic.refresh_from_db()
    assert not ic.has_unpublished_changes
