"""Every edit of a published instance's draft is a change the model editor must offer to publish."""

from typing import TYPE_CHECKING

import pytest

from nodes.change_ops import announce_draft_change, change_operation
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


def _published() -> InstanceConfig:
    ic = _db_instance()
    ic.save_revision().publish()
    ic.refresh_from_db()
    assert ic.live_revision_id is not None
    assert not ic.has_unpublished_changes
    return ic


def test_a_change_operation_on_a_published_instance_marks_unpublished_changes() -> None:
    ic = _published()

    with change_operation(ic, user=None, action='node.update'):
        pass

    ic.refresh_from_db()
    assert ic.has_unpublished_changes


def test_a_failed_change_operation_leaves_the_flag_alone() -> None:
    ic = _published()

    with pytest.raises(RuntimeError), change_operation(ic, user=None, action='node.update'):
        raise RuntimeError('refused')

    ic.refresh_from_db()
    assert not ic.has_unpublished_changes


def test_a_change_outside_an_operation_marks_unpublished_changes() -> None:
    ic = _published()

    announce_draft_change(ic)

    ic.refresh_from_db()
    assert ic.has_unpublished_changes


def test_a_never_published_instance_is_not_flagged() -> None:
    ic = _db_instance()

    with change_operation(ic, user=None, action='node.update'):
        pass

    ic.refresh_from_db()
    assert not ic.has_unpublished_changes
