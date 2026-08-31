from importlib import import_module

import pytest

from nodes.defs import InstanceModelSpec, YearsSpec

pytestmark = pytest.mark.django_db

enable_user_management = import_module('frameworks.migrations.0026_backfill_instance_user_management').enable_user_management


def test_user_management_migration_preserves_other_spec_fields() -> None:
    spec = InstanceModelSpec(years=YearsSpec(reference=2020, target=2030))
    spec.features.show_refresh_prompt = True

    migrated = InstanceModelSpec.model_validate(enable_user_management(spec))

    assert migrated.features.enable_user_management is True
    assert migrated.features.show_refresh_prompt is True
    assert migrated.years == spec.years


def test_user_management_migration_accepts_serialized_spec() -> None:
    migrated = InstanceModelSpec.model_validate(enable_user_management({'features': {}}))

    assert migrated.features.enable_user_management is True
