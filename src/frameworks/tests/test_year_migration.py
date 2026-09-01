from importlib import import_module

import pytest

from nodes.defs import InstanceModelSpec, YearsSpec

pytestmark = pytest.mark.django_db

update_instance_year_spec = import_module('frameworks.migrations.0025_move_instance_years_to_spec').update_instance_year_spec


def test_year_migration_preserves_spec_and_moves_framework_values() -> None:
    spec = InstanceModelSpec(years=YearsSpec(reference=2018, target=2030, model_end=2060))

    migrated = InstanceModelSpec.model_validate(
        update_instance_year_spec(
            spec,
            baseline_year=2021,
            target_year=2040,
            default_target_year=2030,
        )
    )

    assert migrated.years == YearsSpec(reference=2021, target=2040, model_end=2060)


def test_year_migration_builds_missing_spec_with_framework_default_target() -> None:
    migrated = InstanceModelSpec.model_validate(
        update_instance_year_spec(
            None,
            baseline_year=2022,
            target_year=None,
            default_target_year=2030,
        )
    )

    assert migrated.years.reference == 2022
    assert migrated.years.target == 2030
