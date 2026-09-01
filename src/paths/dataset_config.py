"""
Configure the kausal_common.datasets app.

There is some project-specific configration required for the reusable datasets apps
found in kausal_common.datasets to make it adapt to different use cases in Watch
and Paths. The configuration must be found in the module
dataset_config under the project directory.
"""

from __future__ import annotations

import typing

import paths.utils

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from django.db.models import Model


def schema_default_scope():
    # Only call in view contexts where the context has been initialized
    from paths.context import realm_context

    return realm_context.get().realm


def validate_unit(unit: str) -> None:
    """Raise `ValidationError` if `unit` is not a valid unit."""
    paths.utils.validate_unit(unit)


def validate_metric_spec(spec: dict[str, typing.Any]) -> None:
    """Raise `ValidationError` if `spec` is not a valid `DatasetMetric.spec` payload."""
    from django.core.exceptions import ValidationError
    from pydantic import ValidationError as PydanticValidationError

    from datasets.defs import DatasetMetricSpec

    try:
        DatasetMetricSpec.model_validate(spec)
    except PydanticValidationError as e:
        errors = '; '.join(err['msg'] for err in e.errors())
        raise ValidationError(f'Invalid metric spec: {errors}') from None


DATA_SOURCE_DEFAULT_SCOPE_CONTENT_TYPE: tuple[str, str] = ('nodes', 'instanceconfig')
SCHEMA_HAS_SINGLE_DATASET: bool = True
SCHEMA_DEFAULT_SCOPE_FUNCTION: Callable[[], Model] | None = schema_default_scope
SHOW_DATASETS_IN_MENU: bool = True
SHOW_SCHEMAS_IN_MENU: bool = False

SCHEMA_PERMISSION_POLICY: str = 'paths.dataset_permission_policy.DatasetSchemaPermissionPolicy'
DATASET_PERMISSION_POLICY: str = 'paths.dataset_permission_policy.DatasetPermissionPolicy'
DATA_POINT_PERMISSION_POLICY: str = 'paths.dataset_permission_policy.DataPointPermissionPolicy'
DATA_SOURCE_PERMISSION_POLICY: str = 'paths.dataset_permission_policy.DataSourcePermissionPolicy'
DATA_POINT_COMMENT_PERMISSION_POLICY: str = 'paths.dataset_permission_policy.DataPointCommentPermissionPolicy'
DATASET_SOURCE_REFERENCE_PERMISSION_POLICY: str = 'paths.dataset_permission_policy.DatasetSourceReferencePermissionPolicy'
DATASET_METRIC_PERMISSION_POLICY: str = 'paths.dataset_permission_policy.DatasetMetricPermissionPolicy'
