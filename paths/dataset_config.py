"""
Configure the kausal_common.datasets app.

There is some project-specific configration required for the reusable datasets apps
found in kausal_common.datasets to make it adapt to different use cases in Watch
and Paths. The configuration must be found in the module
dataset_config under the project directory.
"""
from __future__ import annotations

DATASOURCE_DEFAULT_SCOPE_CONTENT_TYPE: tuple[str, str] =  ('nodes', 'instanceconfig')
SCHEMA_HAS_SINGLE_DATASET: bool = True
SHOW_DATASETS_IN_MENU: bool = True
SHOW_SCHEMAS_IN_MENU: bool = False
