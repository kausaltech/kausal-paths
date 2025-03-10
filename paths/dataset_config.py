"""
Configure the kausal_common.datasets app.

There is some project-specific configration required for the reusable datasets apps
found in kausal_common.datasets to make it adapt to different use cases in Watch
and Paths. The configuration must be found in the module
dataset_config under the project directory.
"""
from __future__ import annotations

DATASOURCE_DEFAULT_SCOPE_CONTENT_TYPE =  ('nodes', 'instanceconfig')
