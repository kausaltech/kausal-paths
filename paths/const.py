from __future__ import annotations

from kausal_common.const import WILDCARD_DOMAINS_HEADER as WILDCARD_DOMAINS_HEADER  # noqa: PLC0414

INSTANCE_IDENTIFIER_HEADER = 'x-paths-instance-identifier'
INSTANCE_HOSTNAME_HEADER = 'x-paths-instance-hostname'

FRAMEWORK_ADMIN_ROLE = 'framework-admin'
FRAMEWORK_VIEWER_ROLE = 'framework-viewer'
INSTANCE_ADMIN_ROLE = 'instance-admin'
INSTANCE_VIEWER_ROLE = 'instance-viewer'

MODEL_CALC_OP = 'model.calculate'
MODEL_CACHE_OP = 'model.cache'
NODE_CALC_OP = 'node.calculate'


# Django Channels
INSTANCE_CHANGE_TYPE = 'instance.change'
INSTANCE_CHANGE_GROUP = 'instance_change'
