from __future__ import annotations

from typing import Literal

from kausal_common.const import WILDCARD_DOMAINS_HEADER as WILDCARD_DOMAINS_HEADER  # noqa: PLC0414

INSTANCE_IDENTIFIER_HEADER = 'x-paths-instance-identifier'
INSTANCE_HOSTNAME_HEADER = 'x-paths-instance-hostname'

FrameworkRoleIdentifier = Literal[
    'framework-admin',
    'framework-viewer',
]
InstanceRoleIdentifier = Literal[
    'instance-super-admin',
    'instance-admin',
    'instance-viewer',
    'instance-reviewer',
]
PathsRoleIdentifier = FrameworkRoleIdentifier | InstanceRoleIdentifier | Literal['none']

FRAMEWORK_ADMIN_ROLE: FrameworkRoleIdentifier = 'framework-admin'
FRAMEWORK_VIEWER_ROLE: FrameworkRoleIdentifier = 'framework-viewer'
INSTANCE_SUPER_ADMIN_ROLE: InstanceRoleIdentifier = 'instance-super-admin'
INSTANCE_ADMIN_ROLE: InstanceRoleIdentifier = 'instance-admin'
INSTANCE_VIEWER_ROLE: InstanceRoleIdentifier = 'instance-viewer'
INSTANCE_REVIEWER_ROLE: InstanceRoleIdentifier = 'instance-reviewer'

NONE_ROLE = 'none'

MODEL_CALC_OP = 'model.calculate'
MODEL_CACHE_OP = 'model.cache'
NODE_CALC_OP = 'node.calculate'


# Django Channels
INSTANCE_CHANGE_TYPE = 'instance.change'
INSTANCE_CHANGE_GROUP = 'instance_change'
