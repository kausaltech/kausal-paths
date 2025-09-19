from __future__ import annotations

from wagtail.models import PagePermissionTester
from wagtail.permission_policies.pages import PagePermissionPolicy


class PathsPagePermissionPolicy(PagePermissionPolicy):
    pass


class PathsPagePermissionTester(PagePermissionTester):
    pass
