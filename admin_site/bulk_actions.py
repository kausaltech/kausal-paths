from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from wagtail.snippets.bulk_actions.delete import DeleteBulkAction

from kausal_common.users import user_or_none

if TYPE_CHECKING:
    from django.db.models import Model

    from kausal_common.models.permission_policy import ModelPermissionPolicy


class PermissionAwareDeleteBulkAction(DeleteBulkAction):
    """
    Bulk Delete action which checks object-level permissions.

    The base class, Wagtail's DeleteBulkAction only checks
    Django's model permissions.
    """

    @override
    def check_perm(self, snippet: Model):
        allowed_by_super = super().check_perm(snippet)
        if not allowed_by_super:
            return False

        get_permission_policy = getattr(snippet, 'permission_policy', None)
        if get_permission_policy is None:
            return allowed_by_super

        permission_policy: ModelPermissionPolicy[Any] = get_permission_policy()
        user = user_or_none(self.request.user)
        if user is None:
            return False
        return permission_policy.user_has_perm(user, 'delete', snippet)
