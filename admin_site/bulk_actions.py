from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from django.utils.translation import gettext_lazy as _
from wagtail.models import ReferenceIndex
from wagtail.snippets.bulk_actions.delete import DeleteBulkAction

from kausal_common.users import user_or_none

if TYPE_CHECKING:
    from django.db.models import Model

    from kausal_common.models.permission_policy import ModelPermissionPolicy


class PermissionAwareDeleteBulkAction(DeleteBulkAction):
    """
    Bulk Delete action which checks object-level permissions and protected references.

    The base class, Wagtail's DeleteBulkAction only checks
    Django's model permissions.
    """

    @override
    def check_perm(self, snippet: Model):
        allowed_by_super = super().check_perm(snippet)
        if not allowed_by_super:
            return False

        # Check if object has protected references that would prevent deletion
        usage = ReferenceIndex.get_grouped_references_to(snippet)
        if usage.is_protected:
            return False

        get_permission_policy = getattr(snippet, 'permission_policy', None)
        if get_permission_policy is None:
            return allowed_by_super

        permission_policy: ModelPermissionPolicy[Any] = get_permission_policy()
        user = user_or_none(self.request.user)
        if user is None:
            return False
        return permission_policy.user_has_perm(user, 'delete', snippet)

    @override
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Build dict mapping PKs to protection reasons for items that can't be deleted
        protection_reasons = {}
        for item in context.get('items_with_no_access', []):
            usage = ReferenceIndex.get_grouped_references_to(item)
            if usage.is_protected:
                refs = [str(source_obj) for source_obj, _ in usage]
                if refs:
                    protection_reasons[item.pk] = _('Referenced by: %(refs)s') % {
                        'refs': ', '.join(refs)
                    }

        context['protection_reasons'] = protection_reasons
        return context
