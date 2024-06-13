from django.db import models
from django.utils.translation import gettext_lazy as _

from .base import AbstractUser


class User(AbstractUser):  # type: ignore[django-manager-missing]
    selected_instance = models.ForeignKey(
        'nodes.InstanceConfig', null=True, blank=True, on_delete=models.SET_NULL
    )
    email = models.EmailField(_('email address'), unique=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    autocomplete_search_field = 'email'

    def natural_key(self):
        # If we don't override this, it will use `get_username()`, which may not always return the email field. The
        # manager's `get_by_natural_key()`, on the other hand, will expect that the natural key is the email field since
        # we specified `USERNAME_FIELD = 'email'`. We can't just override `get_by_natural_key()` because, if I remember
        # correctly, in some places, Django expects this to actually match with field specified in `USERNAME_FIELD`.
        return (self.email,)


    def get_active_instance(self):
        # TODO
        return self.selected_instance

    def get_adminable_instances(self):
        from nodes.models import InstanceConfig
        return InstanceConfig.objects.adminable_for(self)

    def can_access_admin(self) -> bool:
        if not self.is_active:
            return False
        if not self.is_staff:
            return False
        return True
