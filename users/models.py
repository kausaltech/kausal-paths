from django.db import models
from django.utils.translation import gettext_lazy as _

from nodes.models import InstanceConfig

from .base import AbstractUser


class User(AbstractUser):
    selected_instance = models.ForeignKey(
        'nodes.InstanceConfig', null=True, blank=True, on_delete=models.SET_NULL
    )
    email = models.EmailField(_('email address'), unique=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    autocomplete_search_field = 'email'

    def get_active_instance(self):
        # TODO
        return self.selected_instance

    def get_adminable_instances(self):
        # TODO
        return InstanceConfig.objects.all()

    def can_access_admin(self) -> bool:
        if not self.is_active:
            return False
        if not self.is_staff:
            return False
        return True
