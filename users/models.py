import uuid

from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils.translation import gettext_lazy as _

from nodes.models import InstanceConfig


class User(AbstractUser):
    uuid = models.UUIDField(default=uuid.uuid4, unique=True)
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
