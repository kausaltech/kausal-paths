from django.contrib.auth.models import AbstractUser
from django.db import models

from nodes.models import InstanceConfig


class User(AbstractUser):
    selected_instance = models.ForeignKey(
        'nodes.InstanceConfig', null=True, blank=True, on_delete=models.SET_NULL
    )

    def get_active_instance(self):
        # TODO
        return self.selected_instance

    def get_adminable_instances(self):
        # TODO
        return InstanceConfig.objects.all()
