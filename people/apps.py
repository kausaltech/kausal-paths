from django.apps import AppConfig
from django.utils.translation import gettext_lazy as _

class PeopleConfig(AppConfig):
    name = 'people'
    verbose_name = _('People')
