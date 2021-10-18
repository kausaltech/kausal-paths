from django.db import models
from django.core.validators import RegexValidator
from django.utils.translation import gettext_lazy as _


class IdentifierValidator(RegexValidator):
    def __init__(self, regex=None, **kwargs):
        if regex is not None:
            regex = r'^[a-z0-9_]+$'
        super().__init__(regex, **kwargs)


class IdentifierField(models.CharField):
    def __init__(self, *args, **kwargs):
        if 'validators' not in kwargs:
            kwargs['validators'] = [IdentifierValidator()]
        if 'max_length' not in kwargs:
            kwargs['max_length'] = 50
        if 'verbose_name' not in kwargs:
            kwargs['verbose_name'] = _('identifier')
        super().__init__(*args, **kwargs)
