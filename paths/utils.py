from __future__ import annotations

import uuid

from django import forms
from django.conf import settings
from django.contrib.postgres.fields import ArrayField
from django.core.exceptions import ValidationError
from django.core.validators import RegexValidator
from django.db import models
from django.utils.translation import gettext, gettext_lazy as _

from pint.errors import UndefinedUnitError

from kausal_common.models.ordered import OrderedModel as OrderedModel  # noqa: PLC0414
from kausal_common.models.types import copy_signature


class IdentifierValidator(RegexValidator):
    regex = r'^[a-z0-9-_]+$'


class InstanceIdentifierValidator(RegexValidator):
    regex = r'^[a-z0-9-]+$'


class IdentifierField(models.CharField):
    def __init__(self, *args, **kwargs):
        validator_kwargs = {}
        if 'regex' in kwargs:
            validator_kwargs['regex'] = kwargs.pop('regex')
        if 'validators' not in kwargs:
            kwargs['validators'] = [IdentifierValidator(**validator_kwargs)]
        if 'max_length' not in kwargs:
            kwargs['max_length'] = 50
        if 'verbose_name' not in kwargs:
            kwargs['verbose_name'] = _('identifier')
        super().__init__(*args, **kwargs)


class ChoiceArrayField(ArrayField):
    """
    A field that allows us to store an array of choices.

    Uses Django 1.9's postgres ArrayField
    and a MultipleChoiceField for its formfield.
    """

    @copy_signature(ArrayField.formfield)
    def formfield(self, **kwargs) -> forms.Field:
        defaults = {
            'form_class': forms.MultipleChoiceField,
            'choices': self.base_field.choices,
        }
        defaults.update(kwargs)
        # Skip our parent's formfield implementation completely as we don't
        # care for it.
        return super(ArrayField, self).formfield(**defaults)  # type: ignore[arg-type]


def validate_unit(s: str):
    from nodes.context import unit_registry

    try:
        unit = unit_registry.parse_units(s)
    except UndefinedUnitError as e:
        if isinstance(e.unit_names, str):
            unit_str = e.unit_names
        else:
            unit_str = ', '.join(e.unit_names)
        raise ValidationError('%s: %s' % (gettext("Invalid unit"), unit_str)) from None
    except (ValueError, TypeError):
        raise ValidationError(gettext("Invalid unit")) from None
    return unit


class UnitField(models.CharField):
    def __init__(self, *args, **kwargs):
        defaults = dict(
            validators=[validate_unit],
            verbose_name=_('unit'),
            blank=True,
            max_length=50,
        )
        for key, val in defaults.items():
            if key not in kwargs:
                kwargs[key] = val
        super().__init__(*args, **kwargs)


class UUIDIdentifierField(models.UUIDField):
    def __init__(self, *args, **kwargs):
        defaults = dict(
            editable=False,
            verbose_name=_('uuid'),
            unique=True,
            default=uuid.uuid4,
        )
        for key, val in defaults.items():
            if key not in kwargs:
                kwargs[key] = val
        super().__init__(*args, **kwargs)


class UserModifiableModel(models.Model):
    created_at = models.DateTimeField(verbose_name=_('created at'), editable=False, auto_now_add=True)
    created_by = models.ForeignKey('users.User', null=True, on_delete=models.SET_NULL, editable=False, related_name='+')
    updated_at = models.DateTimeField(verbose_name=_('updated at'), editable=False, auto_now=True)
    updated_by = models.ForeignKey('users.User', null=True, on_delete=models.SET_NULL, editable=False, related_name='+')

    class Meta:
        abstract = True


def get_supported_languages():
    yield from settings.LANGUAGES


def get_default_language() -> str:
    return settings.LANGUAGES[0][0]
