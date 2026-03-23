from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from django import forms
from django.conf import settings
from django.contrib.postgres.fields import ArrayField
from django.core.exceptions import ValidationError
from django.core.validators import RegexValidator
from django.db import models
from django.utils.translation import gettext, gettext_lazy as _

from kausal_common.models.ordered import OrderedModel as OrderedModel  # noqa: PLC0414

if TYPE_CHECKING:
    from collections.abc import Generator

    from django_stubs_ext import StrOrPromise


class IdentifierValidator(RegexValidator):
    regex = r'^[a-z0-9-_]+$'


class InstanceIdentifierValidator(RegexValidator):
    regex = r'^[a-z0-9-]+$'


class IdentifierField[ST: str | None = str, GT: str | None = str](models.CharField[ST, GT]):
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


class ChoiceArrayField[ST, GT](ArrayField[ST, GT]):
    """
    A field that allows us to store an array of choices.

    Uses Django 1.9's postgres ArrayField
    and a MultipleChoiceField for its formfield.
    """

    def formfield(
        self, form_class: type[forms.Field] | None = None, choices_form_class: type[forms.ChoiceField] | None = None, **kwargs
    ) -> forms.Field | None:
        form_class = form_class or forms.MultipleChoiceField
        choices = kwargs.pop('choices', self.base_field.choices)
        # Skip our parent's formfield implementation completely as we don't
        # care for it.
        return super(ArrayField, self).formfield(
            form_class=form_class, choices_form_class=choices_form_class, choices=choices, **kwargs
        )


def validate_unit(s: str):
    from pint.errors import UndefinedUnitError

    from nodes.units import unit_registry

    try:
        unit = unit_registry.parse_units(s)
    except UndefinedUnitError as e:
        if isinstance(e.unit_names, str):
            unit_str = e.unit_names
        else:
            unit_str = ', '.join(e.unit_names)
        raise ValidationError('%s: %s' % (gettext('Invalid unit'), unit_str)) from None
    except ValueError, TypeError:
        raise ValidationError(gettext('Invalid unit')) from None
    return unit


class UnitField[T: str | None = str](models.CharField[T, T]):
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


class UUIDIdentifierField(models.UUIDField[uuid.UUID, uuid.UUID]):
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


def get_supported_languages() -> Generator[tuple[str, StrOrPromise]]:
    yield from settings.LANGUAGES


def get_default_language() -> str:
    return settings.LANGUAGES[0][0]
