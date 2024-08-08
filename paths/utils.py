import uuid

from django import forms
from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import RegexValidator
from django.utils.translation import gettext, gettext_lazy as _
from pint.errors import UndefinedUnitError


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

    def formfield(self, **kwargs):
        defaults = {
            'form_class': forms.MultipleChoiceField,
            'choices': self.base_field.choices,
        }
        defaults.update(kwargs)
        # Skip our parent's formfield implementation completely as we don't
        # care for it.
        # pylint:disable=bad-super-call
        return super(ArrayField, self).formfield(**defaults)


def validate_unit(s: str):
    from nodes.context import unit_registry

    try:
        unit = unit_registry.parse_units(s)
    except UndefinedUnitError as e:
        if isinstance(e.unit_names, str):
            unit_str = e.unit_names
        else:
            unit_str = ', '.join(e.unit_names)
        raise ValidationError('%s: %s' % (gettext("Invalid unit"), unit_str))
    except (ValueError, TypeError):
        raise ValidationError(gettext("Invalid unit"))
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
            default=uuid.uuid4
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


class OrderedModel(models.Model):
    order = models.PositiveIntegerField(default=0, editable=True, verbose_name=_('order'))
    sort_order_field = 'order'

    def __init__(self, *args, order_on_create=None, **kwargs):
        """
        Specify `order_on_create` to set the order to that value when saving if the instance is being created. If it is
        None, the order will instead be set to <maximum existing order> + 1.
        """
        super().__init__(*args, **kwargs)
        self.order_on_create = order_on_create

    @property
    def sort_order(self):
        return self.order

    def get_sort_order_max(self):
        """
        Method used to get the max sort_order when a new instance is created.
        If you order depends on a FK (eg. order of books for a specific author),
        you can override this method to filter on the FK.
        ```
        def get_sort_order_max(self):
            qs = self.__class__.objects.filter(author=self.author)
            return qs.aggregate(Max(self.sort_order_field))['sort_order__max'] or 0
        ```
        """
        qs = self.__class__.objects.all()
        if hasattr(self, 'filter_siblings'):
            qs = self.filter_siblings(qs)  # type: ignore

        return qs.aggregate(models.Max(self.sort_order_field))['%s__max' % self.sort_order_field] or 0

    def save(self, *args, **kwargs):
        if self.pk is None:
            if getattr(self, 'order_on_create', None) is not None:
                self.order = self.order_on_create
            else:
                self.order = self.get_sort_order_max() + 1
        super().save(*args, **kwargs)

    class Meta:
        abstract = True


def get_supported_languages():
    for x in settings.LANGUAGES:
        yield x


def get_default_language():
    return settings.LANGUAGES[0][0]
