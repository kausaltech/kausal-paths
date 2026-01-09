from __future__ import annotations

from django import template

register = template.Library()


@register.filter
def get_item(dictionary, key):
    """Look up a value in a dictionary by key."""
    if dictionary is None:
        return None
    return dictionary.get(key)

