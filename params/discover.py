from __future__ import annotations

import importlib
import inspect
import pkgutil
from functools import cache
from pathlib import Path

from params.global_params import (
    BoolGlobalParameter,
    GlobalParameter,
    NumberGlobalParameter,
    StringGlobalParameter,
)

from . import base as param_base, param as typed_params
from .base import Parameter


def translation_to_param(translation: type[GlobalParameter]) -> Parameter:
    from params.param import BoolParameter, NumberParameter, StringParameter

    if issubclass(translation, NumberGlobalParameter):
        return NumberParameter(local_id=translation.id, label=translation.name)
    if issubclass(translation, StringGlobalParameter):
        return StringParameter(local_id=translation.id, label=translation.name)
    if issubclass(translation, BoolGlobalParameter):
        return BoolParameter(local_id=translation.id, label=translation.name)
    raise ValueError(f'Unknown parameter translation type: {translation}')


@cache
def discover_global_parameters() -> dict[str, Parameter]:
    """Discover all the supported parameter classes by iterating through package modules."""

    this_pkg = __package__
    this_path = Path(__file__).parent
    pkgs = pkgutil.iter_modules([str(this_path)], prefix='%s.' % this_pkg)

    all_params: dict[str, Parameter] = {}

    both_bases = param_base.__dict__ | typed_params.__dict__
    base_classes = {x for x in both_bases.values() if inspect.isclass(x) and issubclass(x, Parameter)}

    for p in pkgs:
        if p.name in ('%s.discover' % this_pkg, '%s.param' % this_pkg):
            continue

        mod = importlib.import_module(p.name)
        for attr in mod.__dict__.values():
            if not inspect.isclass(attr):
                continue
            if not issubclass(attr, (Parameter, GlobalParameter)):
                continue
            if attr in base_classes:
                continue
            param_id: str | None = getattr(attr, 'id', None)
            if param_id is None:
                continue

            if param_id in all_params:
                raise Exception('Module %s has duplicated parameter id: %s' % (p.name, param_id))
            assert issubclass(attr, GlobalParameter)
            param = translation_to_param(attr)
            all_params[param_id] = param

    return all_params
