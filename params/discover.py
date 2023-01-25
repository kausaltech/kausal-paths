import inspect
import importlib
import os
import pkgutil

from . import param as param_base


def discover_parameter_types():
    """Discover all the supported parameter classes by iterating through package modules."""

    this_pkg = __package__
    this_path = os.path.dirname(__file__)
    pkgs = pkgutil.iter_modules([this_path], prefix='%s.' % this_pkg)

    all_params = {}

    base_classes = {
        x for x in param_base.__dict__.values()
        if inspect.isclass(x) and issubclass(x, param_base.Parameter)
    }

    for p in pkgs:
        if p.name in ('%s.discover' % this_pkg, '%s.param' % this_pkg):
            continue

        mod = importlib.import_module(p.name)
        for attr in mod.__dict__.values():
            if not inspect.isclass(attr):
                continue
            if attr in base_classes:
                continue
            if not issubclass(attr, param_base.Parameter):
                continue
            if not hasattr(attr, 'id'):
                continue
            if attr.id in all_params:
                raise Exception("Module %s has duplicated parameter id: %s" % (p.name, attr.id))
            all_params[attr.id] = attr

    return all_params
