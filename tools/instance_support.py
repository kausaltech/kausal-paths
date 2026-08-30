"""
Load a model instance and its nodes, for command-line tools.

Split out of `notebooks/notebook_support.py`, which mixed two unrelated things: an
IPython bootstrap (magics, `ipympl`, plotly) and generic instance loading. Only the
latter is here, because the callers are `tools/upload_new_dataset.py` and
`tools/collect_city_data.py` -- ordinary CLIs with nothing notebook-shaped about them.
That mattered once they moved into `tools/`: `notebooks/*` is excluded from ruff and
mypy (`pyproject.toml`), and a checked package must not import an unchecked one.

`notebooks/notebook_support.py` keeps the notebook half and re-exports these two, so a
notebook that already imports them from there is unaffected.
"""

# ruff: noqa: INP001  # tools/ is an implicit namespace package by design; run with `-m`.

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodes.context import Context
    from nodes.models import InstanceConfig
    from nodes.node import Node

# Django models are imported inside the functions, not at module scope, so that this
# module can be imported before `init_django()` has run -- which is what every caller
# does, since they configure Django in their own `__main__` preamble.


def _get_instance_from_db(instance_id: str) -> InstanceConfig | None:
    from nodes.models import InstanceConfig

    ic = InstanceConfig.objects.filter(identifier=instance_id).first()
    if ic is None:
        return None
    return ic


def get_context(instance_id: str) -> Context:
    """
    Return the context of `instance_id`, from the database if it has a row and from YAML if not.

    NOTE: that rule is a heuristic, not the instance's declared `config_source`. An
    instance whose `config_source` is `yaml` but which also has an `InstanceConfig` row
    -- the normal state, since `ensure_spec()` derives a minimal one -- is loaded here
    from the database, which is not how the site loads it. `tools/debug_instance.py`
    does this properly (`_load_from_yaml` / `_load_from_db`, honouring `config_source`
    unless `--source` overrides it), and these two should be consolidated onto it.
    Behaviour is preserved verbatim in the move; changing it is a separate change.
    """
    from common import polars_ext  # noqa: F401
    from nodes.instance_loader import InstanceLoader

    ic = _get_instance_from_db(instance_id)
    if ic is not None:
        return ic.get_instance().context

    project_root = Path(__file__).parent.parent
    config_fn = (Path(project_root) / 'configs' / ('%s.yaml' % instance_id)).resolve()
    loader = InstanceLoader.from_yaml(config_fn)
    context = loader.context
    context.cache.clear()
    return context


class InstanceNodes(dict[str, 'Node']):
    """The instance's nodes by id, with the context they came from kept alongside."""

    context: Context


def get_nodes(instance_id: str) -> InstanceNodes:
    context = get_context(instance_id)
    out = InstanceNodes(context.nodes)
    out.context = context
    return out
