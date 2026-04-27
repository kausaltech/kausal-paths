import os
import sys

if 'polars' in sys.modules:
    raise RuntimeError('Polars was already imported. `common.polars` must be imported before `polars`.')

# This has to be set; otherwise the process' RSS will grow indefinitely.
# Also, it has to be set before polars gets imported.
# See: https://github.com/pola-rs/polars/issues/23128
os.environ['_RJEM_MALLOC_CONF'] = 'dirty_decay_ms:0,muzzy_decay_ms:0'

import polars as pl  # noqa: F401  # pyright: ignore[reportUnusedImport]

from . import polars_ext as polars_ext
