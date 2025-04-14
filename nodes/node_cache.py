from __future__ import annotations

import base64
import hashlib
import inspect
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from pprint import pprint
from time import time_ns
from typing import TYPE_CHECKING, Any, Concatenate, cast

import sentry_sdk
import xxhash

from nodes.datasets import DVCDataset
from nodes.exceptions import NodeHashingError

if TYPE_CHECKING:
    from common.cache import CacheResult
    from common.polars import PathsDataFrame
    from nodes.context import Context

    from .node import Node


DEBUG_CACHE_MISSES: bool = os.environ.get('DEBUG_CACHE_MISSES', '0') == '1'


class_fname_cache: dict[type, str] = {}

@dataclass(slots=True)
class HashingState:
    node_class_hash: dict[type, int] = field(default_factory=dict)
    """Mapping between class and module mtime"""

    upstream_node_keys: dict[str, str] = field(default_factory=dict)
    """Mapping between node ID and node cache key"""

    upstream_dataset_keys: dict[str, set[str]] = field(default_factory=dict)
    """Mapping between node ID and dataset cache keys"""

type NodeHasherFuncT[**P, R] = Callable[Concatenate[NodeHasher, P], R]


@dataclass
class NodeHasher:
    node: Node

    # When was this node last changed (timestamp in ns)
    modified_at: int | None = field(init=False, default=None)
    last_hash: bytes | None = field(init=False, default=None)
    last_hash_time: int | None = field(init=False, default=None)
    param_hash: bytes | None = field(init=False, default=None)
    mtime_hash: bytes | None = field(init=False, default=None)
    dim_hash: bytes | None = field(init=False, default=None)
    metrics_hash: bytes | None = field(init=False, default=None)

    @staticmethod
    def _wrap_hashing_error[**P, R](func: NodeHasherFuncT[P, R]) -> NodeHasherFuncT[P, R]:
        @wraps(func)
        def report_error(*args, **kwargs) -> Any:
            _rich_traceback_omit = True
            try:
                return func(*args, **kwargs)
            except Exception as e:
                node = args[0]
                if isinstance(e, NodeHashingError):
                    e.add_node(node)
                    raise
                raise NodeHashingError(args[0], "Unable to hash node") from e
        return cast(NodeHasherFuncT[P, R], report_error)

    def _get_cached_hash(self) -> bytes | None:
        if self.last_hash is None or self.last_hash_time is None:
            return None
        if self.modified_at is None or self.modified_at <= self.last_hash_time:
            return self.last_hash
        return None

    @classmethod
    def get_class_hash(cls, node_class: type, state: HashingState) -> bytes:
        mtime_hash = b''

        for parent_class in node_class.mro():
            if parent_class is object:
                continue
            if parent_class in state.node_class_hash:
                mod_mtime = state.node_class_hash[parent_class]
            else:
                fn = class_fname_cache.get(parent_class)
                if fn is None:
                    fn = inspect.getfile(parent_class)
                    class_fname_cache[parent_class] = fn

                try:
                    mod_mtime = Path(fn).stat().st_mtime_ns
                except TypeError:
                    continue
                state.node_class_hash[parent_class] = mod_mtime
            mtime_hash += str(mod_mtime).encode('ascii')
        return mtime_hash

    def calculate_hash(self, state: HashingState) -> bytes:
        try:
            ret = self._calculate_hash(state)
            state.upstream_node_keys[self.node.id] = self._get_cache_key(self.node, ret)
        except Exception as e:
            if isinstance(e, NodeHashingError):
                e.add_node(self.node)
                raise
            raise NodeHashingError(self.node, "Unable to hash node") from e
        return ret

    def is_run_cached(self) -> bool:
        cached_hash = self._get_cached_hash()
        if cached_hash is None:
            return False
        key = self._get_cache_key(self.node, cached_hash)
        cache = self.node.context.cache
        return cache.is_run_cached(key=key)

    def is_local_cached(self) -> bool:
        cached_hash = self._get_cached_hash()
        if cached_hash is None:
            return False
        key = self._get_cache_key(self.node, cached_hash)
        cache = self.node.context.cache
        return cache.is_local_cached(key=key)

    def is_cached(self, run: bool = True, local: bool = False) -> bool:
        # FIXME
        return self.is_run_cached()

    def _calculate_hash(self, state: HashingState) -> bytes:  # noqa: C901, PLR0912, PLR0915
        cached_hash = self._get_cached_hash()
        if cached_hash is not None:
            return cached_hash

        h = xxhash.xxh64()
        cache_parts = []

        def hash_part(typ: str, part: str, val: str | bytes) -> None:
            try:
                if isinstance(val, str):
                    cache_parts.append((typ, part, val))
                    val = val.encode('ascii', 'backslashreplace')
                else:
                    cache_parts.append((typ, part, base64.b64encode(val).decode('ascii')))
            except Exception:
                self.node.logger.error("Unable to hash node: %s %s (value %r)" % (typ, part, val))
                raise
            h.update(val)

        hash_part('id', '', self.node.id)
        if self.metrics_hash is None:
            metrics_hash = b''
            for m in self.node.output_metrics.values():
                metrics_hash += m.calculate_hash()
            self.metrics_hash = metrics_hash
        hash_part('metrics', '', self.metrics_hash)

        if self.dim_hash is None:
            if self.node.output_dimensions:
                dim_hash = b''
                for dim in self.node.output_dimensions.values():
                    dim_hash += dim.calculate_hash()
                self.dim_hash = dim_hash
            else:
                self.dim_hash = b''
        hash_part('dimensions', '', cast(bytes, self.dim_hash))

        for node in self.node.input_nodes:
            hash_part('input node', node.id, node.hasher.calculate_hash(state=state))

        param_hash = ''
        for _, param in sorted(self.node.parameters.items(), key=lambda x: x[0]):
            ph = param.calculate_hash()
            param_hash += ph
            hash_part('param', param.local_id, ph)
        for param_id in self.node.global_parameters:
            gp = self.node.context.get_parameter(param_id, required=False)
            if gp is not None:
                ph = gp.calculate_hash()
                param_hash += ph
                hash_part('global param', gp.global_id, ph)
        self.param_hash = hashlib.md5(param_hash.encode('ascii'), usedforsecurity=False).digest()

        for ds in self.node.input_dataset_instances:
            if isinstance(ds, DVCDataset):
                state.upstream_dataset_keys.setdefault(self.node.id, set()).add(ds.get_cache_key(self.node.context))
            hash_part('dataset', ds.id, ds.calculate_hash(self.node.context))

        if self.mtime_hash is None:
            self.mtime_hash = self.get_class_hash(type(self), state=state)
        hash_part('mtime', '', self.mtime_hash)

        ret = h.digest()
        self.last_hash = ret
        self.last_hash_time = time_ns()
        self.prev_hash_parts = getattr(self, 'last_hash_parts', None)
        self.last_hash_parts = cache_parts
        return ret

    def mark_modified(self) -> None:
        self.modified_at = time_ns()
        if self.param_hash is None:
            return
        self.param_hash = None
        for node in self.node.output_nodes:
            node.hasher.mark_modified()

    @staticmethod
    def _get_cache_key(node: Node, node_hash: bytes) -> str:
        return 'node:%s:%s' % (node.id, node_hash.hex())

    @classmethod
    def prefetch_nodes(cls, context: Context, nodes: list[Node]) -> None:
        state = HashingState()
        node_count = len(nodes)
        with sentry_sdk.start_span(op='model.hash', name='Hashing %d nodes' % node_count):
            for node in nodes:
                node.hasher.calculate_hash(state=state)
        with sentry_sdk.start_span(op='model.prefetch', name='Prefetching %d nodes' % node_count):
            cls._prefetch_from_state(context=context, state=state)

    @classmethod
    def _prefetch_from_state(cls, context: Context, state: HashingState) -> None:
        cache = context.cache
        keys_by_node_id = {v: k for k, v in state.upstream_node_keys.items()}
        node_keys = set(keys_by_node_id.keys())
        found_node_ids = cache.prefetch_keys(keys=node_keys, caller_ids=keys_by_node_id)
        all_dataset_keys: set[str] = set()
        for node_id, dataset_keys in state.upstream_dataset_keys.items():
            if node_id in found_node_ids:
                continue
            all_dataset_keys.update(dataset_keys)
        cache.prefetch_keys(keys=all_dataset_keys)


    def get_cached_output(self) -> CacheResult[PathsDataFrame]:
        state = HashingState()
        cache = self.node.context.cache

        node_hash = self.calculate_hash(state=state)
        cache_key = self._get_cache_key(self.node, node_hash)
        cache_res = cache.get(cache_key)
        if not cache_res.is_hit:
            self._prefetch_from_state(context=self.node.context, state=state)
            if DEBUG_CACHE_MISSES:
                self.node.logger.debug("Cache miss for node %s" % self.node.id)

        out = cache_res.obj

        if DEBUG_CACHE_MISSES and out is None and self.prev_hash_parts:
            # ruff: noqa: T203

            print(self.node.id)
            if len(self.prev_hash_parts) != len(self.last_hash_parts):
                print('Length mismatch!!')
                pprint(self.prev_hash_parts)
                pprint(self.last_hash_parts)
                pprint('\n\n')
            for old, new in zip(self.prev_hash_parts, self.last_hash_parts, strict=False):
                if old[0] == 'input node' and new[0] == 'input node' and old[1] == new[1]:
                    continue
                if old != new:
                    pprint('\told: %s\n\tnew: %s' % (old, new))
        return cache_res

    def cache_output(self, res: CacheResult, df: PathsDataFrame):
        self.node.context.cache.set(res.key, df.copy())
