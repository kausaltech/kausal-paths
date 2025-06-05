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

from kausal_common.deployment import get_deployment_build_id

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
    """
    State container for tracking hashing information across multiple nodes.

    This class maintains various mappings used during the node hashing process
    to avoid redundant calculations and enable efficient cache key generation.
    """

    node_class_hash: dict[type, int] = field(default_factory=dict)
    """Mapping between node class types and their module modification times (mtime).

    Used to detect when source code has changed and invalidate cached hashes accordingly.
    """

    upstream_node_keys: dict[str, str] = field(default_factory=dict)
    """Mapping between node IDs and their calculated cache keys.

    Stores the final cache keys for nodes that have been processed, enabling
    efficient lookup and dependency tracking.
    """

    upstream_dataset_keys: dict[str, set[str]] = field(default_factory=dict)
    """Mapping between node IDs and the set of dataset cache keys they depend on.

    Tracks which datasets each node uses so that cache prefetching can be
    optimized and dataset changes can trigger appropriate cache invalidation.
    """

type NodeHasherFuncT[**P, R] = Callable[Concatenate[NodeHasher, P], R]


@dataclass
class NodeHasher:
    """
    Handles hash calculation and caching for individual nodes.

    This class is responsible for computing deterministic hashes that represent
    the complete state of a node, including its parameters, dependencies, and
    source code. These hashes are used for cache invalidation and to determine
    when cached results can be reused.
    """

    node: Node
    """The node instance this hasher is associated with."""

    # When was this node last changed (timestamp in ns)
    modified_at: int | None = field(init=False, default=None)
    """Timestamp (in nanoseconds) when this node was last modified."""

    last_hash: bytes | None = field(init=False, default=None)
    """The most recently calculated hash for this node."""

    last_hash_time: int | None = field(init=False, default=None)
    """Timestamp (in nanoseconds) when the last hash was calculated."""

    param_hash: bytes | None = field(init=False, default=None)
    """Cached hash of the node's parameters."""

    mtime_hash: bytes | None = field(init=False, default=None)
    """Cached hash based on source code modification times."""

    dim_hash: bytes | None = field(init=False, default=None)
    """Cached hash of the node's output dimensions."""

    metrics_hash: bytes | None = field(init=False, default=None)
    """Cached hash of the node's output metrics."""

    @staticmethod
    def _wrap_hashing_error[**P, R](func: NodeHasherFuncT[P, R]) -> NodeHasherFuncT[P, R]:
        """
        Provide better error reporting for hashing functions.

        Args:
            func: The function to wrap

        Returns:
            The wrapped function that catches exceptions and converts them to NodeHashingError

        """
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
        return cast('NodeHasherFuncT[P, R]', report_error)

    def _get_cached_hash(self) -> bytes | None:
        """
        Return the cached hash if it's still valid, None otherwise.

        A cached hash is considered valid if the node hasn't been modified
        since the hash was last calculated.

        Returns:
            The cached hash bytes if valid, None if the hash needs recalculation

        """
        if self.last_hash is None or self.last_hash_time is None:
            return None
        if self.modified_at is None or self.modified_at <= self.last_hash_time:
            return self.last_hash
        return None

    @classmethod
    def get_class_hash(cls, node_class: type, state: HashingState) -> bytes:
        """
        Calculate a hash based on the modification times of a class and its parent classes.

        This method traverses the method resolution order (MRO) of the given class
        and creates a hash based on the modification times of the source files
        containing each class definition.

        Args:
            node_class: The class to calculate the hash for
            state: HashingState object to cache computed values

        Returns:
            Hash bytes representing the combined modification times of all classes in the MRO

        """
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
        """
        Calculate the complete hash for this node and update the hashing state.

        This is the main entry point for hash calculation. It delegates to _calculate_hash
        for the actual computation and then updates the state with the resulting cache key.

        Args:
            state: HashingState object to track computed values and cache keys

        Returns:
            The calculated hash bytes for this node

        Raises:
            NodeHashingError: If hash calculation fails for any reason

        """
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
        """
        Check if this node's results are cached in the run cache and available for execution.

        Returns:
            True if cached results exist and can be used instead of running the node

        """
        cached_hash = self._get_cached_hash()
        if cached_hash is None:
            return False
        key = self._get_cache_key(self.node, cached_hash)
        cache = self.node.context.cache
        return cache.is_run_cached(key=key)

    def is_local_cached(self) -> bool:
        """
        Check if this node's results are cached in the local cache.

        Returns:
            True if cached results exist in the local cache

        """
        cached_hash = self._get_cached_hash()
        if cached_hash is None:
            return False
        key = self._get_cache_key(self.node, cached_hash)
        cache = self.node.context.cache
        return cache.is_local_cached(key=key)

    def is_cached(self, run: bool = True, local: bool = False) -> bool:
        """
        Check if this node's results are cached according to the specified criteria.

        Args:
            run: Whether to check for run cache availability
            local: Whether to check for local cache availability

        Returns:
            True if the node is cached according to the specified criteria

        Note:
            Currently this method always returns the result of is_run_cached()
            regardless of the parameters (marked as FIXME).

        """
        # FIXME
        return self.is_run_cached()

    def _calculate_hash(self, state: HashingState) -> bytes:  # noqa: C901, PLR0912, PLR0915
        """
        Calculate the complete hash for this node.

        This method computes a comprehensive hash that includes:
        - Build ID (deployment version)
        - Node ID
        - Output metrics and dimensions
        - Input node hashes
        - Parameter values (both local and global)
        - Input dataset hashes
        - Class modification times

        Args:
            state: HashingState object for caching intermediate results

        Returns:
            The calculated hash bytes

        Raises:
            Exception: If any part of the hashing process fails

        """
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

        if build_id := get_deployment_build_id():
            hash_part('build_id', '', build_id)

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
        hash_part('dimensions', '', self.dim_hash)

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
        """
        Mark this node as modified and propagate the modification to dependent nodes.

        This method sets the modification timestamp and invalidates cached hashes.
        It also recursively marks all output nodes (dependents) as modified to
        ensure proper cache invalidation throughout the dependency graph.
        """
        self.modified_at = time_ns()
        if self.param_hash is None:
            return
        self.param_hash = None
        for node in self.node.output_nodes:
            node.hasher.mark_modified()

    @staticmethod
    def _get_cache_key(node: Node, node_hash: bytes) -> str:
        """
        Generate a cache key from a node and its hash.

        Args:
            node: The node to generate a key for
            node_hash: The hash bytes for the node

        Returns:
            A string cache key in the format 'node:{node_id}:{hash_hex}'

        """
        return 'node:%s:%s' % (node.id, node_hash.hex())

    @classmethod
    def prefetch_nodes(cls, context: Context, nodes: list[Node]) -> None:
        """
        Prefetch cache data for multiple nodes efficiently.

        This method calculates hashes for all provided nodes and then prefetches
        their cache data in batch operations for improved performance.

        Args:
            context: The context containing cache configuration
            nodes: List of nodes to prefetch cache data for

        """
        state = HashingState()
        node_count = len(nodes)
        with sentry_sdk.start_span(op='model.hash', name='Hashing %d nodes' % node_count):
            for node in nodes:
                node.hasher.calculate_hash(state=state)
        with sentry_sdk.start_span(op='model.prefetch', name='Prefetching %d nodes' % node_count):
            cls._prefetch_from_state(context=context, state=state)

    @classmethod
    def _prefetch_from_state(cls, context: Context, state: HashingState) -> None:
        """
        Prefetch cache data based on computed hashing state.

        This method examines the hashing state to determine which cache keys
        need to be prefetched and performs the prefetch operations efficiently.

        Args:
            context: The context containing cache configuration
            state: HashingState containing computed cache keys and dependencies

        """
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
        """
        Retrieve cached output for this node if available.

        This method calculates the node's current hash and attempts to retrieve
        cached results. If no cached results are found, it triggers prefetching
        of related cache data and optionally logs debug information about cache misses.

        Returns:
            CacheResult containing the cached DataFrame if available, or a cache miss result

        """
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
        """
        Store the computed output in the cache.

        Args:
            res: The CacheResult object containing the cache key and metadata
            df: The PathsDataFrame to store in the cache

        """
        self.node.context.cache.set(res.key, df.copy())
