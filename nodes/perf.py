from __future__ import annotations
import time
from rich import box
from rich.text import Text

import contextlib
from contextvars import ContextVar
from types import TracebackType
from typing import TYPE_CHECKING, Generator, Generic, Protocol, Self, TypeVar
from dataclasses import dataclass, field

from rich.table import Table
from rich.console import Console

if TYPE_CHECKING:
    from common.cache import CacheResult


DEBUG = False


@dataclass
class PerfStats:
    nr_calls: int = 0
    exec_time: float = 0
    cum_exec_time: float = 0
    cache_hits: int = 0
    cache_misses: int = 0


class HasId(Protocol):
    id: str


T = TypeVar('T', bound=HasId)

@dataclass(slots=True)
class PerfNodeEntry(Generic[T]):
    run: PerfRunContext[T]
    node: T
    parent: Self | None
    own_exec_time: int = field(init=False, default=0)
    total_exec_time: int = field(init=False, default=0)
    children: list[Self] = field(init=False, default_factory=list)
    last_entered_at: int = field(init=False)
    left_at: int = field(init=False)
    cache_res: CacheResult | None = field(init=False)
    depth: int = field(init=False)

    def __post_init__(self):
        self.last_entered_at = self.now()
        self.cache_res = None
        if not self.parent:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1
        self.debug('create  %s' % self.time_ms(self.last_entered_at))

    def now(self) -> int:
        return self.run.now()

    def time_ms(self, time: int) -> str:
        return '%.3f ms' % (time / 1000000.0)

    if DEBUG:
        def debug(self, msg: str):
            print('%-5s: %s%s: %s' % (self.time_ms(self.now()), '  ' * self.depth, self.node.id, msg))
    else:
        def debug(self, msg: str):
            pass

    def enter_child(self, child: Self):
        now = child.last_entered_at
        own = now - self.last_entered_at
        self.own_exec_time += own
        self.debug('enter child (own %s, own total %s)' % (self.time_ms(own), self.time_ms(self.own_exec_time)))
        self.children.append(child)

    def return_from_child(self, child_pse: Self):
        self.debug('return from child (own total before %s, child total %s)' % (
            self.time_ms(self.total_exec_time), self.time_ms(child_pse.total_exec_time),
        ))
        self.total_exec_time += child_pse.total_exec_time
        self.last_entered_at = child_pse.left_at

    def leave(self: Self):
        self.left_at = self.now()
        own_time = self.left_at - self.last_entered_at
        self.own_exec_time += own_time
        self.total_exec_time += self.own_exec_time
        self.debug('leave at %s (total %s, own %s, new own %s)' % (
            self.time_ms(self.left_at), self.time_ms(self.total_exec_time),
            self.time_ms(self.own_exec_time), self.time_ms(own_time),
        ))
        if self.parent is not None:
            self.parent.return_from_child(self)

    def mark_cache(self, cache_res: CacheResult):
        assert self.cache_res is None
        self.cache_res = cache_res


@dataclass(slots=True)
class PerfRunContext(Generic[T]):
    ctx: PerfContext[T]
    roots: list[PerfNodeEntry[T]] = field(init=False, default_factory=list)
    tip: PerfNodeEntry[T] | None = field(init=False, default=None)
    tip_depth: int = field(init=False, default=0)
    started_at: int = field(init=False, default_factory=time.perf_counter_ns)

    def __post_init__(self):
        pass

    def now(self) -> int:
        return time.perf_counter_ns() - self.started_at

    def enter(self, node: T) -> PerfNodeEntry[T]:
        pse = PerfNodeEntry(self, node, parent=self.tip)
        if self.tip is None:
            self.roots.append(pse)
        else:
            self.tip.enter_child(pse)
        self.tip = pse
        self.tip_depth += 1
        return pse

    def leave(self):
        cur = self.tip
        assert cur is not None
        self.tip_depth -= 1
        cur.leave()
        if cur.parent is None:
            assert cur == self.roots[-1]
        self.tip = cur.parent

    def _dump_recurse(self, siblings: list[PerfNodeEntry[T]], depth: int, table: Table) -> None:
        for pse in siblings:
            total_exec = pse.total_exec_time / 1000000.0
            if total_exec < self.ctx.min_ms:
                continue
            node_id: str = pse.node.id
            if self.ctx.supports_cache:
                cr = pse.cache_res
                if not cr:
                    cache_text = Text('')
                else:
                    cache_text = Text('HIT' if cr.is_hit else 'MISS', style=cr.color)
                cache_cols = [
                    Text(cr.kind.name, style=cr.kind.color) if cr else Text('disabled', style='grey42'),
                    cache_text,
                ]
            else:
                cache_cols = []
            def format_num(num: float) -> Text:
                return Text('%.2f' % num, style='italic')
            table.add_row(
                '%s%s' % ('  ' * depth, node_id),
                Text('  ' * depth) + format_num(total_exec),
                format_num(pse.own_exec_time / 1000000),
                *cache_cols,
            )
            self._dump_recurse(pse.children, depth + 1, table)

    def end(self, failed: bool):
        if not self.roots:
            return

        table = Table(box=box.SIMPLE, row_styles=['on gray3', 'on gray7'], title=self.ctx.description)
        table.add_column('Node', justify='left')
        table.add_column('Cum. time (ms)')
        table.add_column('Own time (ms)')
        if self.ctx.supports_cache:
            table.add_column('Cache')
            table.add_column('Cache kind')
        self._dump_recurse(self.roots, 0, table)
        console = Console()
        console.print(table)


class PerfContext(contextlib.AbstractContextManager, Generic[T]):
    run: PerfRunContext[T] | None
    enabled: bool = False
    min_ms: float
    description: str | None

    def __init__(self, supports_cache: bool, min_ms: float = 0.0, description: str | None = None):
        self.supports_cache = supports_cache
        self.min_ms = float(min_ms)
        self.description = description
        self.run = None

    def __enter__(self):
        run_ctx = PerfRunContext(self)
        assert self.run is None
        self.run = run_ctx
        return run_ctx

    def __exit__(self, __exc_type: type[BaseException] | None, __exc_value: BaseException | None, __traceback: TracebackType | None) -> bool | None:
        run = self.run
        if run is None:
            raise Exception("Exiting context with no previous run active")
        run.end(__exc_type is not None)
        self.run = None
        return None

    @contextlib.contextmanager
    def exec_node(self, node: T) -> Generator[None | PerfNodeEntry[T], None, None]:
        if not self.enabled:
            yield None
            return

        if self.run is None:
            yield None
            return

        yield self.run.enter(node)
        self.run.leave()

    def record_cache(self, node: T, is_hit: bool):
        if not self.enabled:
            return

        # TODO
        return
