import contextlib
from types import TracebackType
import typing
from datetime import datetime
from dataclasses import dataclass, field

from rich.table import Table
from rich.console import Console
import sentry_sdk

if typing.TYPE_CHECKING:
    from .node import Node
    from .context import Context


@dataclass
class PerfStats:
    nr_calls: int = 0
    exec_time: float = 0
    cum_exec_time: float = 0
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass(slots=True)
class PerfStackEntry:
    node: 'Node'
    exec_time: float = 0
    cum_exec_time: float = 0
    cache_hits: int = 0
    cache_misses: int = 0
    first_start: datetime = field(init=False)
    start: datetime | None = field(init=False)

    def __post_init__(self):
        self.start = datetime.now()
        self.first_start = self.start

    def pause(self):
        now = datetime.now()
        assert self.start is not None
        self.exec_time += (now - self.start).total_seconds()
        self.start = None

    def resume(self):
        now = datetime.now()
        self.start = now

    def end(self):
        now = datetime.now()
        assert self.start is not None
        self.exec_time += (now - self.start).total_seconds()
        self.cum_exec_time = (now - self.first_start).total_seconds()


class PerfContext(contextlib.AbstractContextManager):
    stats_by_class: dict[typing.Type, PerfStats]
    node_stack: list[PerfStackEntry]
    enabled: bool = False
    context: 'Context'

    def __init__(self, context: 'Context'):
        self.node_stack = []
        self.context = context

    def __enter__(self) -> 'typing.Self':
        self.start()
        return self

    def __exit__(self, __exc_type: type[BaseException] | None, __exc_value: BaseException | None, __traceback: TracebackType | None) -> bool | None:
        if __exc_type:
            self.node_stack = []
        self.stop()
        return None

    def start(self):
        self.enabled = True
        if self.node_stack:
            print("Node stack was not empty")
            self.node_stack = []
        self.stats_by_class = dict()

    def stop(self):
        self.enabled = False
        old_stack = self.node_stack
        self.node_stack = []
        if old_stack:
            print(old_stack)
            self.context.instance.logger.error("Node stack was not empty")

    def node_start(self, node: 'Node'):
        if not self.enabled:
            return
        kls = type(node)
        if kls not in self.stats_by_class:
            self.stats_by_class[kls] = PerfStats()
        if self.node_stack:
            last_entry = self.node_stack[-1]
            last_entry.pause()
        self.node_stack.append(PerfStackEntry(node))

    def node_end(self, node: 'Node'):
        if not self.enabled:
            return

        entry = self.node_stack.pop()
        assert entry.node == node
        entry.end()

        st = self.stats_by_class[type(node)]
        st.exec_time += entry.exec_time
        st.cum_exec_time += entry.cum_exec_time
        st.cache_misses += entry.cache_misses
        st.cache_hits += entry.cache_hits
        st.nr_calls += 1

        if self.node_stack:
            last_entry = self.node_stack[-1]
            last_entry.resume()

    def record_cache(self, node: 'Node', is_hit: bool):
        if not self.enabled:
            return

        entry = self.node_stack[-1]
        assert entry.node == node
        if is_hit:
            entry.cache_hits += 1
        else:
            entry.cache_misses += 1

    def print(self):
        assert not self.node_stack

        table = Table(title="Execution time by class")
        table.add_column("Class", justify="left")
        table.add_column("Calls")
        table.add_column("Time (s)")
        table.add_column("Cumulative time (s)")
        table.add_column("Cache hits")
        table.add_column("Cache misses")

        kl_time = sorted(self.stats_by_class.items(), key=lambda x: x[1].exec_time, reverse=True)
        for kls, st in kl_time:
            mod_name = '%s.%s' % (kls.__module__, kls.__name__)
            table.add_row(
                mod_name,
                '%d' % st.nr_calls,
                '%.3f' % st.exec_time,
                '%.3f' % st.cum_exec_time,
                '%d' % st.cache_hits,
                '%d' % st.cache_misses
            )
        console = Console()
        console.print(table)
