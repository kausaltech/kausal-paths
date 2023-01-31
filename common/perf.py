import enum
import threading
import time
import inspect


pc_data = threading.local()


class PerfCounter:
    class Level(enum.Enum):
        INFO = 0
        DEBUG = 1
        VERBOSE_DEBUG = 2

    start: int
    last_value: int
    display_counter: int
    tag: str
    show_time_to_last: bool

    shown_level: int = Level.INFO.value

    @classmethod
    def change_level(cls, level: Level):
        cls.shown_level = level.value

    def __init__(self, tag: str | None = None, show_time_to_last: bool = False, level: Level = Level.INFO):
        self.level = level

        self.start = time.perf_counter_ns()
        self.last_value = self.start
        self.display_counter = 0

        if tag is None:
            # If no tag given, default to the name of the calling func
            frame = inspect.currentframe()
            assert frame is not None
            calling_frame = frame.f_back
            assert calling_frame is not None
            tag = calling_frame.f_code.co_name

        self.tag = tag

        if level.value <= PerfCounter.shown_level:
            if not hasattr(pc_data, 'depth'):
                pc_data.depth = 0
            pc_data.depth += 1

        self.show_time_to_last = show_time_to_last

    def __del__(self):
        if self.level.value > PerfCounter.shown_level:
            return

        if hasattr(pc_data, 'depth'):
            pc_data.depth -= 1

    def measure(self) -> float:
        now = time.perf_counter_ns()
        cur_ms = (now - self.last_value) / 1000000
        self.last_value = now
        return cur_ms

    def display(self, name=None, show_time_to_last=False):
        if self.level.value > PerfCounter.shown_level:
            return

        if not name:
            name = '%d' % self.display_counter
            self.display_counter += 1

        now = time.perf_counter_ns()
        cur_ms = (now - self.start) / 1000000
        tag_str = '[%s] ' % self.tag if self.tag else ''
        if self.show_time_to_last:
            diff_str = ' (to previous %4.1f ms)' % ((now - self.last_value) / 1000000)
        else:
            diff_str = ''
        print('%s%s%6.1f ms%s: %s' % ((pc_data.depth - 1) * '  ', tag_str, cur_ms, diff_str, name))
        self.last_value = now
