import logging
from datetime import datetime, UTC
from logging import LogRecord, StreamHandler
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

from logfmter.formatter import Logfmter
from rich.console import ConsoleRenderable
from rich.containers import Renderables
from rich.logging import RichHandler
from rich.text import Text, TextType
from rich.traceback import Traceback

if TYPE_CHECKING:
    from rich.console import Console, RenderableType

FormatTimeCallable = Callable[[datetime], Text]


class LogRender:
    def __init__(
        self,
        show_time: bool = True,
        show_level: bool = False,
        show_path: bool = True,
        time_format: Union[str, FormatTimeCallable] = "[%x %X]",
        omit_repeated_times: bool = True,
        level_width: Optional[int] = 8,
    ) -> None:
        self.show_time = show_time
        self.show_level = show_level
        self.show_path = show_path
        self.time_format = time_format
        self.omit_repeated_times = omit_repeated_times
        self.level_width = level_width
        self._last_time: Optional[Text] = None

    def __call__(
        self,
        console: "Console",
        renderables: Sequence["ConsoleRenderable"],
        name: str,
        log_time: Optional[datetime] = None,
        time_format: Optional[Union[str, FormatTimeCallable]] = None,
        level: TextType = "",
        path: Optional[str] = None,
        line_no: Optional[int] = None,
        link_path: Optional[str] = None,
    ) -> Renderables:
        from rich.table import Table

        output = Table.grid(padding=(0, 1))
        output.expand = True
        if self.show_time:
            output.add_column(style="log.time")
        if self.show_level:
            output.add_column(style="log.level", width=self.level_width)
        output.add_column(ratio=1, style="log.message", overflow="fold")
        if self.show_path and path:
            output.add_column(style="log.path")
        row: List["RenderableType"] = []
        if self.show_time:
            log_time = log_time or console.get_datetime()
            time_format = time_format or self.time_format
            if callable(time_format):
                log_time_display = time_format(log_time)
            else:
                log_time_display = Text(log_time.strftime(time_format))
            if log_time_display == self._last_time and self.omit_repeated_times:
                row.append(Text(" " * len(log_time_display)))
            else:
                row.append(log_time_display)
                self._last_time = log_time_display
        if self.show_level:
            row.append(level)

        if len(renderables) == 1 and '\n' not in renderables[0].plain:
            row.append(Renderables(renderables))
            renderables = []
        else:
            row.append('')

        if self.show_path and path:
            path_text = Text()
            path_text.append(
                name, style=f"link file://{link_path}#{line_no}" if link_path else ""
            )
            row.append(path_text)

        output.add_row(*row)

        return Renderables([output] + renderables)  # type: ignore


ISO_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'


class LogHandler(RichHandler):
    _log_render: LogRender  # type: ignore[assignment]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lr = self._log_render
        self._log_render = LogRender(
            show_time=lr.show_time,
            show_level=lr.show_level,
            show_path=lr.show_path,
            time_format=ISO_FORMAT,
            omit_repeated_times=lr.omit_repeated_times,
            level_width=None,
        )

    def render_message(self, record: LogRecord, message: str) -> ConsoleRenderable:
        extra: dict[str, Any] = getattr(record, 'extra', {})
        tenant_id = extra.get('tenant', None)
        if tenant_id:
            message = '[%s] %s' % (extra['tenant'], message)
        if 'session' in extra:
            message = '<%s> %s' % (extra['session'], message)
        ret = super().render_message(record, message)
        return ret

    def render(
        self,
        *,
        record: LogRecord,
        traceback: Optional[Traceback],
        message_renderable: "ConsoleRenderable",
    ) -> "ConsoleRenderable":
        """Render log for display.

        Args:
            record (LogRecord): logging Record.
            traceback (Optional[Traceback]): Traceback instance or None for no Traceback.
            message_renderable (ConsoleRenderable): Renderable (typically Text) containing log message contents.

        Returns:
            ConsoleRenderable: Renderable to display log.
        """
        path = Path(record.pathname).name
        level = self.get_level_text(record)
        time_format = None if self.formatter is None else self.formatter.datefmt
        log_time = datetime.fromtimestamp(record.created)

        log_renderable = self._log_render(
            self.console,
            [message_renderable] if not traceback else [message_renderable, traceback],
            name=record.name,
            log_time=log_time,
            time_format=time_format,
            level=level,
            path=path,
            line_no=record.lineno,
            link_path=record.pathname if self.enable_link_path else None,
        )
        return log_renderable


class LogFmtFormatter(Logfmter):
    def __init__(self):
        keys = ['time', 'level']
        mapping = {
            'time': 'asctime',
            'level': 'levelname',
        }
        super().__init__(keys=keys, mapping=mapping, datefmt=ISO_FORMAT)

    @classmethod
    def get_extra(cls, record: logging.LogRecord) -> dict:
        ret = super().get_extra(record)
        if 'taskName' in ret:
            del ret['taskName']
        if 'extra' in ret:
            del ret['extra']
        extra = getattr(record, 'extra', {})
        for key, val in extra.items():
            if key in ret:
                continue
            ret[key] = val
        return ret

    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created, UTC).strftime(ISO_FORMAT)


class LogFmtHandlerError(StreamHandler):
    def __init__(self, stream=None):
        if stream is None:
            stream = sys.stderr
        super().__init__(stream)
        self.formatter = LogFmtFormatter()
        self.addFilter(lambda rec: rec.levelno <= logging.INFO)


class LogFmtHandlerInfo(StreamHandler):
    def __init__(self, stream=None):
        if stream is None:
            stream = sys.stdout
        super().__init__(stream)
        self.formatter = LogFmtFormatter()
        self.addFilter(lambda rec: rec.levelno > logging.INFO)


class UwsgiReqLogHandler(StreamHandler):
    def __init__(self, stream=None):
        if stream is None:
            stream = sys.stdout
        super().__init__(stream)

    def format(self, record: LogRecord) -> str:
        s = str(record.msg).rstrip('\n')
        return s

    def emit(self, record: LogRecord) -> None:
        # Only emit health check logs only for 5 mins after starting
        if ' path=/healthz' in record.msg:
            if record.relativeCreated > 5 * 60 * 1000:
                return
        print(self.format(record))
