from datetime import datetime
from pathlib import Path
from logging import LogRecord
from typing import Any, Iterable, List, Optional, TYPE_CHECKING, Sequence, Union, Callable
from rich.console import ConsoleRenderable

from rich.logging import RichHandler
from rich.text import Text, TextType
from rich.traceback import Traceback
from rich.containers import Renderables


if TYPE_CHECKING:
    from rich.console import Console, ConsoleRenderable, RenderableType

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


class LogHandler(RichHandler):
    _log_render: LogRender  # type: ignore[assignment]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lr = self._log_render

        self._log_render = LogRender(
            show_time=lr.show_time,
            show_level=lr.show_level,
            show_path=lr.show_path,
            # time_format=lr.time_format,
            time_format='%Y-%m-%d %H:%M:%S.%f',
            omit_repeated_times=lr.omit_repeated_times,
            level_width=None,
        )

    def render_message(self, record: LogRecord, message: str) -> ConsoleRenderable:
        extra: dict[str, Any] = getattr(record, 'extra', {})
        markup = extra.pop('markup', False)
        if markup:
            setattr(record, 'markup', True)
            def with_style(style: str, msg: str):
                return '[%s]%s[/]' % (style, msg)
        else:
            def with_style(style: str, msg: str):
                return msg
        scope_parts = []
        instance_id = extra.get('instance')
        if instance_id:
            scope_parts.append(with_style('scope.key', instance_id))
        instance_obj_id = extra.get('instance_obj_id')
        if instance_obj_id:
            scope_parts.append(with_style('log.path', instance_obj_id))

        ctx_id = extra.get('context')
        if ctx_id:
            scope_parts.append('[scope.key.special]%s[/]' % ctx_id)

        session_id = extra.get('session')
        if session_id:
            scope_parts.append('sess [json.key]%s[/]' % session_id)

        if scope_parts:
            record.highlighter = None
            message = r'[log.path]\[[/]%s[log.path]][/] %s' % (':'.join(scope_parts), message)
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


def configure_logging():
    from loguru import logger
    logger.configure(
        handlers=[
            dict(sink=LogHandler(), format="{message}"),
        ],
        extra={'markup': True}
    )
