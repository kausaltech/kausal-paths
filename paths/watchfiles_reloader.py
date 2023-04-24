from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional, Sequence, Union

import watchfiles
from django.utils import autoreload


class DjangoPythonFilter(watchfiles.PythonFilter):
    def __init__(self, *, ignore_paths: Optional[Sequence[Union[str, Path]]] = None, extra_extensions: Sequence[str] = ()) -> None:
        if 'site-packages' in self.ignore_dirs:
            # We want to watch site-packages, too
            d = list(self.ignore_dirs)
            d.remove('site-packages')
            self.ignore_dirs = d
        super().__init__(ignore_paths=ignore_paths, extra_extensions=extra_extensions)


class WatchfilesReloader(autoreload.BaseReloader):
    def watched_roots(self, watched_files: list[Path]) -> frozenset[Path]:
        extra_directories = self.directory_globs.keys()
        watched_file_dirs = {f.parent for f in watched_files}
        sys_paths = set([path for path in autoreload.sys_path_directories() if 'site-packages' in str(path)])
        return frozenset((*extra_directories, *watched_file_dirs, *sys_paths))

    def tick(self) -> Generator[None, None, None]:
        watched_files = list(self.watched_files(include_globs=False))
        watched_roots = self.watched_roots(watched_files)
        roots = autoreload.common_roots(watched_roots)
        watcher = watchfiles.watch(*roots, watch_filter=DjangoPythonFilter(extra_extensions=('.yaml')))
        for file_changes in watcher:
            for _, path in file_changes:
                self.notify_file_changed(Path(path))
            yield


def replaced_get_reloader() -> autoreload.BaseReloader:
    return WatchfilesReloader()


def replace_reloader():
    autoreload.get_reloader = replaced_get_reloader
