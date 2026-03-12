from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError

from deepdiff import DeepDiff
from deepdiff.helper import CannotCompare
from redis import Redis
from rich.pretty import pprint

if TYPE_CHECKING:
    from django.core.management.base import CommandParser

    from deepdiff.model import DiffLevel

GQL_URL = '/v1/graphql/'


def compare_func(x: Any, y: Any, _level: DiffLevel | None):
    if not isinstance(x, dict) or not isinstance(y, dict):
        raise CannotCompare

    if '__typename' in x and '__typename' in y and x['__typename'] == y['__typename']:
        if x['__typename'] == 'ActionImpact':
            try:
                return x['action']['id'] == y['action']['id']
            except Exception:
                raise CannotCompare() from None
        elif x['__typename'] == 'YearlyValue':
            return x['year'] == y['year']
    try:
        return x['id'] == y['id']
    except Exception:
        raise CannotCompare() from None


def _diff_responses(fn: Path, data: dict[str, Any], out: dict[str, Any]) -> bool:
    """Return True if there are differences."""
    resp_errors = out.get('errors', [])
    target_errors = data['response'].get('errors', [])
    if resp_errors and not target_errors:
        print('Errors in response:')
        for err in resp_errors:
            print(err)
        return True

    target_data = data['response']['data']
    resp_data = out['data']
    if isinstance(target_data, dict) and isinstance(resp_data, dict) and len(target_data) == len(resp_data) == 1:
        solo_key = next(iter(target_data.keys()))
        target_data = target_data[solo_key]
        resp_data = resp_data[solo_key]

    diff = DeepDiff(target_data, resp_data, math_epsilon=1e-6, iterable_compare_func=compare_func)
    if not diff:
        return False

    print('Differences in response for query %s:' % fn)
    pprint(diff, max_depth=4)
    return True


class Command(BaseCommand):
    help = 'Replay GraphQL request-response logs and validate responses'

    dir: Path
    maxfail: int
    failures: int = 0
    check_perf: bool = False
    redis: Redis | None = None
    keep_cache: bool = False

    def add_arguments(self, parser: CommandParser):
        parser.add_argument('files', metavar='FILE', type=str, nargs='*')
        parser.add_argument('--dir', '-d', metavar='DIR', type=str, action='store', default='./query-store')
        parser.add_argument(
            '--url',
            '-u',
            metavar='URL',
            type=str,
            action='store',
            default=None,
            help='Run against a live HTTP server instead of in-process (e.g. http://127.0.0.1:8000)',
        )
        parser.add_argument('--maxfail', metavar='NUM', type=int, action='store', default=1)
        parser.add_argument('--start-from', metavar='NUM', type=int, action='store', default=0)
        parser.add_argument('--limit', metavar='NUM', type=int, action='store', default=0)
        parser.add_argument(
            '--check-perf', action='store_true', default=False, help='Check query execution times for performance regressions'
        )
        parser.add_argument(
            '--keep-cache', action='store_true', default=False, help='Do not flush the external cache before running'
        )

    def fail(self) -> None:
        self.failures += 1
        if self.failures >= self.maxfail:
            print('Maximum number of failures reached, stopping')
            exit(1)

    def evaluate_perf(self, start_time: float, end_time: float, data: dict[str, Any], fn: Path) -> None:
        if not self.check_perf:
            return
        exec_time = (end_time - start_time) * 1000
        reference_exec_time = data['execution_time']
        time_diff = exec_time - reference_exec_time
        if exec_time > reference_exec_time * 1.1 and time_diff > 500:
            print(
                f'Execution time {exec_time:.0f}ms is more than 10% longer than '
                + f'reference {reference_exec_time:.0f}ms for query {fn}'
            )
            self.fail()

    # ------------------------------------------------------------------
    # In-process mode (default)
    # ------------------------------------------------------------------

    def _replay_inprocess(self, fns: list[Path]) -> None:
        from django.test import Client

        client = Client()
        count = 0

        for i, fn in enumerate(fns):
            if i + 1 < self.start_from:
                continue
            print('[%d/%d] %s' % (i + 1, len(fns), fn))

            with fn.open('r') as f:
                data = json.load(f)

            has_session = data.get('has_session', False)
            if not has_session:
                client.cookies.clear()

            query = data['query']
            variables = data.get('variables') or {}
            headers = {hdr: val for hdr, val in (data.get('headers') or {}).items() if val}

            kwargs: dict[str, Any] = {}
            for key, val in headers.items():
                kwargs['HTTP_' + key.upper().replace('-', '_')] = val

            body = json.dumps({'query': query, 'variables': variables, 'operationName': data.get('operation_name')})
            start_time = time.time()
            resp = client.post(GQL_URL, body, content_type='application/json', **kwargs)
            end_time = time.time()
            out = json.loads(resp.content)
            if 'errors' in out:
                print('Errors in response:')
                for err in out['errors']:
                    print(err)
                exit(1)

            if _diff_responses(fn, data, out):
                print(f'Differences in response for query {fn}')
                self.fail()

            self.evaluate_perf(start_time, end_time, data, fn)

            count += 1
            if self.limit and count >= self.limit:
                break

    def flush_external_cache(self) -> None:
        if not self.redis or self.keep_cache:
            return
        self.redis.flushdb()

    # ------------------------------------------------------------------
    # HTTP mode (--url)
    # ------------------------------------------------------------------

    async def _replay_http_one(self, session: Any, fn: Path, data: dict[str, Any]) -> None:
        query = data['query']
        variables = data['variables']
        headers = {hdr: val for hdr, val in data['headers'].items() if val}
        async with session.post(
            GQL_URL,
            json=dict(query=query, variables=variables, operationName=data['operation_name']),
            headers=headers,
        ) as resp:
            out = await resp.json()
            if resp.cookies:
                session.cookie_jar.update_cookies(resp.cookies)

        if _diff_responses(fn, data, out):
            self.failures += 1
            if self.failures >= self.maxfail:
                exit(1)

    async def _replay_http(self, fns: list[Path]) -> None:
        import aiohttp

        count = 0
        async with aiohttp.ClientSession(base_url=self.url) as session:
            for i, fn in enumerate(fns):
                if i + 1 < self.start_from:
                    continue
                print('[%d/%d] %s' % (i + 1, len(fns), fn))
                with fn.open('r') as f:
                    data = json.load(f)
                has_session = data.get('has_session', False)
                if not has_session:
                    session.cookie_jar.clear()
                await self._replay_http_one(session, fn, data)
                count += 1
                if self.limit and count >= self.limit:
                    break

    # ------------------------------------------------------------------

    def _get_input_files(self, paths: list[str]) -> list[Path]:
        if paths:
            files = [Path(path) for path in paths]
            for path in files:
                if not path.exists():
                    raise CommandError(f'Input file does not exist: {path}')
                if not path.is_file():
                    raise CommandError(f'Input path is not a file: {path}')
            return files

        files = list(self.dir.glob('*.json'))
        files.sort()
        return files

    def handle(self, *args, **options):
        self.dir = Path(options['dir'])
        files = self._get_input_files(options['files'])
        self.url = options['url']
        self.maxfail = options['maxfail']
        self.start_from = options['start_from']
        self.limit = options['limit']
        self.check_perf = options['check_perf']
        self.keep_cache = options['keep_cache']

        os.environ['DISABLE_GRAPHQL_CACHE'] = '1'

        if redis_url := os.getenv('REDIS_URL'):
            self.redis = Redis.from_url(redis_url)
            self.flush_external_cache()
        else:
            self.redis = None
        if self.url:
            asyncio.run(self._replay_http(files))
        else:
            self._replay_inprocess(files)

        if self.failures:
            exit(1)
