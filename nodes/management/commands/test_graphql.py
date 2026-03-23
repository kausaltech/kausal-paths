from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand

import aiohttp
import loguru
from deepdiff import DeepDiff
from deepdiff.helper import CannotCompare
from rich.pretty import pprint

if TYPE_CHECKING:
    from django.core.management.base import CommandParser

    from deepdiff.model import DiffLevel

logger = loguru.logger.opt(colors=True)


def compare_func(x: Any, y: Any, _level: DiffLevel | None = None):
    if not isinstance(x, dict) or not isinstance(y, dict):
        raise CannotCompare
    try:
        return x['id'] == y['id']
    except Exception:
        raise CannotCompare() from None


class Command(BaseCommand):
    help = 'Replay GraphQL request-response logs and validate responses'

    dir: Path
    maxfail: int
    failures: int = 0

    def add_arguments(self, parser: CommandParser):
        parser.add_argument('--dir', '-d', metavar='DIR', type=str, action='store', default='./query-store')
        parser.add_argument('--url', '-u', metavar='URL', type=str, action='store', default='http://127.0.0.1:8000')
        parser.add_argument('--maxfail', metavar='NUM', type=int, action='store', default=1)
        parser.add_argument('--start-from', metavar='NUM', type=int, action='store', default=0)

    async def replay_query(self, session: aiohttp.ClientSession, fn: Path, data: dict[str, Any]):
        query = data['query']
        variables = data['variables']
        headers = {hdr: val for hdr, val in data['headers'].items() if val}
        async with session.post(
            '/v1/graphql/',
            json=dict(
                query=query,
                variables=variables,
                operationName=data['operation_name'],
            ),
            headers=headers,
        ) as resp:
            out = await resp.json()
            if resp.cookies:
                session.cookie_jar.update_cookies(resp.cookies)
            resp_errors = out.get('errors', [])
            target_errors = data['response'].get('errors', [])
            if resp_errors and not target_errors:
                print('Errors in response:')
                for err in resp_errors:
                    print(err)
                exit(1)
            target_data = data['response']['data']
            resp_data = out['data']
            if isinstance(target_data, dict) and isinstance(resp_data, dict) and len(target_data) == len(resp_data) == 1:
                solo_key = next(iter(target_data.keys()))
                target_data = target_data[solo_key]
                resp_data = resp_data[solo_key]
            diff = DeepDiff(target_data, resp_data, math_epsilon=1e-6, iterable_compare_func=compare_func)
            if not diff:
                return
            self.failures += 1
            print('Differences in response for query %s:' % fn)
            # print('Expected response:')
            # print(json.dumps(data['response'], indent=2))
            # print('Actual response:')
            # print(json.dumps(obj=out, indent=2))
            # print('Differences:')
            pprint(diff, max_depth=4)
            if self.failures >= self.maxfail:
                exit(1)

    async def replay_queries(self, fns: list[Path]):
        async with aiohttp.ClientSession(base_url=self.url) as session:
            for i, fn in enumerate(fns):
                if i + 1 < self.start_from:
                    continue
                logger.info('[%d/%d] running query %s' % (i + 1, len(fns), fn))
                with fn.open('r') as f:
                    data = json.load(f)
                    has_session = data['has_session']
                    if not has_session:
                        session.cookie_jar.clear()
                    await self.replay_query(session, fn, data)

    def handle(self, *args, **options):
        self.dir = Path(options['dir'])
        files = list(self.dir.glob('*.json'))
        files.sort()
        self.url = options['url']
        self.maxfail = options['maxfail']
        self.start_from = options['start_from']
        asyncio.run(self.replay_queries(files))
        if self.failures:
            exit(1)
