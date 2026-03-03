from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand

import aiohttp
import loguru
from recursive_diff import recursive_diff
from rich import print

if TYPE_CHECKING:
    from django.core.management.base import CommandParser

logger = loguru.logger.opt(colors=True)


class Command(BaseCommand):
    help = 'Replay GraphQL request-response logs and validate responses'

    dir: Path
    maxfail: int
    failures: int = 0

    def add_arguments(self, parser: CommandParser):
        parser.add_argument('--dir', '-d', metavar='DIR', type=str, action='store', default='./query-store')
        parser.add_argument('--url', '-u', metavar='URL', type=str, action='store', default='http://127.0.0.1:8000')
        parser.add_argument('--maxfail', metavar='NUM', type=int, action='store', default=1)

    async def replay_query(self, session: aiohttp.ClientSession, fn: Path, data: dict[str, Any]):
        logger.info("Running query %s" % fn)
        query = data['query']
        variables = data['variables']
        headers = {hdr: val for hdr, val in data['headers'].items() if val}
        async with session.post('/v1/graphql/', json=dict(
            query=query,
            variables=variables,
            operationName=data['operation_name'],
        ), headers=headers) as resp:
            out = await resp.json()
            diff = list(recursive_diff(data['response'], out))
            if diff:
                self.failures += 1
                for d in diff:
                    print(d)
                if self.failures >= self.maxfail:
                    exit(1)


    async def replay_queries(self, fns: list[Path]):
        async with aiohttp.ClientSession(base_url=self.url) as session:
            for fn in fns:
                with fn.open('r') as f:
                    data = json.load(f)
                    await self.replay_query(session, fn, data)

    def handle(self, *args, **options):
        self.dir = Path(options['dir'])
        files = list(self.dir.glob('*.json'))
        files.sort()
        self.url = options['url']
        self.maxfail = options['maxfail']
        asyncio.run(self.replay_queries(files))
