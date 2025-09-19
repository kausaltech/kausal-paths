from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from django.core.management.base import BaseCommand, CommandParser

import loguru
from recursive_diff import recursive_diff
from rich import print
from rich.console import Console
from rich.traceback import Traceback

from kausal_common.logging.warnings import register_warning_handler

from nodes.datasets import JSONDataset
from nodes.exceptions import NodeError
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from nodes.actions.action import ImpactOverview
    from nodes.context import Context

logger = loguru.logger.opt(colors=True)

console = Console()


def make_comparable(table: dict):
    schema: dict = table['schema']
    pks: list = schema['primaryKey']
    pks.sort()
    fields: list = schema['fields']

    fields.sort(key=lambda x: x['name'])
    data: list = table['data']
    data.sort(key=lambda x: tuple([x[key] for key in pks]))
    return data


class Command(BaseCommand):
    help = 'Validate computation models and store/compare results'

    store: Path | None
    compare: Path | None

    def add_arguments(self, parser: CommandParser):
        parser.add_argument('instances', metavar='INSTANCE_ID', type=str, nargs='*')
        parser.add_argument('--skip', dest='skip', metavar='INSTANCE_ID', action='append')
        parser.add_argument('--store', dest='store', metavar='DIR', type=str, action='store')
        parser.add_argument('--compare', dest='compare', metavar='DIR', type=str, action='store')
        parser.add_argument('--start-from', dest='start_from', metavar='INSTANCE_ID', action='store')

    def run_nodes(self, ctx: Context):
        for node in ctx.nodes.values():
            logger.info('Checking node {node}', node=node.id)
            try:
                node.check()
            except NodeError as e:
                if e.__cause__:
                    err = e.__cause__
                    tb = Traceback.from_exception(type(err), err, err.__traceback__)
                    console.print(tb)
                    logger.error(
                        'Error in instance {instance}\nNode dependency path: {dep_path}',
                        instance=ctx.instance.id,
                        dep_path=e.get_dependency_path(),
                    )
                    exit(1)
                raise

            if self.store:
                df = node.get_output_pl()
                out = JSONDataset.serialize_df(df)
                fn = self.store / ('%s-%s.json' % (ctx.instance.id, node.id))
                logger.info('Storing output to %s' % fn)
                with fn.open('w') as f:
                    json.dump(out, f, indent=2)
            elif self.compare:
                fn = self.compare / ('%s-%s.json' % (ctx.instance.id, node.id))
                if not fn.exists():
                    logger.warning('Skipping compare for %s' % fn)
                    continue
                logger.info('Comparing output to %s' % fn)
                with fn.open('r') as f:
                    data = json.load(f)
                df = node.get_output_pl()
                df_ser = JSONDataset.serialize_df(df)
                diffs = list(recursive_diff(make_comparable(data), make_comparable(df_ser)))
                if diffs:
                    logger.error('Instance %s, node %s differs' % (ctx.instance.id, node.id))
                    print(diffs)

    def run_aep(self, ctx: Context, aep: ImpactOverview):
        cn = aep.cost_node
        if cn:
            cost_id = cn.id
        else:
            cost_id = 'None'
        logger.info('Calculating impact overview: %s:%s' % (cost_id, aep.effect_node.id))

    def check_instance(self, ic: InstanceConfig):
        instance = ic.get_instance()
        ctx = instance.context
        logger.info('Loading datasets')
        ctx.load_all_dvc_datasets()
        with ctx.run():
            self.run_nodes(ctx)
            for aep in ctx.impact_overviews:
                self.run_aep(ctx, aep)

    def handle(self, *args, **options):
        instance_ids = options['instances']
        register_warning_handler()
        if not instance_ids:
            instance_ids = list(InstanceConfig.objects.all().order_by('identifier').values_list('identifier', flat=True))
        if options['store']:
            self.store = Path(options['store'])
            self.store.mkdir(exist_ok=True)
        else:
            self.store = None

        self.compare = Path(options['compare']) if options['compare'] else None

        start_from = options['start_from']
        for iid in instance_ids:
            if start_from:
                if iid == start_from:
                    start_from = None
                else:
                    continue
            if options['skip'] and iid in options['skip']:
                continue
            ic = InstanceConfig.objects.get(identifier=iid)
            if ic.has_framework_config():
                continue
            logger.info('Checking instance %s' % iid)
            self.check_instance(ic)
