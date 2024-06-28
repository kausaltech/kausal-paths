from __future__ import annotations

import json
import os

import loguru
from django.core.management.base import BaseCommand, CommandParser
from recursive_diff import recursive_diff
from rich import print

from nodes.datasets import JSONDataset
from nodes.models import InstanceConfig

logger = loguru.logger.opt(colors=True)


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

    def add_arguments(self, parser: CommandParser):
        parser.add_argument('instances', metavar='INSTANCE_ID', type=str, nargs='*')
        parser.add_argument('--skip', dest='skip', metavar='INSTANCE_ID', action='append')
        parser.add_argument('--store', dest='store', metavar='DIR', type=str, action='store')
        parser.add_argument('--compare', dest='compare', metavar='DIR', type=str, action='store')
        parser.add_argument('--start-from', dest='start_from', metavar='INSTANCE_ID', action='store')

    def check_instance(self, ic: InstanceConfig, store: str, compare: str):
        instance = ic.get_instance()
        ctx = instance.context
        logger.info('Loading datasets')
        ctx.load_all_dvc_datasets()
        with ctx.run():
            for node in ctx.nodes.values():
                logger.info("Checking node {}".format(node.id))
                node.check()
                if store:
                    df = node.get_output()
                    out = JSONDataset.serialize_df(df)
                    fn = os.path.join(store, '%s-%s.json' % (instance.id, node.id))
                    logger.info("Storing output to %s" % fn)
                    with open(fn, 'w') as f:
                        json.dump(out, f, indent=2)
                elif compare:
                    fn = os.path.join(compare, '%s-%s.json' % (instance.id, node.id))
                    try:
                        os.stat(fn)
                    except FileNotFoundError as e:
                        logger.warning("Skipping compare; %s" % str(e))
                        continue
                    logger.info("Comparing output to %s" % fn)
                    with open(fn, 'r', encoding='utf8') as f:
                        data = json.load(f)
                    df = node.get_output()
                    df_ser = JSONDataset.serialize_df(df)
                    diffs = list(recursive_diff(make_comparable(data), make_comparable(df_ser)))
                    if diffs:
                        logger.error("Instance %s, node %s differs" % (instance.id, node.id))
                        print(diffs)


    def handle(self, *args, **options):
        instance_ids = options['instances']
        if not instance_ids:
            instance_ids = list(InstanceConfig.objects.all().order_by('identifier').values_list('identifier', flat=True))
        if options['store']:
            os.makedirs(options['store'], exist_ok=True)
        start_from = options['start_from']
        for id in instance_ids:
            if options['skip'] and id in options['skip']:
                continue
            if start_from:
                if id == start_from:
                    start_from = None
                else:
                    continue
            ic = InstanceConfig.objects.get(identifier=id)
            logger.info("Checking instance {}", id)
            self.check_instance(ic, options['store'], options['compare'])
