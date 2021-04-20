from nodes.actions.base import Action
from nodes.scenario import Scenario
from typing import Dict
import os
import importlib
import dvc_pandas
import yaml
from dvc_pandas import pull_datasets
from . import Dataset, Context
from pages.base import Page


class InstanceLoader:
    pages: Dict[str, Page]

    def make_node(self, node_class, config):
        ds_config = config.get('input_datasets', [])
        datasets = []
        for ds in ds_config:
            o = Dataset(id=ds['id'], column=ds.get('column'))
            datasets.append(o)
        node = node_class(self.context, config['id'], input_datasets=datasets)
        node.config = config
        return node

    def setup_nodes(self):
        for nc in self.config['nodes']:
            klass = nc['type'].split('.')
            node_name = klass.pop(-1)
            klass.insert(0, 'nodes')
            mod = importlib.import_module('.'.join(klass))
            node_class = getattr(mod, node_name)
            node = self.make_node(node_class, nc)
            self.context.add_node(node)

    def setup_actions(self):
        for nc in self.config['actions']:
            klass = nc['type'].split('.')
            node_name = klass.pop(-1)
            klass.insert(0, 'nodes')
            klass.insert(1, 'actions')
            mod = importlib.import_module('.'.join(klass))
            node_class = getattr(mod, node_name)
            node = self.make_node(node_class, nc)
            self.context.add_node(node)
            node.param_defaults['enabled'] = False

    def setup_edges(self):
        # Setup edges
        for node in self.context.nodes.values():
            for out_id in node.config.get('output_nodes', []):
                out_node = self.context.get_node(out_id)
                out_node.input_nodes.append(node)
                node.output_nodes.append(out_node)
                # FIXME: Check for cycles?

    def setup_scenarios(self):
        for sc in self.config['scenarios']:
            actions = sc.pop('actions', [])
            scenario = Scenario(**sc)
            for act_conf in actions:
                node = self.context.get_node(act_conf.pop('id'))
                assert isinstance(node, Action)
                scenario.actions.append([node, act_conf])
            self.context.add_scenario(scenario)

    def print_graph(self, node=None, indent=0):
        if node is None:
            all_nodes = self.context.nodes.values()
            root_nodes = list(filter(lambda node: not node.output_nodes, all_nodes))
            assert len(root_nodes) == 1
            node = root_nodes[0]

        print('  ' * indent + str(node))
        for in_node in node.input_nodes:
            self.print_graph(in_node, indent + 1)

    def load_datasets(self, datasets):
        for ds in datasets:
            self.context.add_dataset(ds)

    def setup_pages(self):
        self.pages = {}

        for pc in self.config['pages']:
            assert pc['id'] not in self.pages
            page = Page(**pc)
            cards = pc.get('cards', [])
            page.add_cards(cards, self.context)
            self.pages[pc['id']] = page

    def __init__(self, fn):
        data = yaml.load(open(fn, 'r'), Loader=yaml.Loader)
        self.context = Context()
        self.config = data['instance']
        self.context.target_year = self.config['target_year']

        os.environ['DVC_PANDAS_REPOSITORY'] = self.config['dataset_repo']
        if False:
            dvc_pandas.pull_datasets()
        self.load_datasets(self.config.get('datasets', []))

        self.setup_nodes()
        self.setup_actions()
        self.setup_edges()
        self.setup_scenarios()
        self.setup_pages()
