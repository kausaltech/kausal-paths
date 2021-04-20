from dataclasses import dataclass
from nodes.node import Node
from nodes.actions.base import Action
from nodes.scenario import Scenario
from typing import Dict
import os
import importlib
import dvc_pandas
import yaml
from dvc_pandas import pull_datasets
from . import Dataset, Context
from pages.base import EmissionPage, Page


@dataclass
class Instance:
    id: str
    name: str


class InstanceLoader:
    pages: Dict[str, Page]
    instance: Instance

    def make_node(self, node_class, config) -> Node:
        ds_config = config.get('input_datasets', [])
        datasets = []
        for ds in ds_config:
            o = Dataset(id=ds['id'], column=ds.get('column'))
            datasets.append(o)
        node = node_class(self.context, config['id'], input_datasets=datasets)
        node.name = config.get('name')
        node.color = config.get('color')
        node.config = config
        return node

    def setup_nodes(self):
        for nc in self.config.get('nodes', []):
            klass = nc['type'].split('.')
            node_name = klass.pop(-1)
            klass.insert(0, 'nodes')
            mod = importlib.import_module('.'.join(klass))
            node_class = getattr(mod, node_name)
            node = self.make_node(node_class, nc)
            self.context.add_node(node)

    def generate_nodes_from_emission_sectors(self):
        mod = importlib.import_module('nodes.simple')
        node_class = getattr(mod, 'SectorEmissions')
        dataset_id = self.config.get('emission_dataset')
        for ec in self.config.get('emission_sectors', []):
            parent_id = ec.pop('part_of', None)
            data_col = ec.pop('column', None)
            nc = dict(
                output_nodes=[parent_id] if parent_id else [],
                input_datasets=[dict(id=dataset_id, column=data_col)] if data_col else [],
                **ec
            )
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
            page_type = pc.pop('type')
            if page_type == 'emission':
                node_id = pc.pop('node')
                node = self.context.get_node(node_id)
                page = EmissionPage(**pc, node=node)
            elif page_type == 'card':
                # FIXME
                cards = pc.get('cards', [])
                # page.add_cards(cards, self.context)
                raise Exception('Card page unsupported for now')
            else:
                raise Exception('Invalid page type: %s' % page_type)

            self.pages[pc['id']] = page

    def __init__(self, fn):
        data = yaml.load(open(fn, 'r'), Loader=yaml.Loader)
        self.context = Context()
        self.config = data['instance']
        self.instance = Instance(id=self.config['id'], name=self.config['name'])
        self.context.dataset_repo_url = self.config['dataset_repo']
        self.context.target_year = self.config['target_year']
        if False:
            dvc_pandas.pull_datasets()
        self.load_datasets(self.config.get('datasets', []))

        self.generate_nodes_from_emission_sectors()
        self.setup_nodes()
        self.setup_actions()
        self.setup_edges()
        self.setup_scenarios()
        self.setup_pages()

        self.context.generate_baseline_values()
        for scenario in self.context.scenarios.values():
            if scenario.default:
                break
        else:
            raise Exception('No default scenario defined')
        self.context.activate_scenario(scenario)
