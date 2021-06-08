from dataclasses import dataclass

import dvc_pandas
import importlib
import re
import yaml
from nodes.node import Node
from nodes.actions.base import ActionNode
from nodes.scenario import Scenario
from typing import Dict

from common.i18n import TranslatedString
from pages.base import ActionPage, EmissionPage, Page
from . import Dataset, Context


@dataclass
class Instance:
    id: str
    name: str
    context: Context
    pages: Dict[str, Page] = None


class InstanceLoader:
    instance: Instance

    def make_trans_string(self, config, attr):
        default = config.get(attr)
        langs = {}
        if default is not None:
            langs[self.config['default_language']] = default
        for key in config.keys():
            m = re.match(r'%s_([a-z]+)' % attr, key)
            if m is None:
                continue
            langs[m.groups()[0]] = config[key]
        if not langs:
            return None
        return TranslatedString(**langs)

    def make_node(self, node_class, config) -> Node:
        ds_config = config.get('input_datasets', None)
        datasets = []

        unit = config.get('unit')
        if unit is None:
            unit = getattr(node_class, 'unit')
        if unit:
            unit = self.context.unit_registry(unit).units

        # If the graph doesn't specify input datasets, the node
        # might.
        if ds_config is None:
            ds_config = getattr(node_class, 'input_datasets', [])

        for ds in ds_config:
            if isinstance(ds, str):
                dc = {'id': ds}
            else:
                dc = ds
            o = Dataset(**dc)
            datasets.append(o)

        if 'historical_values' in config or 'forecast_values' in config:
            datasets.append(Dataset.from_fixed_values(
                id=config['id'], unit=unit,
                historical=config.get('historical_values'),
                forecast=config.get('forecast_values'),
            ))

        node = node_class(self.context, config['id'], input_datasets=datasets)
        node.name = self.make_trans_string(config, 'name')
        node.color = config.get('color')
        if 'quantity' in config:
            node.quantity = config['quantity']
        node.unit = unit
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
        unit = self.config.get('emission_unit')

        for ec in self.config.get('emission_sectors', []):
            parent_id = ec.pop('part_of', None)
            data_col = ec.pop('column', None)
            nc = dict(
                output_nodes=[parent_id] if parent_id else [],
                input_datasets=[dict(
                    id=dataset_id,
                    column=data_col,
                    forecast_from=self.config.get('emission_forecast_from'),
                )] if data_col else [],
                unit=unit,
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
                assert node not in out_node.input_nodes
                out_node.input_nodes.append(node)
                assert out_node not in node.output_nodes
                node.output_nodes.append(out_node)

            for in_id in node.config.get('input_nodes', []):
                in_node = self.context.get_node(in_id)
                assert node not in in_node.output_nodes
                in_node.output_nodes.append(node)
                assert in_node not in node.input_nodes
                node.input_nodes.append(in_node)

        # FIXME: Check for cycles?

    def setup_scenarios(self):
        for sc in self.config['scenarios']:
            actions = sc.pop('actions', [])
            all_actions_enabled = sc.pop('all_actions_enabled', False)
            scenario = Scenario(**sc)
            if all_actions_enabled:
                for node in self.context.nodes.values():
                    if not isinstance(node, ActionNode):
                        continue
                    param = node.get_param('enabled')
                    scenario.params.append((param, True))
            """
            for act_conf in actions:
                node = self.context.get_node(act_conf.pop('id'))
                assert isinstance(node, Action)
                scenario.actions.append([node, act_conf])
            """
            self.context.add_scenario(scenario)

    def print_graph(self, node=None, indent=0):
        from colored import fg, attr

        if node is None:
            all_nodes = self.context.nodes.values()
            root_nodes = list(filter(lambda node: not node.output_nodes, all_nodes))
            assert len(root_nodes) == 1
            node = root_nodes[0]

        if isinstance(node, ActionNode):
            node_color = 'green'
        else:
            node_color = 'yellow'
        node_str = f"{fg(node_color)}{node.id} "
        node_str += f"{fg('grey_50')}{str(type(node))} "
        node_str += attr('reset')
        print('  ' * indent + node_str)
        for in_node in node.input_nodes:
            self.print_graph(in_node, indent + 1)

    def load_datasets(self, datasets):
        for ds in datasets:
            self.context.add_dataset(ds)

    def setup_pages(self):
        instance = self.instance
        instance.pages = {}

        for pc in self.config['pages']:
            assert pc['id'] not in instance.pages
            page_type = pc.pop('type')
            if page_type == 'emission':
                node_id = pc.pop('node')
                node = self.context.get_node(node_id)
                page = EmissionPage(**pc, node=node)
            elif page_type == 'card':
                raise Exception('Card page unsupported for now')
            else:
                raise Exception('Invalid page type: %s' % page_type)

            instance.pages[pc['id']] = page

        for node in self.context.nodes.values():
            if not isinstance(node, ActionNode):
                continue
            page = ActionPage(id=node.id, name=node.name, path='/actions/%s' % node.id, action=node)
            instance.pages[node.id] = page

    def __init__(self, fn):
        data = yaml.load(open(fn, 'r'), Loader=yaml.Loader)
        self.context = Context()
        self.config = data['instance']
        self.instance = Instance(
            id=self.config['id'], name=self.config['name'], context=self.context
        )
        self.context.dataset_repo = dvc_pandas.Repository(repo_url=self.config['dataset_repo'])
        self.context.target_year = self.config['target_year']

        self.load_datasets(self.config.get('datasets', []))

        self.generate_nodes_from_emission_sectors()
        self.setup_nodes()
        self.setup_actions()
        self.setup_edges()
        self.setup_scenarios()
        self.setup_pages()

        for scenario in self.context.scenarios.values():
            if scenario.default:
                break
        else:
            raise Exception('No default scenario defined')
        self.context.activate_scenario(scenario)
