import importlib
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, Optional

import dvc_pandas
import yaml

from common.i18n import TranslatedString
from nodes.actions import ActionNode
from nodes.exceptions import NodeError
from nodes.node import Node
from nodes.scenario import CustomScenario, Scenario
from pages.base import ActionPage, EmissionPage, Page

from . import Context, Dataset


logger = logging.getLogger(__name__)


@dataclass
class Instance:
    id: str
    name: TranslatedString
    context: Context
    target_year_goal: Optional[float] = None
    reference_year: Optional[int] = None
    minimum_historical_year: Optional[int] = None
    maximum_historical_year: Optional[int] = None

    pages: Optional[Dict[str, Page]] = None
    content_refreshed_at: Optional[datetime] = field(init=False)

    def __post_init__(self):
        self.content_refreshed_at = None

    def refresh(self):
        """Reload the Django models that have the rich-text content related to nodes.

        Reload only happens if the Instance has been updated since the last refresh.
        """
        from pages.models import InstanceContent

        iobj = InstanceContent.objects.filter(identifier=self.id).first()
        if iobj is None:
            return
        if self.content_refreshed_at is not None and self.content_refreshed_at >= iobj.modified_at:
            return

        context = self.context
        for pc in iobj.nodes.all():
            if pc.node_id not in context.nodes:
                logger.error('NodeContent exists for missing node ID: %s' % pc.node_id)
                continue
            node = context.nodes[pc.node_id]
            node.content = pc


class InstanceLoader:
    instance: Instance

    def make_trans_string(self, config: Dict, attr: str, pop: bool = False):
        default = config.get(attr)
        if pop and default is not None:
            del config[attr]
        langs = {}
        if default is not None:
            langs[self.config['default_language']] = default
        for key in list(config.keys()):
            m = re.match(r'%s_([a-z]+)' % attr, key)
            if m is None:
                continue
            langs[m.groups()[0]] = config[key]
            if pop:
                del config[key]
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
                ds_id = ds
                dc = {}
            else:
                ds_id = ds.pop('id')
                dc = ds
            o = Dataset(id=ds_id, **dc)
            datasets.append(o)

        if 'historical_values' in config or 'forecast_values' in config:
            datasets.append(Dataset.from_fixed_values(
                id=config['id'], unit=unit,
                historical=config.get('historical_values'),
                forecast=config.get('forecast_values'),
            ))

        node = node_class(
            self.context, config['id'], input_datasets=datasets,
        )
        node.name = self.make_trans_string(config, 'name')
        node.description = self.make_trans_string(config, 'description')
        node.color = config.get('color')
        node.target_year_goal = config.get('target_year_goal')

        if 'quantity' in config:
            node.quantity = config['quantity']
        node.unit = unit
        if node.id in self._input_nodes or node.id in self._output_nodes:
            raise Exception('Node %s is already configured' % node.id)
        assert node.id not in self._input_nodes
        assert node.id not in self._output_nodes
        self._input_nodes[node.id] = config.get('input_nodes', [])
        self._output_nodes[node.id] = config.get('output_nodes', [])

        params = config.get('params', [])
        if params:
            if isinstance(params, dict):
                params = [dict(id=param_id, value=value) for param_id, value in params.items()]
            # Ensure that the node class allows these parameters
            class_allowed_params = {p.id: p for p in getattr(node_class, 'allowed_params', [])}
            node.allowed_params = []
            for pc in params:
                param_id = pc.pop('id')
                name = self.make_trans_string(pc, 'name', pop=True)
                description = self.make_trans_string(pc, 'description', pop=True)
                scenario_values = pc.pop('values', {})
                param_obj = class_allowed_params.get(param_id)
                if param_obj is None:
                    raise NodeError(node, "Parameter %s not allowed by node class" % param_id)
                # Merge parameter values
                fields = asdict(param_obj)
                fields.update(pc)
                if description is not None:
                    fields['description'] = description
                if name is not None:
                    fields['name'] = description
                param = type(param_obj)(**fields)
                for scenario_id, value in scenario_values.items():
                    scenario_params = node.scenario_params.setdefault(scenario_id, {})
                    scenario_params[param.id] = param.clean(value)

                node.allowed_params.append(type(param_obj)(**fields))
        else:
            node.allowed_params = []

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
            if 'name_en' in ec:
                if 'emissions' not in ec['name_en']:
                    ec['name_en'] += ' emissions'
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
            for out_id in self._output_nodes.get(node.id, []):
                out_node = self.context.get_node(out_id)
                assert node not in out_node.input_nodes
                out_node.input_nodes.append(node)
                assert out_node not in node.output_nodes
                node.output_nodes.append(out_node)

            for in_id in self._input_nodes.get(node.id, []):
                in_node = self.context.get_node(in_id)
                assert node not in in_node.output_nodes
                in_node.output_nodes.append(node)
                assert in_node not in node.input_nodes
                node.input_nodes.append(in_node)

        # FIXME: Check for cycles?

    def setup_scenarios(self):
        default_scenario = None

        for sc in self.config['scenarios']:
            name = self.make_trans_string(sc, 'name', pop=True)
            params_config = sc.pop('params', [])
            params = {}
            for pc in params_config:
                param = self.context.get_param(pc['id'])
                params[param.id] = param.clean(pc['value'])
                # scenario.params[param.id] = param.clean(pc['value'])

            scenario = Scenario(**sc, name=name, context=self.context, params=params, nodes=self.context.nodes.values())

            if scenario.default:
                assert default_scenario is None
                default_scenario = scenario
            self.context.add_scenario(scenario)

        self.context.add_custom_scenario(
            CustomScenario(
                id='custom',
                name='Custom',
                context=self.context,
                base_scenario=default_scenario,
                nodes=self.context.nodes.values(),
            )
        )

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

    def setup_global_params(self):
        context = self.context
        for pc in self.config.get('params', []):
            param_id = pc['id']
            assert param_id not in context.params
            param_type = context.supported_params.get(param_id)
            if param_type is None:
                raise Exception('Unknown parameter: %s' % param_id)
            param_val = pc.pop('value', None)
            if 'is_customizable' not in pc:
                pc['is_customizable'] = False
            param = param_type(**pc)
            param.set(param_val)
            context.params[param_id] = param

    @classmethod
    def from_yaml(cls, filename):
        data = yaml.load(open(filename, 'r'), Loader=yaml.Loader)
        return cls(data['instance'])

    def __init__(self, config):
        self.config = config
        self.context = Context()
        instance_attrs = ['reference_year', 'minimum_historical_year', 'maximum_historical_year']
        self.instance = Instance(
            id=self.config['id'],
            name=self.make_trans_string(self.config, 'name'),
            context=self.context,
            **{attr: self.config.get(attr) for attr in instance_attrs}
        )
        static_datasets = self.config.get('static_datasets')
        if static_datasets is not None:
            if self.config.get('dataset_repo') is not None:
                raise Exception('static_datasets and dataset_repo may not be specified at the same time')
            self.context.dataset_repo = dvc_pandas.StaticRepository(static_datasets)
        else:
            self.context.dataset_repo = dvc_pandas.Repository(repo_url=self.config['dataset_repo'])
        self.context.target_year = self.config['target_year']

        self.load_datasets(self.config.get('datasets', []))
        # Store input and output node configs for each created node, to be used in setup_edges().
        self._input_nodes = {}
        self._output_nodes = {}
        self.generate_nodes_from_emission_sectors()
        self.setup_nodes()
        self.setup_actions()
        self.setup_edges()
        self.setup_global_params()
        self.setup_scenarios()
        self.setup_pages()

        for scenario in self.context.scenarios.values():
            if scenario.default:
                break
        else:
            raise Exception('No default scenario defined')
        self.context.activate_scenario(scenario)
