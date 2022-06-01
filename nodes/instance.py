import importlib
import logging
import re
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, Optional

import dvc_pandas
import pint
from ruamel.yaml import YAML as RuamelYAML
import yaml

from common.i18n import TranslatedString, gettext_lazy as _
from nodes.constants import DecisionLevel
from nodes.exceptions import NodeError
from nodes.node import Node
from nodes.scenario import CustomScenario, Scenario

from . import Context, Dataset, DVCDataset, FixedDataset

logger = logging.getLogger(__name__)


@dataclass
class Instance:
    id: str
    name: TranslatedString
    owner: TranslatedString
    default_language: str
    context: Context
    yaml_file_path: Optional[str] = None
    site_url: Optional[str] = None
    reference_year: Optional[int] = None
    minimum_historical_year: Optional[int] = None
    maximum_historical_year: Optional[int] = None
    supported_languages: Optional[list[str]] = None
    lead_title: Optional[TranslatedString] = None
    lead_paragraph: Optional[TranslatedString] = None

    modified_at: Optional[datetime] = field(init=False)

    @property
    def target_year(self) -> int:
        return self.context.target_year

    def __post_init__(self):
        self.modified_at = None
        if not self.supported_languages:
            self.supported_languages = [self.default_language]
        else:
            if self.default_language not in self.supported_languages:
                self.supported_languages.append(self.default_language)

    def update_dataset_repo_commit(self, commit_id: str):
        assert self.yaml_file_path
        yaml_obj = RuamelYAML()
        with open(self.yaml_file_path, 'r', encoding='utf8') as f:
            data = yaml_obj.load(f)
        if 'instance' in data:
            instance_data = data['instance']
        else:
            instance_data = data
        instance_data['dataset_repo']['commit'] = commit_id
        with open(self.yaml_file_path, 'w', encoding='utf8') as f:
            yaml_obj.dump(data, f)


class InstanceLoader:
    instance: Instance
    yaml_file_path: Optional[str] = None

    def make_trans_string(self, config: Dict, attr: str, pop: bool = False, default_language=None):
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
        return TranslatedString(**langs, default_language=default_language or self.instance.default_language)

    def make_node(self, node_class, config) -> Node:
        ds_config = config.get('input_datasets', None)
        datasets: list[Dataset] = []

        dimensions = getattr(node_class, 'dimensions', None)
        unit = config.get('unit')
        if unit is None:
            unit = getattr(node_class, 'unit', None)
            if not unit and not dimensions:
                raise Exception('Node %s has no unit set' % config['id'])
        if unit and not isinstance(unit, pint.Unit):
            unit = self.context.unit_registry(unit).units

        quantity = config.get('quantity')
        if quantity is None:
            quantity = getattr(node_class, 'quantity', None)
            if not quantity and not dimensions:
                raise Exception('Node %s has no quantity set' % config['id'])

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
            ds_unit = dc.pop('unit', None)
            if ds_unit is not None and not isinstance(ds_unit, pint.Unit):
                ds_unit = self.context.unit_registry(ds_unit).units
                assert isinstance(ds_unit, pint.Unit)
            o = DVCDataset(id=ds_id, unit=ds_unit, **dc)
            datasets.append(o)

        if 'historical_values' in config or 'forecast_values' in config:
            datasets.append(FixedDataset(
                id=config['id'], unit=unit,
                historical=config.get('historical_values'),
                forecast=config.get('forecast_values'),
            ))

        node = node_class(
            id=config['id'],
            context=self.context,
            name=self.make_trans_string(config, 'name'),
            description=self.make_trans_string(config, 'description'),
            color=config.get('color'),
            unit=unit,
            quantity=quantity,
            target_year_goal=config.get('target_year_goal'),
            input_datasets=datasets,
        )

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
            class_allowed_params = {p.local_id: p for p in getattr(node_class, 'allowed_parameters', [])}
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
                node.add_parameter(param)

                for scenario_id, value in scenario_values.items():
                    param.add_scenario_setting(scenario_id, param.clean(value))

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
        emission_unit = self.config.get('emission_unit')
        if emission_unit is not None:
            emission_unit = self.context.unit_registry(emission_unit).units

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
                    unit=emission_unit,
                )] if data_col else [],
                unit=emission_unit,
                **ec
            )
            node = self.make_node(node_class, nc)
            self.context.add_node(node)

    def setup_actions(self):
        for nc in self.config.get('actions', []):
            klass = nc['type'].split('.')
            node_name = klass.pop(-1)
            klass.insert(0, 'nodes')
            klass.insert(1, 'actions')
            mod = importlib.import_module('.'.join(klass))
            node_class = getattr(mod, node_name)
            node = self.make_node(node_class, nc)
            decision_level = nc.get('decision_level')
            if decision_level is not None:
                for name, val in DecisionLevel.__members__.items():
                    if decision_level == name.lower():
                        break
                else:
                    raise Exception('Invalid decision level for action %s: %s' % (nc['id'], val))
                node.decision_level = val
            self.context.add_node(node)

    def setup_edges(self):
        # Setup edges
        for node in self.context.nodes.values():
            for out_id in self._output_nodes.get(node.id, []):
                out_node = self.context.get_node(out_id)
                out_node.add_input_node(node)
                node.add_output_node(out_node)

            for in_id in self._input_nodes.get(node.id, []):
                in_node = self.context.get_node(in_id)
                in_node.add_output_node(node)
                node.add_input_node(in_node)

        # FIXME: Check for cycles?

    def setup_scenarios(self):
        default_scenario = None

        for sc in self.config['scenarios']:
            params_config = sc.pop('params', [])
            for pc in params_config:
                param = self.context.get_parameter(pc['id'])
                param.add_scenario_setting(sc['id'], param.clean(pc['value']))

            name = self.make_trans_string(sc, 'name', pop=True)
            scenario = Scenario(**sc, name=name, notified_nodes=self.context.nodes.values())

            if scenario.default:
                assert default_scenario is None
                default_scenario = scenario
            self.context.add_scenario(scenario)

        self.context.set_custom_scenario(
            CustomScenario(
                id='custom',
                name=_('Custom'),
                base_scenario=default_scenario,
                notified_nodes=self.context.nodes.values(),
            )
        )

    def load_datasets(self, datasets):
        for ds in datasets:
            self.context.add_dataset(ds)

    def setup_global_parameters(self):
        context = self.context
        for pc in self.config.get('params', []):
            param_id = pc.pop('id')
            pc['local_id'] = param_id
            param_type = context.get_parameter_type(param_id)
            param_val = pc.pop('value', None)
            if 'is_customizable' not in pc:
                pc['is_customizable'] = False
            param = param_type(**pc)
            param.set(param_val)
            context.add_global_parameter(param)

    @classmethod
    def from_yaml(cls, filename):
        data = yaml.load(open(filename, 'r', encoding='utf8'), Loader=yaml.Loader)
        if 'instance' in data:
            data = data['instance']

        framework = data.get('framework')
        if framework:
            base_dir = os.path.dirname(filename)
            framework_fn = os.path.join(base_dir, 'frameworks', framework + '.yaml')
            if not os.path.exists(framework_fn):
                raise Exception("Config expects framework but %s does not exist" % framework_fn)
            fw_data = yaml.load(open(framework_fn, 'r', encoding='utf8'), Loader=yaml.Loader)
            if 'nodes' in fw_data:
                data['nodes'] += fw_data['nodes']
            if 'emission_sectors' in fw_data:
                data['emission_sectors'] += fw_data['emission_sectors']


        return cls(data, yaml_file_path=filename)

    def __init__(self, config: dict, yaml_file_path: str = None):
        self.config = config

        static_datasets = self.config.get('static_datasets')
        if static_datasets is not None:
            if self.config.get('dataset_repo') is not None:
                raise Exception('static_datasets and dataset_repo may not be specified at the same time')
            dataset_repo = dvc_pandas.StaticRepository(static_datasets)
        else:
            dataset_repo_config = self.config['dataset_repo']
            repo_url = dataset_repo_config['url']
            commit = dataset_repo_config.get('commit')
            dataset_repo = dvc_pandas.Repository(repo_url=repo_url)
            dataset_repo.set_target_commit(commit)
        target_year = self.config['target_year']
        self.context = Context(dataset_repo, target_year)

        instance_attrs = [
            'reference_year', 'minimum_historical_year', 'maximum_historical_year',
            'default_language', 'supported_languages', 'site_url',
        ]
        self.instance = Instance(
            id=self.config['id'],
            name=self.make_trans_string(self.config, 'name', default_language=self.config['default_language']),
            owner=self.make_trans_string(self.config, 'owner', default_language=self.config['default_language']),
            context=self.context,
            **{attr: self.config.get(attr) for attr in instance_attrs},
            # FIXME: The YAML file seems to specify what's supposed to be in InstanceConfig.lead_title (and other
            # attributes), but not under `instance` but under `pages` for a "page" whose `id' is `home`. It's a mess.
            **self._build_instance_args_from_home_page(),
        )
        self.context.instance = self.instance
        self.instance.yaml_file_path = yaml_file_path

        self.load_datasets(self.config.get('datasets', []))
        # Store input and output node configs for each created node, to be used in setup_edges().
        self._input_nodes = {}
        self._output_nodes = {}
        self.generate_nodes_from_emission_sectors()
        self.setup_nodes()
        self.setup_actions()
        self.setup_edges()
        self.setup_global_parameters()
        self.setup_scenarios()

        for scenario in self.context.scenarios.values():
            if scenario.default:
                break
        else:
            raise Exception('No default scenario defined')
        self.context.activate_scenario(scenario)

    def _build_instance_args_from_home_page(self):
        # FIXME: This is an ugly hack
        pages = self.config.get('pages', [])
        for page in pages:
            if page['id'] == 'home':
                break
        else:
            return {}
        default_language = self.config['default_language']
        return {
            'lead_title': self.make_trans_string(page, 'lead_title', default_language=default_language),
            'lead_paragraph': self.make_trans_string(page, 'lead_paragraph', default_language=default_language),
        }
