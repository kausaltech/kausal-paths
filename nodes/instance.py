import dataclasses
from functools import cached_property
import importlib
import re
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
import threading
from typing import TYPE_CHECKING, Any, Dict, Iterable, Literal, Optional, Sequence, Tuple, Type, overload
from pydantic.dataclasses import dataclass as pydantic_dataclass

import dvc_pandas
import pint
from loguru import logger
from ruamel.yaml import YAML as RuamelYAML, CommentedMap
from ruamel.yaml.comments import LineCol
from rich import print

from common.i18n import I18nBaseModel, I18nStringInstance, TranslatedString, gettext_lazy as _, set_default_language
from nodes.actions.action import ActionEfficiencyPair, ActionGroup, ActionNode
from nodes.constants import DecisionLevel
from nodes.exceptions import NodeError
from nodes.goals import NodeGoalsEntry
from nodes.node import Edge, Node, NodeMetric
from nodes.normalization import Normalization
from nodes.scenario import CustomScenario, Scenario
from nodes.processors import Processor
from nodes.units import Unit
from pages.config import OutcomePage, pages_from_config
from params.param import ReferenceParameter, Parameter

from . import Context, Dataset, DVCDataset, FixedDataset

if TYPE_CHECKING:
    from loguru import Logger
    from .models import InstanceConfig


yaml = RuamelYAML()


class InstanceTerms(I18nBaseModel):
    action: TranslatedString | None = None
    enabled_label: TranslatedString | None = None


@pydantic_dataclass
class InstanceFeatures:
    baseline_visible_in_graphs: bool = True
    show_accumulated_effects: bool = True
    show_significant_digits: int = 3


@dataclass
class Instance:
    id: str
    name: TranslatedString
    owner: TranslatedString
    default_language: str
    context: Context
    _: dataclasses.KW_ONLY
    yaml_file_path: Optional[str] = None
    site_url: Optional[str] = None
    reference_year: Optional[int] = None
    minimum_historical_year: int
    maximum_historical_year: Optional[int] = None
    supported_languages: list[str] = field(default_factory=list)
    lead_title: Optional[TranslatedString] = None
    lead_paragraph: Optional[TranslatedString] = None
    theme_identifier: Optional[str] = None
    features: InstanceFeatures = field(default_factory=InstanceFeatures)
    terms: InstanceTerms = field(default_factory=InstanceTerms)
    action_groups: list[ActionGroup] = field(default_factory=list)
    pages: list[OutcomePage] = field(default_factory=list)

    lock: threading.Lock = field(init=False)

    @property
    def target_year(self) -> int:
        return self.context.target_year

    @property
    def model_end_year(self) -> int:
        return self.context.model_end_year

    @cached_property
    def config(self) -> 'InstanceConfig':
        from .models import InstanceConfig
        return InstanceConfig.objects.get(identifier=self.id)

    def __post_init__(self):
        self.logger: Logger = logger.bind(instance=self.id)
        self.modified_at: datetime | None = None
        self.lock = threading.Lock()
        if isinstance(self.features, dict):
            self.features = InstanceFeatures(**self.features)
        if isinstance(self.terms, dict):
            self.terms = InstanceTerms(**self.terms)
        if not self.supported_languages:
            self.supported_languages = [self.default_language]
        else:
            if self.default_language not in self.supported_languages:
                self.supported_languages.append(self.default_language)

    def update_dataset_repo_commit(self, commit_id: str):
        assert self.yaml_file_path
        with open(self.yaml_file_path, 'r', encoding='utf8') as f:
            data = yaml.load(f)
        if 'instance' in data:
            instance_data = data['instance']
        else:
            instance_data = data
        instance_data['dataset_repo']['commit'] = commit_id
        with open(self.yaml_file_path, 'w', encoding='utf8') as f:
            yaml.dump(data, f)

    def warning(self, msg: Any, *args):
        self.logger.opt(depth=1).warning(msg, *args)

    @overload
    def get_goals(self, goal_id: str) -> NodeGoalsEntry: ...

    @overload
    def get_goals(self) -> list[NodeGoalsEntry]: ...

    def get_goals(self, goal_id: str | None = None):
        ctx = self.context
        outcome_nodes = ctx.get_outcome_nodes()
        goals: list[NodeGoalsEntry] = []
        for node in outcome_nodes:
            if not node.goals:
                continue
            for ge in node.goals.root:
                if not ge.is_main_goal:
                    continue
                if goal_id is not None:
                    if ge.get_id() != goal_id:
                        continue
                goals.append(ge)

        if goal_id:
            if len(goals) != 1:
                raise Exception("Goal with id %s not found", goal_id)
            return goals[0]

        return goals

class InstanceLoader:
    instance: Instance
    default_language: str
    yaml_file_path: Optional[str] = None
    config: CommentedMap
    _input_nodes: dict[str, list[dict | str]]
    _output_nodes: dict[str, list[dict | str]]
    _subactions: dict[str, list[str]]
    _scenario_values: dict[str, list[tuple[Parameter, Any]]]

    @overload
    def make_trans_string(
        self, config: Dict, attr: str, pop: bool = False, required: Literal[True] = True,
        default_language=None
    ) -> TranslatedString: ...

    @overload
    def make_trans_string(
        self, config: Dict, attr: str, pop: bool = False, required: Literal[False] = False,
        default_language=None
    ) -> TranslatedString | None: ...

    def make_trans_string(
        self, config: Dict, attr: str, pop: bool = False, required: bool = False,
        default_language=None
    ):
        default_language = default_language or self.config['default_language']
        all_langs = set([self.config['default_language']])
        all_langs.update(set(self.config.get('supported_languages', [])))

        default = config.get(attr)
        if pop and default is not None:
            del config[attr]
        langs = {}
        if default is not None:
            langs[self.config['default_language']] = default
        for key in list(config.keys()):
            m = re.match(r'%s_(([a-z]{2})(-[A-Z]{2})?)$' % attr, key)
            if m is None:
                continue
            full, lang, region = m.groups()
            if full not in all_langs:
                matches = [x for x in all_langs if x.startswith('%s-' % lang)]
                if len(matches) > 1:
                    raise Exception("Too many languages match %s" % full)
                if len(matches) == 1:
                    full = matches[0]
                else:
                    raise Exception("Unsupported language: %s" % full)

            langs[full] = config[key]
            if pop:
                del config[key]
        if not langs:
            if required:
                raise Exception("Value for field %s missing" % attr)
            return None
        return TranslatedString(**langs, default_language=default_language or self.default_language)

    def setup_processors(self, node: Node, confs: list[dict | str]):
        processors = []
        for idp_conf in confs:
            if isinstance(idp_conf, str):
                class_path = idp_conf
                idp_conf = {}
            else:
                class_path = idp_conf.pop('type')

            params = idp_conf.get('params', {})
            p_class: Type[Processor] = self.import_class(
                class_path, 'nodes.processors', allowed_classes=[Processor]
            )
            processors.append(p_class(self.context, node, params=params))
        return processors

    def make_node(self, node_class: Type[Node], config: dict, yaml_lc: LineCol | None = None) -> Node:
        ds_config = config.get('input_datasets', None)
        datasets: list[Dataset] = []

        metrics_conf = config.get('output_metrics', None)
        metrics: dict[str, NodeMetric] | None
        if metrics_conf is not None:
            metrics = {m['id']: NodeMetric.from_config(m) for m in metrics_conf}
        else:
            metrics = getattr(node_class, 'output_metrics', None)
        unit = config.get('unit')
        if unit is None:
            unit = getattr(node_class, 'default_unit', None)
            if unit is None:
                unit = getattr(node_class, 'unit', None)
            if not unit and not metrics:
                raise Exception('Node %s has no unit set' % config['id'])
        if unit and not isinstance(unit, Unit):
            unit = self.context.unit_registry.parse_units(unit)

        quantity = config.get('quantity')
        if quantity is None:
            quantity = getattr(node_class, 'quantity', None)
            if not quantity and not metrics:
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
                ds_unit = self.context.unit_registry.parse_units(ds_unit)
                assert isinstance(ds_unit, pint.Unit)
            tags = dc.pop('tags', [])
            o = DVCDataset(id=ds_id, unit=ds_unit, tags=tags, **dc)
            datasets.append(o)

        if 'historical_values' in config or 'forecast_values' in config:
            datasets.append(FixedDataset(
                id=config['id'], unit=unit,  # type: ignore
                tags=config.get('tags', []),
                historical=config.get('historical_values'),
                forecast=config.get('forecast_values'),
            ))

        yaml_lct: Tuple[int, int] | None = (yaml_lc.line + 1, yaml_lc.col) if yaml_lc else None  # type: ignore
        node: Node = node_class(
            id=config['id'],
            context=self.context,
            name=self.make_trans_string(config, 'name'),
            short_name=self.make_trans_string(config, 'short_name'),
            quantity=quantity,
            unit=unit,
            description=self.make_trans_string(config, 'description'),
            color=config.get('color'),
            order=config.get('order'),
            is_outcome=config.get('is_outcome', False),
            minimum_year=config.get('minimum_year', None),
            target_year_goal=config.get('target_year_goal'),
            goals=config.get('goals'),
            input_datasets=datasets,
            output_dimension_ids=config.get('output_dimensions'),
            input_dimension_ids=config.get('input_dimensions'),
            output_metrics=metrics,
            yaml_lc=yaml_lct,
            yaml_fn=self.yaml_file_path,
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

                param_obj = class_allowed_params.get(param_id)
                if param_obj is None:
                    raise NodeError(node, "Parameter %s not allowed by node class" % param_id)
                param_class = type(param_obj)

                label = self.make_trans_string(pc, 'label', pop=True) or param_obj.label
                ref = pc.pop('ref', None)
                description = self.make_trans_string(pc, 'description', pop=True) or param_obj.description

                scenario_values = pc.pop('values', {})

                if ref is not None:
                    target = self.context.global_parameters.get(ref)
                    if target is None:
                        raise NodeError(node, "Parameter %s refers to an unknown global parameter: %s" % (param_id, ref))

                    if not isinstance(target, param_class):
                        raise NodeError(node, "Node requires parameter of type %s, but referenced parameter %s is %s" % (
                            param_class, ref, type(target)
                        ))
                    param = ReferenceParameter(
                        local_id=param_obj.local_id, label=param_obj.label, target=target,
                        context=self.context
                    )
                    node.add_parameter(param)
                    continue

                # Merge parameter values
                fields = asdict(param_obj)
                fields.update(pc)
                if description is not None:
                    fields['description'] = description
                if label is not None:
                    fields['label'] = label
                fields['context'] = self.context

                unit = fields.get('unit', None)
                if unit is not None:
                    if isinstance(unit, str):
                        fields['unit'] = self.context.unit_registry.parse_units(unit)

                value = fields.pop('value', None)
                param = param_class(**fields)
                assert isinstance(param, Parameter)
                node.add_parameter(param)

                try:
                    if value is not None:
                        param.set(value)
                except:
                    self.instance.logger.error("Error setting parameter %s for node %s" % (param.local_id, node.id))
                    raise

                for scenario_id, value in scenario_values.items():
                    sv = self._scenario_values.setdefault(scenario_id, list())
                    sv.append((param, param.clean(value)))

        tags = config.get('tags', None)
        if isinstance(tags, str):
            tags = [tags]
        if tags:
            for tag in tags:
                if not isinstance(tag, str):
                    raise NodeError(node, "'tags' must be a list of strings")
            node.tags.update(tags)

        idp_confs = config.get('input_dataset_processors', [])
        node.input_dataset_processors = self.setup_processors(node, idp_confs)

        return node

    def import_class(
        self, path: str, path_prefix: str | None = None,
        allowed_classes: Iterable[Type] | None = None,
        disallowed_classes: Iterable[Type] | None = None,
        node_id: str | None = None
    ) -> Type:
        if not path:
            raise Exception("Node %s: no class path given" % node_id)
        parts = path.split('.')
        class_name = parts.pop(-1)
        if path_prefix:
            prefix_parts = path_prefix.split('.')
            parts = prefix_parts + parts

        mod = importlib.import_module('.'.join(parts))
        klass = getattr(mod, class_name)
        if allowed_classes:
            if not issubclass(klass, tuple(allowed_classes)):
                raise Exception("%s is not a subclass of %s" % (klass, allowed_classes))
        if disallowed_classes:
            for k in disallowed_classes:
                if issubclass(klass, k):
                    raise Exception("%s is a subclass of disallowed %s" % (klass, disallowed_classes))
        return klass

    def setup_dimensions(self):
        from .dimensions import Dimension

        for dc in self.config.get('dimensions', []):
            try:
                dim = Dimension.model_validate(dc)
            except Exception:
                print(dc)
                raise
            assert dim.id not in self.context.dimensions
            self.context.dimensions[dim.id] = dim

    def setup_nodes(self):
        for nc in self.config.get('nodes', []):
            try:
                node_class = self.import_class(
                    nc['type'], 'nodes', allowed_classes=[Node], disallowed_classes=[ActionNode],
                    node_id=nc['id'],
                )
            except ImportError:
                logger.error('Unable to import node class for %s' % nc.get('id'))
                raise

            try:
                node = self.make_node(node_class, nc, yaml_lc=nc.lc)
            except NodeError as err:
                raise
            self.context.add_node(node)

    def generate_nodes_from_emission_sectors(self):
        mod = importlib.import_module('nodes.simple')
        node_class = getattr(mod, 'SectorEmissions')
        dataset_id = self.config.get('emission_dataset')
        emission_unit = self.config.get('emission_unit')
        assert emission_unit is not None
        emission_unit = self.context.unit_registry.parse_units(emission_unit)

        for ec in self.config.get('emission_sectors', []):
            parent_id = ec.pop('part_of', None)
            data_col = ec.pop('column', None)
            data_category = ec.pop('category', None)
            if 'name_en' in ec:
                if 'emissions' not in ec['name_en']:
                    ec['name_en'] += ' emissions'
            nc = dict(
                output_nodes=[parent_id] if parent_id else [],
                input_dimensions=self.config.get('emission_dimensions', []),
                output_dimensions=self.config.get('emission_dimensions', []),
                input_datasets=[dict(
                    id=dataset_id,
                    column=data_col,
                    forecast_from=self.config.get('emission_forecast_from'),
                    unit=emission_unit,
                )] if data_col or data_category else [],
                unit=emission_unit,
                params=dict(category=data_category) if data_category else [],
                **ec
            )
            node = self.make_node(node_class, nc, yaml_lc=ec.lc)
            self.context.add_node(node)

    def setup_actions(self):
        from nodes.actions import ActionNode

        for nc in self.config.get('actions', []):
            klass = nc['type'].split('.')
            node_name = klass.pop(-1)
            klass.insert(0, 'nodes')
            klass.insert(1, 'actions')
            mod = importlib.import_module('.'.join(klass))
            node_class = getattr(mod, node_name)
            node = self.make_node(node_class, nc)
            assert isinstance(node, ActionNode)

            decision_level = nc.get('decision_level')
            if decision_level is not None:
                for name, val in DecisionLevel.__members__.items():
                    if decision_level == name.lower():
                        break
                else:
                    raise Exception('Invalid decision level for action %s: %s' % (nc['id'], decision_level))
                node.decision_level = val

            ag_id = nc.get('group', None)
            if ag_id is not None:
                assert isinstance(ag_id, str)
                for ag in self.instance.action_groups:
                    if ag.id == ag_id:
                        break
                else:
                    raise Exception("Action group '%s' not found for action %s" % (ag_id, nc['id']))
                node.group = ag

            parent_id = nc.get('parent', None)
            if parent_id is not None:
                subs = self._subactions.setdefault(parent_id, [])
                subs.append(node.id)

            self.context.add_node(node)

    def setup_edges(self):
        from nodes.actions.parent import ParentActionNode

        # Setup edges
        ctx = self.context
        for node in ctx.nodes.values():
            try:
                for ec in self._output_nodes.get(node.id, []):
                    edge = Edge.from_config(ec, node=node, is_output=True, context=ctx)
                    node.add_edge(edge)
                    edge.output_node.add_edge(edge)

                for ec in self._input_nodes.get(node.id, []):
                    edge = Edge.from_config(ec, node=node, is_output=False, context=ctx)
                    node.add_edge(edge)
                    edge.input_node.add_edge(edge)
            except Exception:
                logger.error("Error setting up edges for node %s" % node)
                raise

        for parent_id, subs in self._subactions.items():
            parent = ctx.nodes.get(parent_id)
            if parent is None:
                raise Exception("Action parent '%s' not found" % parent_id)
            if not isinstance(parent, ParentActionNode):
                raise Exception("Action '%s' is marked as a parent but is not a ParentActionNode" % parent_id)
            for sub_id in subs:
                node = ctx.get_node(sub_id)
                assert isinstance(node, ActionNode)
                parent.add_subaction(node)
                node.parent_action = parent

        ctx.finalize_nodes()

    def setup_scenarios(self):
        default_scenario = None

        for sc in self.config['scenarios']:
            name = self.make_trans_string(sc, 'name', pop=True)
            params_config = sc.pop('params', [])
            scenario = Scenario(self.context, **sc, name=name)

            for pc in params_config:
                param = self.context.get_parameter(pc['id'])
                scenario.add_parameter(param, param.clean(pc['value']))

            for param, value in self._scenario_values.get(scenario.id, []):
                scenario.add_parameter(param, value)

            if scenario.default:
                assert default_scenario is None
                default_scenario = scenario
            self.context.add_scenario(scenario)

        if default_scenario is None:
            raise Exception("Default scenario not defined")

        for param in self.context.get_all_parameters():
            if not param.is_customizable:
                continue
            if default_scenario.has_parameter(param):
                continue
            default_scenario.add_parameter(param, param.value)

        self.context.set_custom_scenario(
            CustomScenario(
                context=self.context,
                id='custom',
                name=_('Custom'),
                base_scenario=default_scenario,
            )
        )

    # deprecated
    def load_datasets(self, datasets):
        for ds in datasets:
            self.context.add_dataset(ds)

    def setup_global_parameters(self):
        context = self.context
        for pc in self.config.get('params', []):
            param_id = pc.pop('id')
            pc['local_id'] = param_id
            unit_str = pc.get('unit', None)
            if unit_str is not None:
                unit = context.unit_registry.parse_units(unit_str)
                pc['unit'] = unit
            param_type = context.get_parameter_type(param_id)
            param_val = pc.pop('value', None)
            if 'is_customizable' not in pc:
                pc['is_customizable'] = False
            param = param_type(**pc)
            sub_node_ids = pc.get('subscription_nodes', None)
            if sub_node_ids is not None:
                sub_nodes = []
                for node_id in sub_node_ids:
                    sub_nodes += [context.get_node(node_id)]
                param.subscription_nodes = sub_nodes
            param.set(param_val)
            context.add_global_parameter(param)

    def setup_action_efficiency_pairs(self):
        conf = self.config.get('action_efficiency_pairs', [])
        for aepc in conf:
            label = self.make_trans_string(aepc, 'label', pop=False)
            aep = ActionEfficiencyPair.from_config(
                self.context, aepc['cost_node'], aepc['impact_node'], aepc['efficiency_unit'],
                aepc['cost_unit'], aepc['impact_unit'],
                plot_limit_efficiency=aepc.get('plot_limit_efficiency', None),
                invert_cost=aepc.get('invert_cost', False), invert_impact=aepc.get('invert_impact', False), label=label,
            )
            self.context.action_efficiency_pairs.append(aep)

    def setup_normalizations(self):
        ncs = self.config.get('normalizations', [])
        for nc in ncs:
            n = Normalization.from_config(self.context, nc)
            n_id = n.normalizer_node.id
            self.context.add_normalization(n_id, n)

    @classmethod
    def merge_framework_config(cls, confs: list[dict], fw_confs: list[dict]):
        by_id = {d['id']: d for d in confs}
        for fwn in fw_confs:
            n = by_id.get(fwn['id'])
            if n is None:
                confs.append(fwn)
            else:
                continue
                # Merge the configs with the node config overriding framework config
                for key, val in fwn.items():
                    if key not in n:
                        n[key] = val

    @classmethod
    def from_yaml(cls, filename):
        data = yaml.load(open(filename, 'r', encoding='utf8'))
        if 'instance' in data:
            data = data['instance']

        framework = data.get('framework')
        if framework:
            base_dir = os.path.dirname(filename)
            framework_fn = os.path.join(base_dir, 'frameworks', framework + '.yaml')
            if not os.path.exists(framework_fn):
                raise Exception("Config expects framework but %s does not exist" % framework_fn)
            fw_data = yaml.load(open(framework_fn, 'r', encoding='utf8'))
            cls.merge_framework_config(data['nodes'], fw_data.get('nodes', []))
            cls.merge_framework_config(data['emission_sectors'], fw_data.get('emission_sectors', []))

        return cls(data, yaml_file_path=filename)

    def __init__(self, config: dict, yaml_file_path: str | None = None):
        self.yaml_file_path = os.path.abspath(yaml_file_path) if yaml_file_path else None
        self.config = config
        self.default_language = config['default_language']
        with set_default_language(self.default_language):
            self._init_instance()

    def _init_instance(self):
        config = self.config
        static_datasets = self.config.get('static_datasets')
        instance_id = config['id']
        dataset_repo_default_path = None
        if static_datasets is not None:
            if self.config.get('dataset_repo') is not None:
                raise Exception('static_datasets and dataset_repo may not be specified at the same time')
            dataset_repo = dvc_pandas.StaticRepository(static_datasets)
        else:
            dataset_repo_config = self.config['dataset_repo']
            repo_url = dataset_repo_config['url']
            commit = dataset_repo_config.get('commit')
            dataset_repo = dvc_pandas.Repository(
                repo_url=repo_url, dvc_remote=dataset_repo_config.get('dvc_remote'),
                cache_prefix=instance_id,
            )
            dataset_repo.set_target_commit(commit)
            dataset_repo_default_path = dataset_repo_config.get('default_path')
        target_year = self.config['target_year']
        model_end_year = self.config.get('model_end_year', target_year)
        self.context = Context(
            dataset_repo, target_year, model_end_year=model_end_year, dataset_repo_default_path=dataset_repo_default_path
        )

        instance_attrs = [
            'reference_year', 'minimum_historical_year', 'maximum_historical_year',
            'supported_languages', 'site_url', 'theme_identifier',
        ]
        agc_all = self.config.get('action_groups', [])
        agcs = []
        for agc in agc_all:
            assert 'name' in agc
            ag = ActionGroup(agc['id'], self.make_trans_string(agc, 'name'), agc.get('color'))
            agcs.append(ag)

        self.instance = Instance(
            id=self.config['id'],
            name=self.make_trans_string(self.config, 'name', required=True),
            owner=self.make_trans_string(self.config, 'owner', required=True),
            default_language=self.config['default_language'],
            context=self.context,
            action_groups=agcs,
            features=self.config.get('features', {}),
            terms=self.config.get('terms', {}),
            pages=pages_from_config(self.config.get('pages', [])),
            **{attr: self.config.get(attr) for attr in instance_attrs},  # type: ignore
            # FIXME: The YAML file seems to specify what's supposed to be in InstanceConfig.lead_title (and other
            # attributes), but not under `instance` but under `pages` for a "page" whose `id' is `home`. It's a mess.
            **self._build_instance_args_from_home_page(),
        )
        self.context.instance = self.instance
        self.instance.yaml_file_path = self.yaml_file_path

        # Deprecated
        # self.load_datasets(self.config.get('datasets', []))

        # Store input and output node configs for each created node, to be used in setup_edges().
        self._input_nodes = {}
        self._output_nodes = {}
        self._subactions = {}
        self._scenario_values = {}

        self.setup_dimensions()
        self.generate_nodes_from_emission_sectors()
        self.setup_global_parameters()
        self.setup_nodes()
        self.setup_actions()
        self.setup_edges()
        self.setup_action_efficiency_pairs()
        self.setup_scenarios()
        self.setup_normalizations()

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
