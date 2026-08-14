"""
Parse a YAML-shaped config dict into an ``InstanceSnapshot`` without building a runtime.

This is the parse half of the loader inversion: it consumes the merged config
dict (the output of ``InstanceYAMLConfig.load_for_entrypoint``) and produces
the same spec objects that ``nodes/spec_export.py`` derives from a fully
initialized runtime. The contract is exact equivalence with the export path —
verified by ``tools/parse_oracle.py`` across all YAML instances.

Constraints: no ``Context``, no database access, no runtime ``Node`` or
``Dataset`` construction. Node *classes* are imported for their metadata
(default units, class-level output metrics, allowed parameters), which is a
pure function of the class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID, uuid3

from kausal_common.i18n.pydantic import TranslatedString

from paths.identifiers import identifier_or_none

from nodes.constants import VALUE_COLUMN, DecisionLevel
from nodes.defs import (
    ActionConfig,
    DatasetPortSpec,
    InputDatasetDef,
    InstanceModelSpec,
    NodeSpec,
    SimpleConfig,
    YearsSpec,
)
from nodes.defs.instance_defs import ActionGroup, DatasetRepoSpec, InstanceFeatures, InstanceMetadata, InstanceTerms
from nodes.defs.node_defs import NodeSpecExtra
from nodes.defs.port_def import InputPortDef, OutputPortDef
from nodes.dimensions import Dimension
from nodes.goals import NodeGoals
from nodes.instance_serialization import (
    DatasetPortSnapshot,
    EdgeSnapshot,
    InstanceSnapshot,
    NodeSnapshot,
)
from nodes.units import Unit, unit_registry
from nodes.visualizations import NodeVisualizations
from nodes.yaml_port_refs import YamlPortReferenceCatalog

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from nodes.defs.transform_def import EdgeTransformOp
    from nodes.node import Node, NodeMetric
    from nodes.scenario import Scenario
    from params import Parameter


class InstanceParseError(Exception):
    pass


def _make_trans_string(
    config: dict[str, Any], attr: str, *, pop: bool = False, required: bool = False
) -> TranslatedString | None:
    """Delegate to the loader's YAML i18n-key convention reader."""
    from nodes.instance_loader import make_trans_string

    if required:
        return make_trans_string(config, attr, pop=pop, required=True)
    return make_trans_string(config, attr, pop=pop, required=False)


def _require_trans_string(config: dict[str, Any], attr: str, *, pop: bool = False) -> TranslatedString:
    val = _make_trans_string(config, attr, pop=pop, required=True)
    assert val is not None
    return val


def _to_ts(val: Any) -> TranslatedString | None:
    if val is None:
        return None
    if isinstance(val, TranslatedString):
        return val
    return TranslatedString(str(val))


# ---------------------------------------------------------------------------
# Node classes: import + metadata (no instantiation)
# ---------------------------------------------------------------------------


def import_node_class(type_path: str, *, is_action: bool) -> type[Node]:
    import importlib

    from nodes.actions.action import ActionNode
    from nodes.node import Node

    if type_path.startswith('nodes.'):
        prefix = None
    else:
        prefix = 'nodes.actions' if is_action else 'nodes'
    parts = type_path.split('.')
    class_name = parts.pop(-1)
    if prefix:
        parts = prefix.split('.') + parts
    mod = importlib.import_module('.'.join(parts))
    klass = getattr(mod, class_name)
    if not issubclass(klass, Node):
        raise InstanceParseError(f'{type_path} is not a Node subclass')
    if is_action != issubclass(klass, ActionNode):
        kind = 'an ActionNode' if is_action else 'a non-action Node'
        raise InstanceParseError(f'{type_path} is not {kind}')
    return klass


# ---------------------------------------------------------------------------
# Per-node parsed state
# ---------------------------------------------------------------------------


@dataclass
class _ParsedEdgeDimension:
    """Mirror of runtime ``EdgeDimension``: group refs expanded to categories."""

    categories: list[str]
    exclude: bool
    flatten: bool


@dataclass
class _ParsedEdge:
    """Mirror of runtime ``Edge`` for the parse pass."""

    from_node: str
    to_node: str
    tags: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    from_dimensions: dict[str, _ParsedEdgeDimension] = field(default_factory=dict)
    to_dimensions: dict[str, _ParsedEdgeDimension] | None = None
    # (from_output_metric_id, to_port_id) pairs, accumulated while building
    # the target node's input ports — mirrors Edge._to_port_ids et al.
    port_pairs: list[tuple[str, UUID]] = field(default_factory=list)

    def replace_to_port_id(self, old: UUID, new: UUID) -> None:
        for idx, (metric_id, port_id) in enumerate(self.port_pairs):
            if port_id == old:
                self.port_pairs[idx] = (metric_id, new)
                return
        raise InstanceParseError(f'Port {old} not found in edge {self.from_node}:{self.to_node}')


@dataclass
class _ParsedNode:
    identifier: str
    config: dict[str, Any]
    node_class: type[Node]
    is_action: bool
    # Populated in the metrics pass:
    output_metrics: dict[str, NodeMetric] = field(default_factory=dict)
    unit: Unit | None = None
    quantity: str | None = None
    class_metric_ids: frozenset[str] = frozenset()
    input_dimensions: list[str] = field(default_factory=list)
    output_dimensions: list[str] = field(default_factory=list)
    internal_dims: set[str] = field(default_factory=set)
    # Populated in the params pass:
    params: list[Parameter] = field(default_factory=list)
    # Populated in the edge/port pass:
    edges: list[_ParsedEdge] = field(default_factory=list)
    input_ports: list[InputPortDef] = field(default_factory=list)
    output_ports: list[OutputPortDef] = field(default_factory=list)
    dataset_defs: list[InputDatasetDef] = field(default_factory=list)
    has_fixed_dataset: bool = False


@dataclass
class _InputPortMultiCandidate:
    port: InputPortDef
    old_port_id: UUID
    edge: _ParsedEdge
    metric: NodeMetric
    group: str
    role: str | None = None


class InstanceConfigParser:
    """One-shot parser: construct with the merged config dict, call ``parse()``."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        instance_uuid: UUID,
        node_uuids: dict[str, UUID] | None = None,
        port_references: YamlPortReferenceCatalog | None = None,
    ) -> None:
        self.config = config
        self.instance_uuid = instance_uuid
        self.node_uuids = node_uuids or {}
        self.port_references = port_references or YamlPortReferenceCatalog()
        self._resolved_node_uuids: dict[str, UUID] = {}
        self.default_language: str = config['default_language']
        self.other_languages: list[str] = config.get('supported_languages', [])
        self._terms = InstanceTerms()
        self.dimensions: dict[str, Dimension] = {}
        self.global_params: dict[str, Parameter] = {}
        self.nodes: dict[str, _ParsedNode] = {}
        # scenario_id -> list of (param_global_id, cleaned value)
        self._scenario_values: dict[str, list[tuple[str, Any]]] = {}

    # -- uuid derivations (must match nodes/spec_export.py) ------------------

    def _uuid_from_identifiers(self, identifiers: Sequence[str]) -> UUID:
        return uuid3(self.instance_uuid, ':'.join(identifiers))

    def _node_uuid(self, identifier: str, authored_uuid: str | UUID | None = None) -> UUID:
        node_uuid = self._resolved_node_uuids.get(identifier)
        if node_uuid is None:
            node_uuid = (
                UUID(str(authored_uuid))
                if authored_uuid is not None
                else self.node_uuids.get(identifier) or self._uuid_from_identifiers([identifier])
            )
            self._resolved_node_uuids[identifier] = node_uuid
        return node_uuid

    # -- top level ------------------------------------------------------------

    def parse(self) -> InstanceSnapshot:
        from kausal_common.i18n.pydantic import set_i18n_context

        with set_i18n_context(self.default_language, self.other_languages):
            return self._parse()

    def _parse(self) -> InstanceSnapshot:
        config = self.config
        self._terms = InstanceTerms.model_validate(config.get('terms', {}))
        metadata = self._parse_metadata()
        self._parse_dimensions()
        self._parse_global_params()

        for nc in self._expanded_emission_sector_configs():
            self._add_node(nc, is_action=False)
        for nc in config.get('nodes', []):
            self._add_node(nc, is_action=False)
        for nc in config.get('actions', []):
            self._add_node(nc, is_action=True)

        for parsed in self.nodes.values():
            self._parse_node_metrics(parsed)
            self._parse_node_dimensions(parsed)
            self._parse_node_datasets(parsed)
            self._parse_node_params(parsed)
            if parsed.is_action:
                self._ensure_enabled_param(parsed)

        self._build_edges()
        for parsed in self.nodes.values():
            parsed.output_ports = self._build_output_ports(parsed)
        for parsed in self.nodes.values():
            parsed.input_ports = self._build_input_ports(parsed)

        spec = self._parse_instance_spec()
        node_snapshots = [self._build_node_snapshot(parsed) for parsed in self.nodes.values()]
        edges = self._build_edge_snapshots()
        dataset_ports = self._build_dataset_port_snapshots()

        # Positions stay unassigned: this is the pre-resolution parse-side
        # snapshot, and the dataset fan-out at sync changes binding cardinality.
        return InstanceSnapshot(
            metadata=metadata,
            spec=spec,
            nodes=node_snapshots,
            bindings=[*edges, *dataset_ports],
        )

    # -- metadata & instance spec ---------------------------------------------

    def _parse_metadata(self) -> InstanceMetadata:
        config = self.config
        name = _require_trans_string(config, 'name')
        owner = _require_trans_string(config, 'owner')
        return InstanceMetadata(
            uuid=self.instance_uuid,
            identifier=config['id'],
            name=name,
            owner=owner,
            lead_title=_make_trans_string(config, 'lead_title'),
            lead_paragraph=_make_trans_string(config, 'lead_paragraph'),
            primary_language=self.default_language,
            other_languages=[lang for lang in self.other_languages if lang != self.default_language],
        )

    def _parse_dimensions(self) -> None:
        for dc in self.config.get('dimensions', []):
            dim = Dimension.from_yaml_config(dict(dc))
            if dim.id in self.dimensions:
                raise InstanceParseError(f'Duplicate dimension {dim.id}')
            self.dimensions[dim.id] = dim

    def _parse_global_params(self) -> None:
        from params.discover import discover_global_parameters

        prototypes = discover_global_parameters()
        for pc_orig in self.config.get('params', []):
            pc = dict(pc_orig)
            param_id = pc.pop('id')
            pc['local_id'] = param_id
            unit_str = pc.get('unit')
            if unit_str is not None:
                pc['unit'] = unit_registry.parse_units(unit_str)
            proto = prototypes.get(param_id)
            if proto is None:
                raise InstanceParseError(f'Unknown global parameter: {param_id}')
            param_val = pc.pop('value', None)
            if 'is_customizable' not in pc:
                pc['is_customizable'] = False
            pc['label'] = _make_trans_string(pc, 'label', pop=True)
            pc['description'] = _make_trans_string(pc, 'description', pop=True)
            param = type(proto)(**pc)
            if param_val is not None:
                param.value = param.clean(param_val)
            self.global_params[param_id] = param

    def _parse_instance_spec(self) -> InstanceModelSpec:
        from nodes.defs.action_def import ImpactOverviewSpec
        from nodes.defs.instance_defs import NormalizationSpec
        from pages.config import pages_from_config

        from .excel_results import InstanceResultExcel

        config = self.config
        target_year = config['target_year']
        years = YearsSpec(
            reference=config.get('reference_year'),
            min_historical=config['minimum_historical_year'],
            max_historical=config.get('maximum_historical_year'),
            target=target_year,
            model_end=config.get('model_end_year', target_year),
        )
        if config.get('reference_year') is None:
            raise InstanceParseError('Reference year must be given for the instance.')

        repo_conf = config.get('dataset_repo')
        dataset_repo = DatasetRepoSpec.model_validate(repo_conf) if repo_conf is not None else None

        agcs: list[ActionGroup] = []
        for idx, agc in enumerate(config.get('action_groups', [])):
            agcs.append(
                ActionGroup(
                    id=agc['id'],
                    name=_require_trans_string(dict(agc), 'name'),
                    color=agc.get('color'),
                    order=idx,
                )
            )

        impact_overviews: list[ImpactOverviewSpec] = []
        seen_overview_ids: set[str] = set()
        rename_map = {
            'effect_node': 'effect_node_id',
            'cost_node': 'cost_node_id',
            'stakeholder_dimension': 'stakeholder_dimension_id',
            'outcome_dimension': 'outcome_dimension_id',
        }
        for aepc in config.get('impact_overviews', []):
            spec_config = dict(aepc)
            for old_name, new_name in rename_map.items():
                if old_name in spec_config and new_name not in spec_config:
                    spec_config[new_name] = spec_config.pop(old_name)
            overview = ImpactOverviewSpec.from_yaml_config(spec_config)
            assert overview.id is not None
            if overview.id in seen_overview_ids:
                raise InstanceParseError(f"Duplicate impact overview id '{overview.id}'")
            seen_overview_ids.add(overview.id)
            impact_overviews.append(overview)

        normalizations: list[NormalizationSpec] = []
        for nc in config.get('normalizations', []):
            spec_config = dict(nc)
            if 'normalizer_node' in spec_config and 'normalizer_node_id' not in spec_config:
                spec_config['normalizer_node_id'] = spec_config.pop('normalizer_node')
            normalizations.append(NormalizationSpec.model_validate(spec_config))

        features = InstanceFeatures.model_validate(config.get('features', {}))
        terms = self._terms
        result_excels = [InstanceResultExcel.from_yaml_config(r).to_spec() for r in config.get('result_excels', [])]

        scenarios = self._parse_scenarios()

        return InstanceModelSpec(
            years=years,
            dataset_repo=dataset_repo,
            dimensions=[dim.model_dump(exclude_none=True) for dim in self.dimensions.values()],
            features=features,
            terms=terms,
            result_excels=result_excels,
            pages=pages_from_config(config.get('pages', [])),
            impact_overviews=impact_overviews,
            normalizations=normalizations,
            params=list(self.global_params.values()),
            action_groups=agcs,
            scenarios=scenarios,
            theme_identifier=config.get('theme_identifier'),
            sample_size=config.get('sample_size', 0),
        )

    # -- scenarios --------------------------------------------------------------

    def _find_param(self, global_id: str) -> Parameter:
        if '.' in global_id:
            node_id, local_id = global_id.rsplit('.', 1)
            parsed = self.nodes.get(node_id)
            if parsed is not None:
                for p in parsed.params:
                    if p.local_id == local_id:
                        return p
            raise InstanceParseError(f'Parameter {global_id} not found')
        param = self.global_params.get(global_id)
        if param is None:
            raise InstanceParseError(f'Parameter {global_id} not found')
        return param

    def _parse_scenarios(self) -> list[Scenario]:  # noqa: C901, PLR0912
        from django.utils.translation import gettext_lazy as _

        from nodes.scenario import Scenario, ScenarioKind

        scenario_confs: list[dict[str, Any]] = self.config.get('scenarios', [])
        if not scenario_confs:
            scenario_confs = [{'id': 'default', 'name': TranslatedString(str(_('Default'))), 'default': True}]

        scenarios: list[Scenario] = []
        default_scenario: Scenario | None = None
        for sc_orig in scenario_confs:
            sc = dict(sc_orig)
            name = _require_trans_string(sc, 'name', pop=True)
            params_config = sc.pop('params', [])
            actual_historical_years = sc.pop('actual_historical_years', None)
            default = sc.pop('default', False)
            scenario_id: str = sc.pop('id')
            kind: ScenarioKind | None = None
            if default:
                kind = ScenarioKind.DEFAULT
            elif scenario_id == 'progress_tracking':
                kind = ScenarioKind.PROGRESS_TRACKING
            elif scenario_id == 'baseline':
                kind = ScenarioKind.BASELINE
            scenario = Scenario(id=scenario_id, name=name, actual_historical_years=actual_historical_years, kind=kind, **sc)
            for pc in params_config:
                param = self._find_param(pc['id'])
                scenario.param_values[pc['id']] = param.clean(pc['value'])
            for global_id, value in self._scenario_values.get(scenario_id, []):
                scenario.param_values[global_id] = value
            # Mirror ActionNode.on_scenario_created: every action's 'enabled'
            # parameter participates in every scenario.
            for parsed_node in self.nodes.values():
                if not parsed_node.is_action:
                    continue
                enabled_gid = f'{parsed_node.identifier}.enabled'
                if enabled_gid not in scenario.param_values:
                    scenario.param_values[enabled_gid] = scenario.all_actions_enabled
            if scenario.default:
                if default_scenario is not None:
                    raise InstanceParseError('Multiple default scenarios')
                default_scenario = scenario
            scenarios.append(scenario)

        if default_scenario is None:
            raise InstanceParseError('Default scenario not defined')

        for global_id, param in self._iter_params_with_global_ids():
            if not param.is_customizable:
                continue
            if global_id in default_scenario.param_values:
                continue
            default_scenario.param_values[global_id] = param.value

        # The loader activates the default scenario at the end of init, so the
        # exported parameter *values* reflect it. Mirror the activation.
        for global_id, value in default_scenario.param_values.items():
            param = self._find_param(global_id)
            param.value = param.clean(value) if value is not None else None

        return scenarios

    def _iter_params_with_global_ids(self) -> Iterator[tuple[str, Parameter]]:
        yield from self.global_params.items()
        for parsed in self.nodes.values():
            for p in parsed.params:
                yield f'{parsed.identifier}.{p.local_id}', p

    # -- emission sectors -------------------------------------------------------

    def _expanded_emission_sector_configs(self) -> list[dict[str, Any]]:
        config = self.config
        sectors = config.get('emission_sectors', [])
        if not sectors:
            return []
        dataset_id = config.get('emission_dataset')
        emission_unit = config.get('emission_unit')
        assert emission_unit is not None
        emission_unit = unit_registry.parse_units(emission_unit)
        dims = config.get('emission_dimensions', [])

        expanded: list[dict[str, Any]] = []
        for ec_orig in sectors:
            ec = dict(ec_orig)
            # setup_validation_graph *overwrites* these on the sector configs
            # before expansion (an authored `type:` on a sector is ignored);
            # reproduce its net effect.
            ec['type'] = 'simple.SectorEmissions'
            ec['unit'] = config.get('emission_unit')
            ec['input_dimensions'] = config.get('emission_dimensions')
            ec['output_dimensions'] = config.get('emission_dimensions')

            parent_id = ec.pop('part_of', None)
            data_col = ec.pop('column', None)
            data_category = ec.pop('category', None)
            unit = ec.pop('unit', emission_unit)
            dim_i = ec.pop('input_dimensions', dims)
            dim_o = ec.pop('output_dimensions', dims)
            if 'name_en' in ec and 'emissions' not in ec['name_en']:
                ec['name_en'] += ' emissions'
            nc = dict(
                output_nodes=[parent_id] if parent_id else [],
                input_dimensions=dim_i,
                output_dimensions=dim_o,
                input_datasets=[
                    dict(
                        id=dataset_id,
                        column=data_col,
                        forecast_from=config.get('emission_forecast_from'),
                        unit=emission_unit,
                    ),
                ]
                if data_col or data_category
                else [],
                unit=unit,
                params=dict(category=data_category) if data_category else [],
                **ec,
            )
            expanded.append(nc)
        return expanded

    # -- nodes --------------------------------------------------------------------

    def _add_node(self, nc: dict[str, Any], *, is_action: bool) -> None:
        node_id = nc['id']
        if node_id in self.nodes:
            raise InstanceParseError(f'Node {node_id} is already configured')
        node_class = import_node_class(nc['type'], is_action=is_action)
        self.nodes[node_id] = _ParsedNode(
            identifier=node_id,
            config=nc,
            node_class=node_class,
            is_action=is_action,
        )

    def _parse_node_metrics(self, parsed: _ParsedNode) -> None:  # noqa: C901, PLR0912, PLR0915
        """Mirror ``make_node`` + ``Node._init_metrics`` metric normalization."""
        from nodes.node import NodeMetric

        config = parsed.config
        node_class = parsed.node_class
        metrics_conf = config.get('output_metrics')
        metrics: dict[str, NodeMetric] | None
        if metrics_conf is not None:
            metrics = {m['id']: NodeMetric.from_config(m) for m in metrics_conf}
            class_metrics_def = cast('dict[str, NodeMetric] | None', getattr(node_class, 'output_metrics', None))
            if class_metrics_def:
                col_to_class_key = {m.column_id: k for k, m in class_metrics_def.items()}
                metrics = {col_to_class_key.get(key, key): metric for key, metric in metrics.items()}
            class_metrics = None
        else:
            metrics = None
            class_metrics = cast('dict[str, NodeMetric] | None', getattr(node_class, 'output_metrics', None))

        unit = config.get('unit')
        if unit is None:
            unit = getattr(node_class, 'default_unit', None)
            if unit is None:
                unit = getattr(node_class, 'unit', None)
            if not unit and not metrics and not class_metrics:
                raise InstanceParseError(f'Node {parsed.identifier} ({node_class.__name__}) has no unit set')
        if unit and not isinstance(unit, Unit):
            unit = unit_registry.parse_units(unit)

        quantity = config.get('quantity')
        if quantity is None:
            quantity = getattr(node_class, 'quantity', None)
            if not quantity and not metrics and not class_metrics:
                raise InstanceParseError(f'Node {parsed.identifier} ({node_class.__name__}) has no quantity set')

        # _init_metrics. Note: for class-level metrics the runtime does a
        # *shallow* dict copy and mutates the shared NodeMetric objects in
        # place (assigning id/column_id/unit); reproduce that exactly.
        if metrics is not None:
            output_metrics = {metric_id: metric.copy() for metric_id, metric in metrics.items()}
        elif class_metrics is not None:
            output_metrics = class_metrics.copy()
        else:
            output_metrics = {}

        if output_metrics:
            from paths.identifiers import validate_identifier

            for met_id, met in output_metrics.items():
                # populate_unit: always derive the parsed unit from default_unit.
                if isinstance(met.default_unit, Unit):
                    met.unit = met.default_unit
                else:
                    met.unit = unit_registry.parse_units(met.default_unit)
                met.id = validate_identifier(met_id, mixed=True)
            if len(output_metrics) == 1:
                metric = next(iter(output_metrics.values()))
                parsed.unit = metric.unit
                parsed.quantity = metric.quantity
                if not metric.column_id:
                    metric.column_id = VALUE_COLUMN
            else:
                parsed.unit = None
                parsed.quantity = None
                for met_id, met in output_metrics.items():
                    if not met.column_id:
                        met.column_id = met_id
        else:
            from nodes.constants import DEFAULT_METRIC

            assert quantity is not None
            assert unit is not None
            parsed.unit = unit
            parsed.quantity = quantity
            output_metrics[DEFAULT_METRIC] = NodeMetric(unit, quantity, id=DEFAULT_METRIC, column_id=VALUE_COLUMN)

        parsed.output_metrics = output_metrics
        class_metrics_attr = getattr(node_class, 'output_metrics', None)
        parsed.class_metric_ids = frozenset(class_metrics_attr.keys()) if isinstance(class_metrics_attr, dict) else frozenset()

    def _parse_node_dimensions(self, parsed: _ParsedNode) -> None:  # noqa: C901
        """Mirror ``Node._init_dimensions`` (class dims + arg dims), skipping internal dims."""
        for direction in ('input', 'output'):
            arg_dims: list[str] | None = parsed.config.get(f'{direction}_dimensions')
            class_dim_ids: list[str] = list(getattr(parsed.node_class, f'{direction}_dimension_ids', []) or [])
            class_dims = getattr(parsed.node_class, f'{direction}_dimensions', None)
            dims: list[str] = []
            if isinstance(class_dims, dict):
                # Class-level dims carry their own Dimension objects (often internal).
                for dim_id, dim in class_dims.items():
                    dims.append(dim_id)
                    if dim.is_internal:
                        parsed.internal_dims.add(dim_id)
            if isinstance(arg_dims, str):
                arg_dims = [arg_dims]
            if arg_dims and class_dim_ids:
                if set(arg_dims) != set(class_dim_ids):
                    raise InstanceParseError(
                        f'Node {parsed.identifier}: invalid dimensions supplied: {arg_dims}; expecting {class_dim_ids}'
                    )
            elif class_dim_ids:
                arg_dims = class_dim_ids
            if arg_dims:
                for dim_id in arg_dims:
                    dim = self.dimensions.get(dim_id)
                    if dim is None:
                        raise InstanceParseError(f'Node {parsed.identifier}: dimension {dim_id} not found')
                    dims.append(dim_id)
                    if dim.is_internal:
                        parsed.internal_dims.add(dim_id)
            setattr(parsed, f'{direction}_dimensions', dims)

    def _parse_node_datasets(self, parsed: _ParsedNode) -> None:
        """Mirror ``_make_node_datasets`` for the binding definitions (no data loading)."""
        config = parsed.config
        ds_config = config.get('input_datasets')
        if ds_config is None:
            ds_config = getattr(parsed.node_class, 'input_datasets', [])

        # Mirror the loader: a processor entry forces interpolation on, a class default
        # yields to an explicit `interpolate:` on the binding.
        class_interpolate = parsed.node_class.interpolates_input_datasets_by_default
        ds_interpolate = False
        idp_confs = config.get('input_dataset_processors', [])
        if idp_confs:
            if len(idp_confs) != 1 or idp_confs[0] != 'LinearInterpolation':
                raise InstanceParseError('Only one LinearInterpolation dataset processor supported')
            ds_interpolate = True

        defs: list[InputDatasetDef] = []
        for ds in ds_config:
            if isinstance(ds, str):
                ds_def = InputDatasetDef(id=ds, interpolate=ds_interpolate or class_interpolate)
            else:
                ds_def = InputDatasetDef.model_validate(ds)
                if ds_interpolate:
                    ds_def.interpolate = True
                elif class_interpolate and 'interpolate' not in ds:
                    ds_def.interpolate = True
            defs.append(ds_def)
        parsed.dataset_defs = defs
        parsed.has_fixed_dataset = 'historical_values' in config or 'forecast_values' in config

    def _parse_node_params(self, parsed: _ParsedNode) -> None:  # noqa: C901, PLR0912
        """Mirror ``_make_node_params``."""
        from params.param import ReferenceParameter

        params = parsed.config.get('params', [])
        if not params:
            return
        if isinstance(params, dict):
            params = [dict(id=param_id, value=value) for param_id, value in params.items()]
        class_allowed: dict[str, Parameter] = {p.local_id: p for p in getattr(parsed.node_class, 'allowed_parameters', [])}
        for pc_orig in params:
            pc = dict(pc_orig)
            param_id = pc.pop('id')
            param_obj = class_allowed.get(param_id)
            if param_obj is None:
                raise InstanceParseError(f'Node {parsed.identifier}: parameter {param_id} not allowed by node class')
            param_class = type(param_obj)

            label = _make_trans_string(pc, 'label', pop=True) or param_obj.label
            ref = pc.pop('ref', None)
            description = _make_trans_string(pc, 'description', pop=True) or param_obj.description
            is_customizable = pc.pop('is_customizable', None)
            scenario_values = pc.pop('values', {})

            if ref is not None:
                target = self.global_params.get(ref)
                if target is None:
                    raise InstanceParseError(
                        f'Node {parsed.identifier}: parameter {param_id} refers to unknown global parameter {ref}'
                    )
                if not isinstance(target, param_class):
                    raise InstanceParseError(
                        f'Node {parsed.identifier} requires parameter of type {param_class}, but {ref} is {type(target)}'
                    )
                ref_param = ReferenceParameter(
                    local_id=param_obj.local_id,
                    label=param_obj.label,
                    target_id=ref,
                )
                parsed.params.append(ref_param)
                continue

            fields = param_obj.model_dump()
            fields.update(pc)
            if description is not None:
                fields['description'] = description
            if label is not None:
                fields['label'] = label
            if is_customizable is not None:
                fields['is_customizable'] = is_customizable
            unit = fields.get('unit')
            if unit is not None and isinstance(unit, str):
                fields['unit'] = unit_registry.parse_units(unit)
            value = fields.pop('value', None)
            param = param_class(**fields)
            if value is not None:
                param.value = param.clean(value)
            parsed.params.append(param)

            global_id = f'{parsed.identifier}.{param.local_id}'
            for scenario_id, sval in scenario_values.items():
                sv = self._scenario_values.setdefault(scenario_id, [])
                sv.append((global_id, param.clean(sval)))

    def _ensure_enabled_param(self, parsed: _ParsedNode) -> None:
        """Mirror ``ActionNode.finalize_init``: every action carries an 'enabled' parameter."""
        from nodes.actions.action import ENABLED_PARAM_ID

        param = next((p for p in parsed.params if p.local_id == ENABLED_PARAM_ID), None)
        if param is None:
            for proto in getattr(parsed.node_class, 'allowed_parameters', []):
                if proto.local_id == ENABLED_PARAM_ID:
                    break
            else:
                raise InstanceParseError(f"Node {parsed.identifier}: 'enabled' is missing from allowed parameters")
            param = proto.copy()
            param.mark_implicit()
            parsed.params.append(param)
        # EnabledParam.set_node applies the instance's custom label, when set.
        enabled_label = self._terms.enabled_label
        if enabled_label:
            param.label = enabled_label
        if param.value is None:
            param.value = param.clean(False)  # noqa: FBT003

    # -- edges & ports -------------------------------------------------------------

    def _parse_edge_dimension(self, dc: dict[str, Any], node_id: str, other_dims: list[str]) -> tuple[str, _ParsedEdgeDimension]:
        """Mirror ``EdgeDimension.from_config``."""
        if 'id' not in dc:
            if len(other_dims) == 1:
                dim_id = other_dims[0]
            else:
                raise InstanceParseError(f'Node {node_id}: dimension id not supplied in edge')
        else:
            dim_id = dc['id']
        dim = self.dimensions.get(dim_id)
        if dim is None:
            raise InstanceParseError(f'Node {node_id}: dimension {dim_id} not found')

        flatten = dc.get('flatten')
        exclude = dc.get('exclude')
        cat_ids = dc.get('categories')
        groups = dc.get('groups')
        if groups is not None:
            if cat_ids is None:
                cat_ids = []
            for gid in groups:
                group_cats = dim.get_cats_for_group(gid)
                cat_ids = cat_ids + [cat.id for cat in group_cats]
        if cat_ids is None:
            cats: list[str] = []
            if flatten not in (None, True) or exclude not in (None, True):
                raise InstanceParseError("When categories are not supplied, you must not supply 'flatten' or 'exclude'")
            flatten = True
            exclude = True
        else:
            for cat_id in cat_ids:
                dim.get(cat_id)  # raises when unknown
            cats = list(cat_ids)
            flatten = bool(flatten)
            exclude = bool(exclude)
        return dim_id, _ParsedEdgeDimension(categories=cats, exclude=exclude, flatten=flatten)

    def _make_edge(self, ec: dict[str, Any] | str, node: _ParsedNode, *, is_output: bool) -> _ParsedEdge:
        if isinstance(ec, str):
            other_id = ec
            ec = {}
        else:
            other_id = ec.get('id')
            if other_id is None:
                raise InstanceParseError(f'Node {node.identifier}: node id not given in edge definition')
        other = self.nodes.get(other_id)
        if other is None:
            raise InstanceParseError(f'Node {node.identifier}: node {other_id} not found')

        output_node, input_node = (other, node) if is_output else (node, other)
        tags = ec.get('tags', [])
        if isinstance(tags, str):
            tags = [tags]

        from_dimensions: dict[str, _ParsedEdgeDimension] = {}
        for dc in ec.get('from_dimensions', []):
            dim_id, ed = self._parse_edge_dimension(dc, node.identifier, input_node.output_dimensions)
            from_dimensions[dim_id] = ed
        to_dimensions: dict[str, _ParsedEdgeDimension] | None = None
        dcs = ec.get('to_dimensions')
        if dcs is not None:
            to_dimensions = {}
            for dc in dcs:
                dim_id, ed = self._parse_edge_dimension(dc, node.identifier, output_node.input_dimensions)
                to_dimensions[dim_id] = ed

        return _ParsedEdge(
            from_node=input_node.identifier,
            to_node=output_node.identifier,
            tags=list(tags),
            metrics=list(ec.get('metrics', []) or []),
            from_dimensions=from_dimensions,
            to_dimensions=to_dimensions,
        )

    def _build_edges(self) -> None:
        """Mirror ``_setup_edges``: same creation order, edge appended to both endpoints."""
        for node in self.nodes.values():
            for ec in node.config.get('output_nodes', []):
                edge = self._make_edge(ec, node, is_output=True)
                node.edges.append(edge)
                self.nodes[edge.to_node if edge.to_node != node.identifier else edge.from_node].edges.append(edge)
            for ec in node.config.get('input_nodes', []):
                edge = self._make_edge(ec, node, is_output=False)
                node.edges.append(edge)
                self.nodes[edge.from_node if edge.from_node != node.identifier else edge.to_node].edges.append(edge)

    def _build_output_ports(self, parsed: _ParsedNode) -> list[OutputPortDef]:
        """Mirror ``_export_output_ports``."""
        role_by_metric_id = {
            declaration.identifier: declaration.role for declaration in parsed.node_class.output_port_declarations
        }
        ports: list[OutputPortDef] = []
        for metric_id, metric in parsed.output_metrics.items():
            assert metric.unit is not None
            fallback_id = self._uuid_from_identifiers([parsed.identifier, metric_id])
            port = OutputPortDef(
                id=self.port_references.output_port_id(
                    self._node_uuid(parsed.identifier, parsed.config.get('uuid')),
                    (str(metric_id), str(metric.column_id)),
                    fallback_id,
                ),
                identifier=identifier_or_none(metric_id),
                role=role_by_metric_id.get(metric_id),
                label=_to_ts(metric.label),
                unit=metric.unit,
                quantity=metric.quantity or None,
                column_id=metric.column_id,
                is_editable=metric_id not in parsed.class_metric_ids,
            )
            port._metric_id = metric_id
            ports.append(port)
        return ports

    # -- input ports (edges + datasets) ----------------------------------------------

    def _dataset_binding_columns(self, parsed: _ParsedNode, ds_def: InputDatasetDef) -> list[str]:
        """Mirror ``_dataset_binding_columns_for_node``."""
        if ds_def.column is not None:
            return [ds_def.column]
        columns: list[str] = []
        seen: set[str] = set()
        for metric in parsed.output_metrics.values():
            if metric.column_id is None:
                continue
            column = str(metric.column_id)
            if column not in seen:
                columns.append(column)
                seen.add(column)
        return columns

    def _metric_for_column(self, parsed: _ParsedNode, column: str) -> NodeMetric | None:
        for metric in parsed.output_metrics.values():
            if str(metric.column_id) == column:
                return metric
        return None

    def _build_dataset_input_ports(self, parsed: _ParsedNode) -> list[InputPortDef]:
        """Mirror ``_export_dataset_input_ports``."""
        ports: list[InputPortDef] = []
        for idx, ds_def in enumerate(parsed.dataset_defs):
            for column in self._dataset_binding_columns(parsed, ds_def):
                metric = self._metric_for_column(parsed, column)
                identifier = None if column == VALUE_COLUMN else identifier_or_none(column)
                fallback_id = self._uuid_from_identifiers([parsed.identifier, 'dataset', str(idx), column])
                ports.append(
                    InputPortDef(
                        id=self.port_references.dataset_port_id(
                            self._node_uuid(parsed.identifier, parsed.config.get('uuid')),
                            ds_def.id,
                            idx,
                            column,
                            fallback_id,
                        ),
                        identifier=identifier,
                        unit=metric.unit if metric is not None else ds_def.unit,
                        quantity=metric.quantity if metric is not None else None,
                    )
                )
        return ports

    def _effective_input_dimension_ids(self, parsed: _ParsedNode, edge: _ParsedEdge) -> tuple[str, ...]:
        if edge.to_dimensions is not None:
            return tuple(edge.to_dimensions.keys())
        return tuple(parsed.input_dimensions)

    def _multiplicity_hint(self, parsed: _ParsedNode, edge: _ParsedEdge) -> tuple[str, str] | None:
        """
        Mirror ``input_port_multiplicity_hint`` from class metadata: ``(group, role)``.

        The class answers whether it has an additive multiport, so that adding another
        class with one does not mean remembering to name it here too.
        """
        declaration = parsed.node_class.additive_multiport_declaration(edge.tags)
        if declaration is None:
            return None
        return str(declaration.instance_identifier), str(declaration.role)

    def _is_compatible_unit(self, unit_a: Unit | None, unit_b: Unit | None, node_id: str) -> bool:
        assert unit_a is not None, f'Unit is missing in node {node_id}. Is it multimetric?'
        assert unit_b is not None, f'Unit {unit_b} is missing when comparing to node {node_id}'
        return unit_a.dimensionality == unit_b.dimensionality

    def _is_multi_candidate_group_compatible(self, parsed: _ParsedNode, candidates: list[_InputPortMultiCandidate]) -> bool:
        """Mirror ``_is_multi_candidate_group_compatible`` (sans logging)."""
        if not candidates:
            return False
        first = candidates[0]
        expected_dims = self._effective_input_dimension_ids(parsed, first.edge)
        expected_unit = first.metric.unit
        expected_quantity = first.metric.quantity
        if parsed.unit is not None and not self._is_compatible_unit(expected_unit, parsed.unit, parsed.identifier):
            return False
        for candidate in candidates[1:]:
            dims = self._effective_input_dimension_ids(parsed, candidate.edge)
            if set(dims) != set(expected_dims):
                return False
            if candidate.metric.quantity != expected_quantity:
                return False
            if not self._is_compatible_unit(candidate.metric.unit, expected_unit, parsed.identifier):
                return False
            if parsed.unit is not None and not self._is_compatible_unit(candidate.metric.unit, parsed.unit, parsed.identifier):
                return False
        return True

    def _apply_multi_hints(
        self, parsed: _ParsedNode, ports: list[InputPortDef], candidates: list[_InputPortMultiCandidate]
    ) -> None:
        """Mirror ``_apply_input_port_multi_hints``."""
        by_group: dict[str, list[_InputPortMultiCandidate]] = {}
        for candidate in candidates:
            by_group.setdefault(candidate.group, []).append(candidate)

        for group, group_candidates in by_group.items():
            if not self._is_multi_candidate_group_compatible(parsed, group_candidates):
                continue
            first = group_candidates[0]
            fallback_id = self._uuid_from_identifiers([parsed.identifier, 'input-group', group])
            group_port_id = self.port_references.input_role_id(
                self._node_uuid(parsed.identifier, parsed.config.get('uuid')),
                group,
                fallback_id,
            )
            group_dimensions = list(self._effective_input_dimension_ids(parsed, first.edge))

            first.port.id = group_port_id
            first.port.identifier = identifier_or_none(group)
            first.port.role = identifier_or_none(first.role) if first.role is not None else None
            first.port.multi = True
            first.port.quantity = first.metric.quantity
            first.port.unit = parsed.unit or first.metric.unit
            first.port.required_dimensions = group_dimensions
            first.port.supported_dimensions = group_dimensions

            ports_to_remove = {candidate.old_port_id for candidate in group_candidates[1:]}
            for candidate in group_candidates:
                candidate.edge.replace_to_port_id(candidate.old_port_id, group_port_id)
            ports[:] = [port for port in ports if port is first.port or port.id not in ports_to_remove]

    def _build_input_ports(self, parsed: _ParsedNode) -> list[InputPortDef]:  # noqa: C901
        """Mirror ``_export_input_ports``."""
        from collections import Counter

        ports = self._build_dataset_input_ports(parsed)
        multi_candidates: list[_InputPortMultiCandidate] = []

        for edge in parsed.edges:
            if edge.to_node != parsed.identifier:
                continue
            from_parsed = self.nodes[edge.from_node]
            if not edge.metrics:
                edge_metric_ids = [metric.column_id for metric in from_parsed.output_metrics.values()]
            else:
                edge_metric_ids = edge.metrics
            seen_metric_ids: set[str] = set()
            for edge_metric_id in edge_metric_ids:
                metrics_by_column_id = {m.column_id: m for m in from_parsed.output_metrics.values()}
                from_metric = from_parsed.output_metrics.get(edge_metric_id)
                if from_metric is None:
                    from_metric = metrics_by_column_id.get(edge_metric_id)
                if from_metric is None:
                    raise InstanceParseError(f'Metric {edge_metric_id} not found in {from_parsed.identifier}')
                assert from_metric.id not in seen_metric_ids
                seen_metric_ids.add(from_metric.id)
                fallback_id = self._uuid_from_identifiers([from_parsed.identifier, parsed.identifier, 'edge', from_metric.id])
                from_port = next(port for port in from_parsed.output_ports if port._metric_id == from_metric.id)
                port_id = self.port_references.edge_port_id(
                    self._node_uuid(parsed.identifier, parsed.config.get('uuid')),
                    self._node_uuid(from_parsed.identifier, from_parsed.config.get('uuid')),
                    from_port.id,
                    fallback_id,
                )
                edge.port_pairs.append((from_metric.id, port_id))
                if len(from_parsed.output_metrics) > 1:
                    port_identifier = identifier_or_none(f'{from_parsed.identifier}_{from_metric.id}')
                else:
                    port_identifier = identifier_or_none(from_parsed.identifier)
                port = InputPortDef(
                    id=port_id,
                    identifier=port_identifier,
                    quantity=from_metric.quantity,
                    unit=from_metric.unit,
                    required_dimensions=[
                        dim_id for dim_id, dimension in (edge.to_dimensions or {}).items() if not dimension.categories
                    ],
                )
                hint = self._multiplicity_hint(parsed, edge)
                if hint is not None:
                    group, role = hint
                    multi_candidates.append(
                        _InputPortMultiCandidate(
                            port=port, old_port_id=port_id, edge=edge, metric=from_metric, group=group, role=role
                        )
                    )
                port._from_node = edge.from_node
                port._edge_metric_id = from_metric.id
                ports.append(port)
        self._apply_multi_hints(parsed, ports, multi_candidates)

        counts = Counter(port.identifier for port in ports if port.identifier is not None)
        duplicates = {identifier for identifier, count in counts.items() if count > 1}
        for port in ports:
            if port.identifier in duplicates:
                port.identifier = None
        return ports

    # -- snapshot assembly -------------------------------------------------------------

    def _parse_type_config(self, parsed: _ParsedNode) -> ActionConfig | SimpleConfig:
        """Mirror ``_export_type_config``."""
        kls = parsed.node_class
        node_class = f'{kls.__module__}.{kls.__qualname__}'
        if not parsed.is_action:
            return SimpleConfig(node_class=node_class)

        config = parsed.config
        # The class default applies when the config doesn't override it —
        # export reads node.decision_level, which is always set.
        decision_level: DecisionLevel | None = getattr(parsed.node_class, 'decision_level', None)
        dl_conf = config.get('decision_level')
        if dl_conf is not None:
            for name, val in DecisionLevel.__members__.items():
                if dl_conf == name.lower():
                    decision_level = val
                    break
            else:
                raise InstanceParseError(f'Node {parsed.identifier}: invalid decision level {dl_conf}')
        group = config.get('group')
        if group is not None:
            group_ids = {agc['id'] for agc in self.config.get('action_groups', [])}
            if group not in group_ids:
                raise InstanceParseError(f"Node {parsed.identifier}: action group '{group}' not found")
        no_effect_value = config.get('no_effect_value')
        if no_effect_value is None:
            no_effect_value = getattr(parsed.node_class, 'no_effect_value', None)
        return ActionConfig(
            decision_level=decision_level,
            group=group,
            parent=config.get('parent'),
            no_effect_value=no_effect_value,
            node_class=node_class,
        )

    def _parse_node_goals(self, parsed: _ParsedNode) -> NodeGoals:
        config = parsed.config
        goals = config.get('goals')
        if goals is not None:
            validated = NodeGoals.model_validate(goals)
        elif config.get('target_year_goal') is not None:
            validated = NodeGoals.model_validate(
                [
                    dict(
                        values=[dict(year=self.config['target_year'], value=config['target_year_goal'])],
                        is_main_goal=config.get('is_outcome', False),
                    )
                ],
            )
        else:
            return NodeGoals()
        return validated

    def _parse_node_extra(self, parsed: _ParsedNode) -> NodeSpecExtra:
        """Mirror ``_export_node_extra``."""
        config = parsed.config
        historical_values = config.get('historical_values') or None
        forecast_values = config.get('forecast_values') or None
        # The export path derives processors from the dataset *instances*
        # (any with interpolate=True). A config-level processor on a node
        # with no datasets is therefore dropped — reproduce that.
        processors: list[str] = []
        has_idp = bool(config.get('input_dataset_processors'))
        if not parsed.node_class.interpolates_input_datasets_by_default and (
            any(ds_def.interpolate for ds_def in parsed.dataset_defs) or (parsed.has_fixed_dataset and has_idp)
        ):
            processors = ['LinearInterpolation']
        tags = config.get('tags')
        if isinstance(tags, str):
            tags = [tags]
        return NodeSpecExtra(
            historical_values=historical_values,
            forecast_values=forecast_values,
            input_dataset_processors=processors,
            tags=list(tags) if tags else [],
        )

    def _parse_node_visualizations(self, parsed: _ParsedNode) -> NodeVisualizations:
        viz_config = parsed.config.get('visualizations')
        if not viz_config:
            return NodeVisualizations()
        viz = NodeVisualizations.model_validate(viz_config)
        self._assign_visualization_ids(viz.root, parsed.identifier, [0])
        return viz

    def _assign_visualization_ids(self, entries: list[Any], node_id: str, counter: list[int]) -> None:
        """
        Mirror ``VisualizationEntry.set_id``: replace 'auto' ids with node_id:N.

        The runtime assigns these during validation with a context; Pydantic
        validates nested models bottom-up, so the numbering is post-order.
        """
        from nodes.visualizations import AUTO_ID

        for entry in entries:
            children = getattr(entry, 'children', None)
            if children:
                self._assign_visualization_ids(children, node_id, counter)
            if entry.id == AUTO_ID:
                entry.id = f'{node_id}:{counter[0]}'
                counter[0] += 1

    def _build_node_snapshot(self, parsed: _ParsedNode) -> NodeSnapshot:
        """Mirror ``export_node_spec`` (with ``node.db_obj`` unset, as during sync)."""
        config = parsed.config
        type_config = self._parse_type_config(parsed)
        name = _make_trans_string(config, 'name')
        short_name = _make_trans_string(config, 'short_name')
        description = _make_trans_string(config, 'description')

        uuid = self._node_uuid(parsed.identifier, config.get('uuid'))
        spec = NodeSpec(
            type_config=type_config,
            input_ports=parsed.input_ports,
            output_ports=parsed.output_ports,
            input_dimensions=[d for d in parsed.input_dimensions if d not in parsed.internal_dims],
            output_dimensions=[d for d in parsed.output_dimensions if d not in parsed.internal_dims],
            params=cast('list[Any]', [param for param in parsed.params if not param.is_implicit]),
            goals=self._parse_node_goals(parsed),
            visualizations=self._parse_node_visualizations(parsed),
            allow_nulls=config.get('allow_nulls', False),
            node_group=config.get('node_group'),
            is_outcome=config.get('is_outcome', False),
            minimum_year=config.get('minimum_year'),
            extra=self._parse_node_extra(parsed),
        )
        return NodeSnapshot(
            uuid=uuid,
            identifier=parsed.identifier,
            name=_to_ts(name),
            short_name=_to_ts(short_name),
            short_description=_to_ts(description),
            color=config.get('color') or '',
            order=config.get('order'),
            is_visible=config.get('is_visible', True),
            spec=spec,
        )

    def _build_edge_snapshots(self) -> list[EdgeSnapshot]:
        """Mirror ``_update_edges`` + ``edge_to_transforms``."""
        snapshots: list[EdgeSnapshot] = []
        for node in self.nodes.values():
            for edge in node.edges:
                if edge.from_node != node.identifier:
                    continue
                if not edge.port_pairs:
                    raise InstanceParseError(f'Edge {edge.from_node}:{edge.to_node} has no port pairs')
                from_ports = self.nodes[edge.from_node].output_ports
                for from_metric_id, to_port_id in edge.port_pairs:
                    for from_port in from_ports:
                        if from_port._metric_id == from_metric_id:
                            break
                    else:
                        raise InstanceParseError(
                            f'No output port for edge {edge.from_node}:{edge.to_node} metric {from_metric_id}'
                        )
                    snapshots.append(
                        EdgeSnapshot(
                            from_node=self._node_uuid(edge.from_node),
                            to_node=self._node_uuid(edge.to_node),
                            from_port=from_port.id,
                            to_port=to_port_id,
                            transformations=self._edge_to_transforms(edge),
                            tags=list(edge.tags) if edge.tags else [],
                        )
                    )
        return snapshots

    def _edge_to_transforms(self, edge: _ParsedEdge) -> list[EdgeTransformOp]:
        """Mirror ``edge_to_transforms``."""
        from nodes.defs.transform_def import AssignDimensionOp, FilterDimensionOp

        transforms: list[EdgeTransformOp] = []
        for dim_id, ed in edge.from_dimensions.items():
            transforms.append(
                FilterDimensionOp(
                    dimension=dim_id,
                    categories=list(ed.categories),
                    flatten=ed.flatten,
                    exclude=ed.exclude,
                )
            )
        if edge.to_dimensions:
            for dim_id, ed in edge.to_dimensions.items():
                if not ed.categories:
                    continue
                if len(ed.categories) != 1:
                    raise InstanceParseError(
                        f'to_dimensions can have only one category for now (got {len(ed.categories)} for {dim_id})'
                    )
                transforms.append(AssignDimensionOp(dimension=dim_id, category=ed.categories[0]))
        return transforms

    def _build_dataset_port_snapshots(self) -> list[DatasetPortSnapshot]:
        """
        Mirror ``_update_dataset_ports``, sans DB metric resolution.

        ``metric`` is set to the node-side port column; the sync write half
        re-pairs it against the dataset schema's metrics (which requires the
        DB) exactly as ``_pair_schema_metrics_to_columns`` does today.
        """
        snapshots: list[DatasetPortSnapshot] = []
        for node in self.nodes.values():
            for idx, ds_def in enumerate(node.dataset_defs):
                # The export path builds the spec from the runtime dataset
                # instance, which does not carry the binding's authored
                # output_dimensions — the field is slated for removal (see
                # docs/architecture/dimension-constraints.md). Mirror the drop.
                spec = DatasetPortSpec.from_input_dataset(ds_def.model_copy(update={'output_dimensions': None}))
                snapshots.extend(
                    DatasetPortSnapshot(
                        node=self._node_uuid(node.identifier),
                        dataset=ds_def.id,
                        port_id=self.port_references.dataset_port_id(
                            self._node_uuid(node.identifier, node.config.get('uuid')),
                            ds_def.id,
                            idx,
                            column,
                            self._uuid_from_identifiers([node.identifier, 'dataset', str(idx), column]),
                        ),
                        metric=column,
                        dataset_index=idx,
                        spec=spec,
                    )
                    for column in self._dataset_binding_columns(node, ds_def)
                )
        return snapshots


def parse_instance_snapshot(
    config: dict[str, Any],
    *,
    instance_uuid: UUID,
    node_uuids: dict[str, UUID] | None = None,
    port_references: YamlPortReferenceCatalog | None = None,
) -> InstanceSnapshot:
    """Parse a merged YAML config dict into an InstanceSnapshot without building a runtime."""
    parser = InstanceConfigParser(
        config,
        instance_uuid=instance_uuid,
        node_uuids=node_uuids,
        port_references=port_references,
    )
    return parser.parse()
