import re
import uuid
from dataclasses import dataclass, replace
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlparse
from uuid import uuid4

from django.db import models, transaction
from django.db.models import QuerySet
from django.http import HttpRequest
from django.utils import timezone

import sentry_sdk
from loguru import logger

from kausal_common.const import WILDCARD_DOMAINS_HEADER
from kausal_common.models.modification_tracking import UserModifiableModel
from kausal_common.models.types import ModelManager
from kausal_common.models.uuid import UUIDIdentifiedModel
from kausal_common.users import user_or_none

from paths.types import CacheablePathsModel, PathsModel, PathsQuerySet

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from rich.repr import RichReprResult

    from kausal_common.models.types import FK, M2M, QS, OneToOne, RevMany
    from kausal_common.users import UserOrAnon

    from paths.schema_context import PathsGraphQLContext

    from frameworks.datasets import FrameworkMeasureDVCDataset2
    from frameworks.object_cache import FrameworkConfigCacheData  # noqa: F401
    from frameworks.permissions import (
        FrameworkConfigPermissionPolicy,
    )
    from nodes.defs.instance_defs import InstanceModelSpec, YearsSpec
    from nodes.gpc import DatasetNode
    from nodes.instance import Instance
    from nodes.instance_serialization import InstanceSnapshot
    from nodes.models import InstanceConfig
    from nodes.node import Node

    type ViewURLRequest = HttpRequest | PathsGraphQLContext


from .framework import Framework, FrameworkDimensionCategory
from .measures import (
    DefaultValueScaling,
    Measure,
    MeasureDataPoint,
    MeasureDataPointQuerySet,
    MeasureTemplate,
    MeasureTemplateDefaultDataPoint,
)


@dataclass
class NodeDimensionSelection:
    node_id: str
    dimensions: dict[str, str] | None
    dataset_index: int | None = None
    """
    Index of the ``city_data`` binding this measure was found through, if it was.

    Values come from the node's output either way. What the index records is *how* the
    measure was matched -- ``None`` for the legacy path, where the node is a thin wrapper
    over one dataset and its class is the whole story -- which is what tells the resolver
    whether the node's unit can be trusted to be the client's.
    """

    metric_col: str | None = None
    """
    Output column holding this measure's series, when the node emits more than one.

    A multi-metric node renames ``Value`` to the metric column, so there is nothing for
    the resolver to read under the usual name. ``None`` means the node emits a single
    metric and ``Value`` is it.
    """

    binding_role: str | None = None
    """
    Which end of the series the binding this measure was found through carries.

    ``'historical'``, ``'goal'``, or ``None`` for an untagged binding and for the legacy
    path. Recorded here rather than re-read from ``input_dataset_instances`` at use time
    because ``_prefer_the_full_trajectory`` may since have moved ``node_id``, which leaves
    ``dataset_index`` pointing into a different node's binding list.
    """


def create_random_token():
    return uuid.uuid4().hex


def filter_viewable_by[QS: QuerySet[PathsModel]](qs: QS, user: UserOrAnon) -> QS:
    model = qs.model
    pp = model.permission_policy()
    return qs.filter(id__in=pp.instances_user_has_permission_for(user, 'view'))


class FrameworkConfigQuerySet(PathsQuerySet['FrameworkConfig']):
    pass


if TYPE_CHECKING:

    class FrameworkConfigManager(ModelManager['FrameworkConfig', FrameworkConfigQuerySet]):
        """Model manager for FrameworkConfig."""

else:
    FrameworkConfigManager = ModelManager.from_queryset(FrameworkConfigQuerySet)


class FrameworkConfig(CacheablePathsModel['FrameworkConfigCacheData'], UserModifiableModel, UUIDIdentifiedModel, models.Model):
    """
    Represents a configuration of a Framework for a specific instance.

    This model links a Framework to an InstanceConfig, allowing for customization
    of framework settings for each organization or instance. Model year boundaries
    are owned by ``InstanceConfig.spec``.
    """

    framework: FK[Framework] = models.ForeignKey(Framework, on_delete=models.CASCADE, related_name='configs')
    instance_config: OneToOne[InstanceConfig] = models.OneToOneField(
        'nodes.InstanceConfig',
        on_delete=models.CASCADE,
        related_name='framework_config',
    )
    organization_name = models.CharField(max_length=200, blank=True, null=True)
    organization_identifier = models.CharField(max_length=200, blank=True, null=True)
    organization_slug = models.CharField(max_length=200, blank=True, null=True)
    categories: M2M[FrameworkDimensionCategory, Any] = models.ManyToManyField(FrameworkDimensionCategory)
    extra = models.JSONField(default=dict, blank=True)
    token = models.CharField(max_length=50, default=create_random_token)

    objects: ClassVar[FrameworkConfigManager] = FrameworkConfigManager()

    instance_config_id: int
    framework_id: int
    measures: RevMany[Measure]
    public_fields: ClassVar = [
        'framework',
        'organization_name',
        'uuid',
        'instance_config',
        'extra',
    ]

    class Meta:
        ordering = ['framework', 'instance_config']
        constraints = [
            models.UniqueConstraint(fields=['framework', 'instance_config'], name='unique_framework_instance'),
        ]

    def __str__(self):
        return f'{self.framework.name}: {self.instance_config.name}'

    def __rich_repr__(self) -> RichReprResult:
        yield 'id', self.pk
        yield 'framework', self.framework.identifier
        yield 'instance', self.instance_config.identifier
        yield 'nr_measures', len(self.measures.all())

    @property
    def instance_years(self) -> YearsSpec:
        """Read persisted instance-owned years without refreshing the shared YAML spec."""
        spec = self.instance_config.spec
        if spec is None:
            raise ValueError(f'Framework instance {self.instance_config.identifier} has no persisted spec')
        return spec.years

    @property
    def reference_year(self) -> int:
        reference_year = self.instance_years.reference
        if reference_year is None:
            raise ValueError(f'Framework instance {self.instance_config.identifier} has no reference year')
        return reference_year

    @classmethod
    def permission_policy(cls) -> FrameworkConfigPermissionPolicy:
        from frameworks.permissions import FrameworkConfigPermissionPolicy

        return FrameworkConfigPermissionPolicy()

    @classmethod
    @transaction.atomic
    def create_instance(
        cls,
        framework: Framework,
        instance_identifier: str,
        org_name: str,
        baseline_year: int,
        uuid: uuid.UUID | None = None,
        target_year: int | None = None,
        user: UserOrAnon | None = None,
    ) -> FrameworkConfig:
        from nodes.defs import InstanceModelSpec
        from nodes.models import InstanceConfig, make_minimal_instance_spec
        from orgs.models import Organization

        # Create new organization for instance
        org = Organization.objects.get(name='NetZeroCities')

        uuid = uuid or uuid4()
        # Identity metadata lives on the columns; the computation spec is
        # populated from the framework YAML further below, once the framework
        # link exists. It's left null here rather than set to an empty spec, so
        # readers of the stored spec don't see a stale, theme-less default.
        #
        # The name is stored bare; `InstanceType.site_title` is what prefixes it
        # with the framework name for display.
        ic = InstanceConfig.objects.create(
            name=org_name,
            identifier=instance_identifier,
            primary_language='en',
            other_languages=[],
            organization=org,
            uuid=uuid,
            spec=None,
        )

        pp = cls.permission_policy()
        if pp.user_is_authenticated(user):
            extra = cls.permission_policy().get_create_defaults(user, framework)
        else:
            extra = {}
        fc = cls.objects.create(
            framework=framework,
            instance_config=ic,
            organization_name=org_name,
            uuid=uuid,
            created_by=user_or_none(user),
            **extra,
        )
        if ic.get_yaml_config_entrypoint() is None:
            ic.spec = InstanceModelSpec()
            ic.save(update_fields=['spec'])
        else:
            ic.ensure_spec()
        year_updates: dict[str, int] = {'reference': baseline_year}
        if target_year is not None:
            year_updates['target'] = target_year
        elif ic.spec is not None and ic.spec.years.target is None:
            default_target_year = framework.defaults.target_year.default
            if default_target_year is not None:
                year_updates['target'] = default_target_year
        ic.update_years(**year_updates)
        if pp.user_is_authenticated(user):
            pp.realm_admin_role.assign_user(ic, user)

        if fc.get_view_url() is not None:
            ic.sync_nodes()
            ic.create_default_content()
            fc.setup_instance_pages()

            # Persist the effective minimal spec, including framework-derived
            # historical boundaries, after the model has been initialized.
            ic.spec = make_minimal_instance_spec(ic.get_instance())
            ic.save(update_fields=['spec'])

        return fc

    def setup_instance_pages(self) -> None:
        """Configure root-page menu state and ActionListPage footer state after default content creation."""
        from pages.models import ActionListPage, PathsPage

        ic = self.instance_config
        root_page = ic.root_page
        assert root_page is not None
        if self.framework.identifier == 'nzc' and isinstance(root_page, PathsPage):
            root_page.show_in_menus = True
            root_page.menu_label = 'Home'
            root_page.save()
        for alp in root_page.get_descendants().type(ActionListPage).specific():
            assert isinstance(alp, ActionListPage)
            alp.show_in_footer = False
            alp.save()

    def _get_default_value_multiplier(self, measure_template: MeasureTemplate) -> float:
        if measure_template.default_value_scaling is None:
            return 1.0
        if measure_template.default_value_scaling == DefaultValueScaling.POPULATION:
            create_context = (self.extra or {}).get('create_context') or {}
            population = create_context.get('population')
            if population is None:
                msg = f'Population is required for default value scaling on {measure_template.uuid}'
                raise ValueError(msg)
            return float(population)
        msg = f'Unsupported default value scaling: {measure_template.default_value_scaling}'
        raise ValueError(msg)

    def _select_default_data_points(
        self,
        *,
        only_year: int | None = None,
    ) -> tuple[
        dict[tuple[int, int], tuple[MeasureTemplateDefaultDataPoint, int]],
        set[int],
        dict[int, MeasureTemplate],
    ]:
        category_ids = set(self.categories.values_list('pk', flat=True))
        default_data_points = (
            MeasureTemplateDefaultDataPoint.objects
            .filter(template__section__framework=self.framework)
            .select_related('template')
            .prefetch_related('categories')
            .order_by('template_id', 'year')
        )

        selected_defaults: dict[tuple[int, int], tuple[MeasureTemplateDefaultDataPoint, int]] = {}
        affected_template_ids: set[int] = set()
        templates_by_id: dict[int, MeasureTemplate] = {}

        for default_data_point in default_data_points:
            measure_template = default_data_point.template
            templates_by_id[measure_template.pk] = measure_template
            affected_template_ids.add(measure_template.pk)
            default_category_ids = {cat.pk for cat in default_data_point.categories.all()}
            if not default_category_ids.issubset(category_ids):
                continue
            if only_year is not None and default_data_point.year != only_year:
                continue

            key = (measure_template.pk, default_data_point.year)
            specificity = len(default_category_ids)
            previous = selected_defaults.get(key)
            if previous is not None and previous[1] >= specificity:
                if previous[1] == specificity:
                    logger.warning(
                        'Duplicate equally specific default datapoint for template '
                        + f'{measure_template.uuid} year {default_data_point.year} in framework {self.framework.identifier}',
                    )
                continue
            selected_defaults[key] = (default_data_point, specificity)

        return selected_defaults, affected_template_ids, templates_by_id

    def _ensure_measures_for_templates(self, affected_template_ids: set[int]) -> dict[int, Measure]:
        measures_qs = self.measures.filter(measure_template_id__in=affected_template_ids)
        measure_by_template_id = {m.measure_template_id: m for m in measures_qs}
        new_measures = [
            Measure(framework_config=self, measure_template_id=template_id)
            for template_id in affected_template_ids
            if template_id not in measure_by_template_id
        ]
        if new_measures:
            Measure.objects.bulk_create(new_measures)
            measure_by_template_id = {
                m.measure_template_id: m for m in self.measures.filter(measure_template_id__in=affected_template_ids)
            }
        return measure_by_template_id

    def _delete_non_matching_default_data_points(self, only_year: int) -> int:
        deleted_count, _ = (
            MeasureDataPoint.objects
            .filter(
                measure__framework_config=self,
                default_value__isnull=False,
                value__isnull=True,
            )
            .exclude(year=only_year)
            .delete()
        )
        return deleted_count

    def _get_measure_default_data_points(
        self,
        affected_template_ids: set[int],
        *,
        only_year: int | None = None,
    ) -> dict[tuple[int, int], MeasureDataPoint]:
        qs = MeasureDataPoint.objects.filter(
            measure__framework_config=self,
            measure__measure_template_id__in=affected_template_ids,
        )
        if only_year is not None:
            qs = qs.filter(year=only_year)
        return {(dp.measure.measure_template_id, dp.year): dp for dp in qs.select_related('measure')}

    @staticmethod
    def _reset_default_values(existing_dps: dict[tuple[int, int], MeasureDataPoint]) -> list[MeasureDataPoint]:
        update_dps: list[MeasureDataPoint] = []
        for dp in existing_dps.values():
            if dp.default_value is None and dp.probable_lower_bound is None and dp.probable_upper_bound is None:
                continue
            dp.default_value = None
            dp.probable_lower_bound = None
            dp.probable_upper_bound = None
            update_dps.append(dp)
        return update_dps

    @transaction.atomic
    def populate_measure_defaults(self, *, only_year: int | None = None) -> int:
        selected_defaults, affected_template_ids, templates_by_id = self._select_default_data_points(only_year=only_year)

        if only_year is not None:
            self._delete_non_matching_default_data_points(only_year)

        if not affected_template_ids:
            return 0

        measure_by_template_id = self._ensure_measures_for_templates(affected_template_ids)
        existing_dps = self._get_measure_default_data_points(affected_template_ids, only_year=only_year)
        update_dps = self._reset_default_values(existing_dps)

        new_dps: list[MeasureDataPoint] = []
        for (template_id, year), (default_data_point, _specificity) in selected_defaults.items():
            measure_template = templates_by_id[template_id]
            multiplier = self._get_default_value_multiplier(measure_template)
            dp = existing_dps.get((template_id, year))
            if dp is None:
                dp = MeasureDataPoint(measure=measure_by_template_id[template_id], year=year)
                new_dps.append(dp)
            elif dp not in update_dps:
                update_dps.append(dp)

            dp.default_value = default_data_point.value * multiplier
            dp.probable_lower_bound = (
                None if default_data_point.probable_lower_bound is None else default_data_point.probable_lower_bound * multiplier
            )
            dp.probable_upper_bound = (
                None if default_data_point.probable_upper_bound is None else default_data_point.probable_upper_bound * multiplier
            )

        if new_dps:
            MeasureDataPoint.objects.bulk_create(new_dps)
        if update_dps:
            MeasureDataPoint.objects.bulk_update(
                update_dps,
                fields=['default_value', 'probable_lower_bound', 'probable_upper_bound'],
            )
        return len(selected_defaults)

    def apply_spec_overrides(self, spec: InstanceModelSpec) -> InstanceModelSpec:
        """Return the shared framework spec with instance-owned and observed years applied."""
        from django.db.models import Max, Min

        instance_years = self.instance_config.ensure_spec().years
        reference_year = instance_years.reference
        mdp_years = MeasureDataPoint.objects.filter(measure__framework_config=self).aggregate(
            min_year=Min('year'),
            max_year=Max('year'),
        )
        years = spec.years.model_copy(
            update={
                'reference': reference_year,
                'min_historical': mdp_years['min_year'] or reference_year,
                'max_historical': mdp_years['max_year'] or reference_year,
                'target': instance_years.target,
            }
        )
        return spec.model_copy(update={'years': years})

    def apply_snapshot_overrides(self, snapshot: InstanceSnapshot) -> InstanceSnapshot:
        """
        Overlay this configuration onto a parsed framework snapshot.

        The framework YAML carries demonstration identity and year boundaries;
        the city-specific values live on this row and its measure datapoints.
        The overlay resolves them into the snapshot once, so the loader needs
        no framework knowledge.
        """
        ic = self.instance_config
        metadata = snapshot.metadata.model_copy(
            update={
                'uuid': ic.uuid,
                'identifier': ic.identifier,
                'name': ic.get_name(),
                'owner': self.organization_name or '',
            }
        )
        spec = self.apply_spec_overrides(snapshot.spec)
        return snapshot.model_copy(update={'metadata': metadata, 'spec': spec})

    def create_model_instance(self, ic: InstanceConfig) -> Instance:
        from nodes.instance_loader import InstanceLoader

        fw = self.framework
        config_fn = ic.get_yaml_config_entrypoint()
        if config_fn is None:
            raise ValueError(f'No YAML config entrypoint found for framework {fw.identifier}')
        loader = InstanceLoader.from_yaml(
            config_fn,
            instance_config=ic,
            snapshot_transform=self.apply_snapshot_overrides,
        )
        return loader.instance

    @staticmethod
    def _get_client_url_parts(
        request: ViewURLRequest | None, client_url: str | None = None
    ) -> tuple[str, str, int | None] | None:
        from paths.schema_context import PathsGraphQLContext

        if client_url is None and request is None:
            return None

        if not client_url and request is not None:
            if isinstance(request, PathsGraphQLContext):
                headers = request.get_request_headers()
            else:
                headers = request.headers
            client_url = headers.get('origin') or headers.get('referer')
            if not client_url and isinstance(request, HttpRequest):
                client_url = request.build_absolute_uri('/')
        if not client_url:
            return None

        parts = urlparse(client_url)
        if parts.scheme not in ('http', 'https') or not parts.hostname:
            return None
        try:
            port = parts.port
        except ValueError:
            port = None
        if (parts.scheme == 'https' and port == 443) or (parts.scheme == 'http' and port == 80):
            port = None
        return parts.scheme, parts.hostname.lower(), port

    @staticmethod
    def _request_wildcard_domains(request: ViewURLRequest | None) -> list[str]:
        from paths.schema_context import PathsGraphQLContext

        if request is None:
            return []

        if isinstance(request, PathsGraphQLContext):
            wildcard_domains = request.wildcard_domains
        else:
            wildcard_domains = request.headers.get(WILDCARD_DOMAINS_HEADER, '').split(',')
        return [domain.strip().lower() for domain in wildcard_domains]

    @staticmethod
    def _format_url(scheme: str, hostname: str, port: int | None, path: str = '') -> str:
        port_str = f':{port}' if port else ''
        return f'{scheme}://{hostname}{port_str}{path}'

    def _get_view_url_from_parts(self, client_parts: tuple[str, str, int | None], request: ViewURLRequest | None) -> str | None:
        from nodes.models import get_instance_identifier_from_wildcard_domain

        fw = self.framework
        ic = self.instance_config
        scheme, hostname, port = client_parts
        _, wildcard_hostname = get_instance_identifier_from_wildcard_domain(
            hostname,
            request=None,
            wildcard_domains=self._request_wildcard_domains(request) or None,
        )
        if wildcard_hostname and fw.use_instance_subdomains:
            return self._format_url(scheme, f'{ic.identifier}.{wildcard_hostname}', port)

        if fw.use_instance_subdomains or fw.root_instance_id is None:
            return None

        if fw.has_cache():
            root_instance = fw.cache.get_root_instance()
        else:
            root_instance = fw.root_instance
        assert root_instance is not None
        for hn in root_instance.hostnames.all():
            if hn.hostname == hostname:
                explicit_match = True
                break
        else:
            explicit_match = False
        if (
            hostname == fw.public_base_fqdn
            or explicit_match
            or (wildcard_hostname is not None and hostname == f'{root_instance.identifier}.{wildcard_hostname}')
        ):
            return self._format_url(scheme, hostname, port, f'/{ic.uuid}')
        return None

    def get_view_url(self, request: ViewURLRequest | None = None, client_url: str | None = None) -> str | None:
        fw = self.framework
        if not fw.public_base_fqdn:
            return None

        ic = self.instance_config
        client_parts = self._get_client_url_parts(request, client_url=client_url)
        if client_parts is not None:
            url = self._get_view_url_from_parts(client_parts, request)
            if url is not None:
                return url
        if fw.use_instance_subdomains:
            return 'https://%s.%s' % (ic.identifier, fw.public_base_fqdn)
        return 'https://%s/%s' % (fw.public_base_fqdn, ic.uuid)

    @property
    def data_points(self) -> MeasureDataPointQuerySet:
        qs = MeasureDataPoint.objects.get_queryset()
        return qs.filter(measure__framework_config=self)

    def notify_change(self, user: UserOrAnon | None = None, save: bool = False):
        self.last_modified_by = user_or_none(user)
        self.last_modified_at = timezone.now()
        if save:
            self.save(update_fields=['last_modified_by', 'last_modified_at'])
        self.instance_config.notify_change()

    def _dimension_name_to_dataset_column_label(self, name: str) -> str:
        return name.replace('_', ' ').capitalize()

    def _get_measure_template_uuids(self, node: DatasetNode) -> list[tuple[str, dict[str, str] | None]]:
        df = node.get_filtered_dataset_df(tag=None)
        if df is None:
            return []
        uuids = {x for x in df.get_column('UUID').to_list() if x is not None}
        dimensions = node.output_dimensions.values()
        column_names = ['UUID'] + [self._dimension_name_to_dataset_column_label(dim.id) for dim in dimensions]
        if len(uuids) < 2 or len(node.output_dimensions) == 0:
            return [(u, None) for u in uuids]

        combinations = set()

        df = df.select(column_names)
        df = node.convert_names_to_ids(df)
        for row in df.iter_rows():
            if row[0] is None:
                continue
            combinations.add(row)
        dim_combinations = [c[1:] for c in combinations]
        if len(dim_combinations) != len(set(dim_combinations)):
            logger.error(f'For node {node.id} unique MeasureTemplate uuids could not be found.')
            return []
        result: list[tuple[str, dict[str, str] | None]] = []
        for _uuid, *categories in combinations:
            dims = {}
            for i, dimension in enumerate(dimensions):
                dims[dimension.id] = categories[i]
            result.append((_uuid, dims))
        return result

    @staticmethod
    def _get_measure_template_uuids_from_binding(
        ds: FrameworkMeasureDVCDataset2,
    ) -> list[tuple[str, dict[str, str] | None]]:
        """
        Read the (uuid, dimension categories) pairs a single ``city_data`` binding carries.

        The ``DatasetNode`` sibling of this method has to undo the GPC sheet's
        human-readable dimension labels via ``convert_names_to_ids``. Here the frame has
        already been through the binding pipeline, so its dimension columns are ids and
        can be read straight off ``dim_ids``.

        Categories are the *only* handle the resolver has on a measure. The values come
        from the node's output, which carries no uuid -- the binding is read to find the
        measure, not to serve it -- so a uuid these cannot pin down to one series is left
        unmapped rather than answered with someone else's numbers:

        - a uuid spread over several categories has no single series;
        - a uuid whose categories do not exclude another uuid's rows cannot be told from
          it, and goes.

        The second test is containment, not equality. A selector matches every row that
        agrees with the categories it names and says nothing about the rest, so a selector
        contained in another is the *broader* one: ``{transport_mode: cars}`` sweeps up
        ``{transport_mode: cars, energy_carrier: electricity}`` as well as its own rows.
        Equal selectors are the symmetric case, each contained in the other, and both go.
        The narrower one survives -- nothing outside it satisfies its extra category.

        The empty selector falls out of the same rule: it is contained in everything, so
        it is fine alone in a binding and ambiguous beside anything else.

        Only the uuids actually in conflict are dropped; the rest of the binding stands.
        Null categories are discarded: they mean the column does not apply to this measure,
        and filtering the node's output for a null category matches nothing.
        """
        df = ds.get_uuid_frame()
        if df is None:
            return []
        dim_ids = [d for d in df.dim_ids if d != 'uuid']

        categories_by_uuid: dict[str, set[tuple[str, ...]]] = {}
        for _uuid, *categories in df.select(['uuid', *dim_ids]).iter_rows():
            if _uuid is None:
                continue
            categories_by_uuid.setdefault(_uuid, set()).add(tuple(categories))

        selectors: dict[str, dict[str, str]] = {}
        ambiguous: list[str] = []
        for _uuid, cats in categories_by_uuid.items():
            if len(cats) > 1:
                ambiguous.append(_uuid)
                continue
            selectors[_uuid] = {dim: cat for dim, cat in zip(dim_ids, next(iter(cats)), strict=True) if cat is not None}

        too_broad = {
            _uuid
            for _uuid, sel in selectors.items()
            if any(other != _uuid and sel.items() <= osel.items() for other, osel in selectors.items())
        }
        ambiguous.extend(sorted(too_broad))
        result: list[tuple[str, dict[str, str] | None]] = [
            (_uuid, sel or None) for _uuid, sel in selectors.items() if _uuid not in too_broad
        ]
        if ambiguous:
            by = f'by {"/".join(dim_ids)}' if dim_ids else 'and carries no dimension to tell them apart'
            msg = (
                f'Dataset {ds.id} does not pin MeasureTemplate(s) {", ".join(sorted(ambiguous))} '
                f'to one series {by}; no placeholder values for them'
            )
            logger.warning(msg)
            sentry_sdk.capture_message(msg)
        return result

    @staticmethod
    def _prefer_historical_bindings(
        instance: Instance,
        values: list[NodeDimensionSelection],
    ) -> list[NodeDimensionSelection]:
        """
        Drop binding selections that lost a same-uuid tie on tags; lower rank wins.

        A single dataset commonly holds one uuid column beside several value columns --
        the historical series, the decarbonisation goal, and so on -- which the config
        binds separately. All of them then claim the same uuid. The placeholder is only
        ever asked for years at or before today (see ``resolve_placeholder_data_points``),
        so the historical series is the one that answers the question.

        The ranking runs *per node*, and settles only the choice this tie-break was added
        for: which column of one dataset a node should be read through. Across nodes it
        decides nothing and must not, because the node-id heuristics that follow know
        things it does not. An action binds a column as ``historical`` while the level node
        downstream of it binds the same column untagged; ranking the two together lets the
        action win on the tag and the ``*_observed`` node never reaches the preference that
        exists for it -- and since the action reports a baseline delta, the cell then shows
        nothing at all.

        A legacy ``DatasetNode`` selection has no binding to rank, and giving it a
        synthetic rank would evict it just as wrongly, so it is always passed through.
        """

        def rank(sel: NodeDimensionSelection) -> int:
            assert sel.dataset_index is not None
            if sel.binding_role == 'historical':
                return 0
            if sel.binding_role == 'goal':
                return 2
            return 1

        best_by_node: dict[str, int] = {}
        for v in values:
            if v.dataset_index is None:
                continue
            best_by_node[v.node_id] = min(rank(v), best_by_node.get(v.node_id, rank(v)))
        return [v for v in values if v.dataset_index is None or rank(v) == best_by_node[v.node_id]]

    @staticmethod
    def _prefer_a_level_over_a_delta(
        instance: Instance,
        values: list[NodeDimensionSelection],
    ) -> list[NodeDimensionSelection]:
        """
        Drop candidates that can only report movement, when one can report a level.

        A cell asks what the plan is for a year, and a node declaring
        ``output_is_baseline_delta`` answers a different question -- how far the plan moves
        from the baseline. Where a level node claims the same uuid, as it commonly does
        when an action and the node it feeds bind the same column, that node is the answer.
        The node-id heuristics cannot see the difference: neither ``new_building_shares``
        nor ``a32_new_building_improvements`` carries a suffix they recognise.

        Falls through untouched when every candidate reports movement, leaving the
        resolver to withhold the value rather than present a delta as a level.
        """
        levels = [v for v in values if not instance.context.nodes[v.node_id].output_is_baseline_delta]
        return levels or values

    @staticmethod
    def _prefer_the_full_trajectory(instance: Instance, sel: NodeDimensionSelection) -> NodeDimensionSelection:
        """
        Point a measure at the node that carries its whole trajectory, where one exists.

        A goal node holds only the target end of a series -- in NZC it begins at the target
        year -- while placeholders are only ever asked for years at or before today, so
        such a measure has nothing to show. Its uuid lands there because the historical
        column of the same dataset is null for it, not because the goal series is what the
        cell wants.

        Which node carries the whole series is a fact about the graph, not about the name:
        a goal feeds an action, and the action combines it with the historical series and
        emits the trajectory. Follow ``goal -> action -> output``. Reading a ``_goal``
        suffix instead would mean a rename silently changed the numbers a city sees, and
        would claim a relationship for any node that merely ends that way.

        An action commonly feeds several nodes -- ``a21_optimised_logistics`` emits both a
        utilisation percentage and vehicle kilometres -- so take the one measuring the same
        quantity as the goal, and only when it can serve the selection's categories and
        metric. Where the graph does not answer unambiguously nothing moves: the measure
        keeps its node and shows nothing, as before.
        """
        from nodes.actions.action import ActionNode

        node = instance.context.nodes.get(sel.node_id)
        if node is None or node.unit is None:
            return sel
        quantity = node.unit

        def can_serve(target: Node) -> bool:
            if target.unit is None or not quantity.is_compatible_with(target.unit):
                return False
            if sel.dimensions and not set(sel.dimensions) <= set(target.output_dimensions):
                return False
            return sel.metric_col is None or sel.metric_col in {m.column_id for m in target.output_metrics.values()}

        targets = {
            target.id
            for action in node.output_nodes
            if isinstance(action, ActionNode)
            for target in action.output_nodes
            if can_serve(target)
        }
        if len(targets) != 1:
            return sel
        return replace(sel, node_id=targets.pop())

    @staticmethod
    def _claimed_uuids(
        describe: str,
        read: Callable[[], list[tuple[str, dict[str, str] | None]]],
    ) -> list[tuple[str, dict[str, str] | None]]:
        """
        Read one source's measure claims, treating a failure as "claims nothing".

        This mapping is built once for the whole framework config and every measure waits
        on it, so an exception escaping here is not one blank cell -- it fails
        ``correspondingNode`` and ``placeholderDataPoints`` for every measure on the tab,
        and being a cached_property it is not even cached, so each measure retries the
        same broken load. A missing DVC dataset or a column removed out from under a
        binding should cost that binding's measures, nothing more.
        """
        try:
            return read()
        except Exception as exc:
            msg = f'Cannot read measure UUIDs from {describe}: {exc}'
            logger.warning(msg)
            sentry_sdk.capture_exception(exc)
            return []

    def _get_node_dimension_selections(self, node_id: str, node: Node) -> list[tuple[str, NodeDimensionSelection]]:
        """Return every (uuid, selection) pair one node claims, by whichever route it carries city data."""
        from frameworks.datasets import FrameworkMeasureDVCDataset2
        from nodes.gpc import DatasetNode

        # Intentionally test for concrete type, filter out subclasses
        if type(node) is DatasetNode:
            # Workaround to filter viz helper nodes
            # FIXME: Implement this better later
            if node.get_parameter_value_str('uuid', required=False):
                return []
            return [
                (_uuid, NodeDimensionSelection(node_id=node_id, dimensions=dimensions))
                for _uuid, dimensions in self._claimed_uuids(f'node {node_id}', partial(self._get_measure_template_uuids, node))
            ]

        # Nodes that take their city data through a tagged binding rather than by being a
        # DatasetNode. The class-based branch above cannot see these, and since f2d6be20
        # the NZC model is built entirely out of them.
        selections: list[tuple[str, NodeDimensionSelection]] = []
        for index, ds in enumerate(node.input_dataset_instances):
            if 'city_data' not in ds.tags or not isinstance(ds, FrameworkMeasureDVCDataset2):
                continue
            # A multi-metric node renames Value to the metric column, so the resolver has
            # to be told which one this binding feeds. The config says so by tagging the
            # binding with the metric's id alongside historical/goal/city_data.
            metric = next((tag for tag in ds.tags if tag in node.output_metrics), None)
            metric_col = node.output_metrics[metric].column_id if metric is not None else None
            # 'observed' is the older spelling of 'historical' and ranks with it.
            role = next((tag for tag in ('historical', 'observed', 'goal') if tag in ds.tags), None)
            role = 'historical' if role == 'observed' else role
            claims = self._claimed_uuids(
                f'binding {index} ({ds.id}) of node {node_id}',
                partial(self._get_measure_template_uuids_from_binding, ds),
            )
            selections += [
                (
                    _uuid,
                    NodeDimensionSelection(
                        node_id=node_id,
                        dimensions=dimensions,
                        dataset_index=index,
                        metric_col=metric_col,
                        binding_role=role,
                    ),
                )
                for _uuid, dimensions in claims
            ]
        return selections

    @cached_property
    def measure_template_uuid_to_node_dimension_selection(self) -> Mapping[str, NodeDimensionSelection]:
        measure_template_uuid_to_multiple_node_dimensions_selections: dict[str, list[NodeDimensionSelection]] = dict()
        instance = self.instance_config.get_instance()
        for node_id, node in instance.context.nodes.items():
            for _uuid, selection in self._get_node_dimension_selections(node_id, node):
                measure_template_uuid_to_multiple_node_dimensions_selections.setdefault(_uuid, []).append(selection)

        re_historical = re.compile(r'.*_historical$')
        re_observed = re.compile(r'.*_observed$')

        measure_template_uuid_to_single_node_dimension_selection: dict[str, NodeDimensionSelection] = dict()
        for _uuid, values in measure_template_uuid_to_multiple_node_dimensions_selections.items():
            if len(values) == 1:
                measure_template_uuid_to_single_node_dimension_selection[_uuid] = values[0]
                continue
            # The node-id heuristics below cannot separate bindings that live on the same
            # node, which is the usual shape of a tie now: one dataset, one uuid column,
            # several value columns. Narrow by binding tag first, and only fall through to
            # the node-id heuristics if that still leaves a genuine choice between nodes.
            candidates = self._prefer_a_level_over_a_delta(instance, self._prefer_historical_bindings(instance, values))
            if len(candidates) == 1:
                measure_template_uuid_to_single_node_dimension_selection[_uuid] = candidates[0]
                continue
            accepted_values = [v for v in candidates if re_observed.match(v.node_id)]
            if len(accepted_values) != 1:
                accepted_values = [v for v in candidates if not re_historical.match(v.node_id)]
            if len(accepted_values) == 1:
                measure_template_uuid_to_single_node_dimension_selection[_uuid] = accepted_values[0]
                continue
            msg = f'Cannot find single Node to match MeasureTemplate {_uuid}: {", ".join([n.node_id for n in candidates])}'
            logger.warning(msg)
            sentry_sdk.capture_message(msg)
        return {
            _uuid: self._prefer_the_full_trajectory(instance, sel)
            for _uuid, sel in measure_template_uuid_to_single_node_dimension_selection.items()
        }
