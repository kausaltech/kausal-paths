from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from django.conf import settings
from django.utils import translation
from graphql import get_argument_values
from graphql.error import GraphQLError
from strawberry.utils.operation import get_first_operation

import sentry_sdk
from loguru import logger

from kausal_common.i18n.pydantic import is_query_with_instance_context, set_i18n_context
from kausal_common.strawberry.context import GraphQLContext
from kausal_common.strawberry.extensions import AuthenticationExtension, ExecutionCacheExtension, GraphQLPerfNode, SchemaExtension

from paths.context import PathsObjectCache, paths_object_cache

from params.storage import SessionStorage

if TYPE_CHECKING:
    from collections.abc import Generator
    from uuid import UUID

    from graphql.language import DirectiveNode, OperationDefinitionNode

    from paths.schema import PreviewMode

    from nodes.constraints.solver import ConstraintSolveResult
    from nodes.instance import Instance
    from nodes.instance_graph import InstanceGraph
    from nodes.instance_graph_cache import LoadedInstanceSnapshot, ResolvedInstanceSource
    from nodes.instance_serialization import InstanceSnapshot
    from nodes.models import InstanceConfig, InstanceConfigQuerySet, PreferredInstanceSource

logger = logger.bind(markup=True)


@dataclass(frozen=True)
class InstanceRuntimeKey:
    instance_pk: int
    source: PreferredInstanceSource
    tolerate_node_failures: bool


@dataclass
class InstanceRequestResources:
    """Request-owned lazy snapshots, graphs, runtimes, and managed lifecycles."""

    default_config: InstanceConfig | None
    default_source: PreferredInstanceSource | None
    default_tolerate_node_failures: bool
    stack: ExitStack
    extension: ActivateInstanceContextExtension
    object_cache: PathsObjectCache
    instances: dict[InstanceRuntimeKey, Instance] = field(default_factory=dict)
    instance_refreshes: set[tuple[int, PreferredInstanceSource]] = field(default_factory=set)
    snapshots: dict[ResolvedInstanceSource, LoadedInstanceSnapshot] = field(default_factory=dict)
    constraint_solves: dict[ResolvedInstanceSource, ConstraintSolveResult] = field(default_factory=dict)

    def resolve_source(
        self,
        config: InstanceConfig | None = None,
        source: PreferredInstanceSource | None = None,
    ) -> tuple[InstanceConfig, PreferredInstanceSource]:
        config = config or self.default_config
        if config is None:
            raise GraphQLError(
                "Unable to determine Paths instance for the request. Use the 'instance' directive or HTTP headers.",
            )
        is_default = self.default_config is not None and config.pk == self.default_config.pk
        if source is None:
            source = self.default_source if is_default else self._draft_source()
        assert source is not None
        return config, source

    @contextmanager
    def _instance_context(
        self,
        config: InstanceConfig,
        source: PreferredInstanceSource,
        tolerate_node_failures: bool,
        force_reinitialize: bool,
    ) -> Generator[Instance]:
        instance: Instance | None = None
        try:
            with (
                config.enter_instance_context(
                    source=source,
                    tolerate_node_failures=tolerate_node_failures,
                    force_reinitialize=force_reinitialize,
                ) as instance,
                instance.lock,
                instance.context.run(),
            ):
                self.extension.activate_instance(instance)
                yield instance
        finally:
            if instance is not None:
                instance.clean()

    def require_instance(
        self,
        config: InstanceConfig | None = None,
        *,
        source: PreferredInstanceSource | None = None,
        tolerate_node_failures: bool | None = None,
        refresh: bool = False,
    ) -> Instance:
        config, source = self.resolve_source(config, source)
        is_default = self.default_config is not None and config.pk == self.default_config.pk
        if tolerate_node_failures is None:
            tolerate_node_failures = self.default_tolerate_node_failures if is_default else False

        key = InstanceRuntimeKey(
            instance_pk=config.pk,
            source=source,
            tolerate_node_failures=tolerate_node_failures,
        )
        refresh_key = (config.pk, source)
        refresh = refresh or refresh_key in self.instance_refreshes
        instance = None if refresh else self.instances.get(key)
        if instance is not None:
            return instance

        perf = self.extension.get_context().graphql_perf
        with (
            perf.exec_node(GraphQLPerfNode('get instance "%s"' % config.identifier)),
            is_query_with_instance_context.set(True),
        ):
            instance = self.stack.enter_context(
                self._instance_context(config, source, tolerate_node_failures, refresh),
            )
        self.instances[key] = instance
        self.instance_refreshes.discard(refresh_key)
        return instance

    def invalidate_instance(
        self,
        config: InstanceConfig | None = None,
        *,
        source: PreferredInstanceSource | None = None,
    ) -> None:
        """Discard request-local runtimes so the next accessor rebuilds lazily."""
        config, source = self.resolve_source(config, source)
        stale_keys = [key for key in self.instances if key.instance_pk == config.pk and key.source == source]
        for key in stale_keys:
            del self.instances[key]
        self.instance_refreshes.add((config.pk, source))

    def require_graph(
        self,
        config: InstanceConfig | None = None,
        *,
        source: PreferredInstanceSource | None = None,
        refresh: bool = False,
    ) -> InstanceGraph:
        from nodes.instance_graph_cache import get_instance_graph, resolve_instance_source

        config, source = self.resolve_source(config, source)
        resolved_source = resolve_instance_source(config, source)
        return get_instance_graph(
            config,
            source,
            object_cache=self.object_cache,
            refresh=refresh,
            snapshot_loader=lambda: self._require_loaded_snapshot(
                config,
                resolved_source,
                refresh=refresh,
            ),
            resolved_source=resolved_source,
        )

    def require_constraint_solve(
        self,
        config: InstanceConfig | None = None,
        *,
        source: PreferredInstanceSource | None = None,
    ) -> ConstraintSolveResult:
        """
        Solve the selected source's constraint program with dataset shape profiles.

        Request-memoized by resolved source: field resolvers across many
        ports and nodes share one solve, and any draft edit changes the
        resolved version, so a stale result is never served.
        """
        from nodes.constraints.validation import solve_instance_constraints
        from nodes.instance_graph_cache import resolve_instance_source

        config, source = self.resolve_source(config, source)
        resolved_source = resolve_instance_source(config, source)
        cached = self.constraint_solves.get(resolved_source)
        if cached is not None:
            return cached
        graph = self.require_graph(config, source=source)
        result = solve_instance_constraints(config, graph, resolved_source)
        self.constraint_solves[resolved_source] = result
        return result

    def _require_loaded_snapshot(
        self,
        config: InstanceConfig,
        source: ResolvedInstanceSource,
        *,
        refresh: bool = False,
    ) -> LoadedInstanceSnapshot:
        from nodes.instance_graph_cache import load_instance_snapshot

        loaded = None if refresh else self.snapshots.get(source)
        if loaded is None:
            loaded = load_instance_snapshot(config, source)
            self.snapshots[source] = loaded
        return loaded

    def require_snapshot(
        self,
        config: InstanceConfig | None = None,
        *,
        source: PreferredInstanceSource | None = None,
        refresh: bool = False,
    ) -> InstanceSnapshot:
        from nodes.instance_graph_cache import resolve_instance_source

        config, source = self.resolve_source(config, source)
        resolved_source = resolve_instance_source(config, source)
        return self._require_loaded_snapshot(config, resolved_source, refresh=refresh).snapshot

    def snapshot_for_instance_type(
        self,
        config: InstanceConfig | None = None,
        *,
        source: PreferredInstanceSource | None = None,
    ) -> InstanceSnapshot | None:
        """Return selected revision content when presentation must follow a snapshot."""
        from nodes.instance_graph_cache import resolve_instance_source

        config, source = self.resolve_source(config, source)
        resolved_source = resolve_instance_source(config, source)
        if resolved_source.kind != 'database-published':
            return None
        return self._require_loaded_snapshot(config, resolved_source).snapshot

    @staticmethod
    def _draft_source() -> PreferredInstanceSource:
        from nodes.models import PreferredInstanceSource

        return PreferredInstanceSource.DRAFT


@dataclass
class PathsGraphQLContext[InstanceType: Instance | None = Instance | None](GraphQLContext):
    instance_config: InstanceConfig | None = None
    cache: PathsObjectCache = field(init=False)
    instance_resources: InstanceRequestResources | None = field(init=False, default=None, repr=False)

    # Populated by DetermineInstanceContextExtension from @instance / @context
    # directive arguments. Consumed by editing mutations for optimistic
    # locking (`expected_version`) and by Phase 4's resolve_instance branch
    # (`preview_mode`).
    preview_mode: PreviewMode | None = None
    expected_version: UUID | None = None
    tolerate_node_failures: bool = False

    def __post_init__(self):
        super().__post_init__()
        user = self.get_user()
        cache = None
        if paths_object_cache.is_set():
            cache = paths_object_cache.get()
            if cache.user != user:
                cache = None
        if cache is None:
            cache = PathsObjectCache(user=user)
        self.cache = cache

    @property
    def instance(self) -> InstanceType:
        if self.instance_resources is None or self.instance_resources.default_config is None:
            return cast('InstanceType', None)
        return cast('InstanceType', self.instance_resources.require_instance())

    def require_instance(
        self,
        config: InstanceConfig | None = None,
        *,
        source: PreferredInstanceSource | None = None,
        tolerate_node_failures: bool | None = None,
        refresh: bool = False,
    ) -> Instance:
        if self.instance_resources is None:
            raise GraphQLError(
                "Unable to determine Paths instance for the request. Use the 'instance' directive or HTTP headers.",
            )
        return self.instance_resources.require_instance(
            config,
            source=source,
            tolerate_node_failures=tolerate_node_failures,
            refresh=refresh,
        )

    def require_instance_graph(
        self,
        config: InstanceConfig | None = None,
        *,
        source: PreferredInstanceSource | None = None,
        refresh: bool = False,
    ) -> InstanceGraph:
        if self.instance_resources is None:
            raise GraphQLError(
                "Unable to determine Paths instance for the request. Use the 'instance' directive or HTTP headers.",
            )
        return self.instance_resources.require_graph(config, source=source, refresh=refresh)

    def require_constraint_solve(
        self,
        config: InstanceConfig | None = None,
        *,
        source: PreferredInstanceSource | None = None,
    ) -> ConstraintSolveResult:
        if self.instance_resources is None:
            raise GraphQLError(
                "Unable to determine Paths instance for the request. Use the 'instance' directive or HTTP headers.",
            )
        return self.instance_resources.require_constraint_solve(config, source=source)

    def invalidate_runtime_instance(
        self,
        config: InstanceConfig | None = None,
        *,
        source: PreferredInstanceSource | None = None,
    ) -> None:
        if self.instance_resources is None:
            raise GraphQLError(
                "Unable to determine Paths instance for the request. Use the 'instance' directive or HTTP headers.",
            )
        self.instance_resources.invalidate_instance(config, source=source)

    def require_instance_snapshot(
        self,
        config: InstanceConfig | None = None,
        *,
        source: PreferredInstanceSource | None = None,
        refresh: bool = False,
    ) -> InstanceSnapshot:
        if self.instance_resources is None:
            raise GraphQLError(
                "Unable to determine Paths instance for the request. Use the 'instance' directive or HTTP headers.",
            )
        return self.instance_resources.require_snapshot(config, source=source, refresh=refresh)

    def instance_snapshot_for_type(
        self,
        config: InstanceConfig | None = None,
        *,
        source: PreferredInstanceSource | None = None,
    ) -> InstanceSnapshot | None:
        if self.instance_resources is None:
            raise GraphQLError(
                "Unable to determine Paths instance for the request. Use the 'instance' directive or HTTP headers.",
            )
        return self.instance_resources.snapshot_for_instance_type(config, source=source)


class PathsSchemaExtension(SchemaExtension[PathsGraphQLContext]):
    context_class: type[PathsGraphQLContext[Instance | None]] = PathsGraphQLContext


class DetermineInstanceContextExtension(PathsSchemaExtension):
    def process_locale_directive(self, ic: InstanceConfig, directive: DirectiveNode) -> str:
        from kausal_common.strawberry.schema import locale_directive

        assert locale_directive.graphql_name is not None
        exec_ctx = self.execution_context
        directive_ast = exec_ctx.schema._schema.get_directive(locale_directive.graphql_name)
        assert directive_ast is not None
        lang = get_argument_values(directive_ast, directive, exec_ctx.variables).get('lang')
        if lang is None:
            raise GraphQLError('Locale directive missing lang argument', directive)

        if lang not in ic.supported_languages:
            raise GraphQLError('unsupported language: %s. Did you run --update-instance?' % lang, directive)
        return lang

    def get_ic_queryset(self) -> InstanceConfigQuerySet:
        from nodes.models import InstanceConfig

        return (
            InstanceConfig.objects.get_queryset().select_related('framework_config').select_related('framework_config__framework')
        )

    def get_instance_by_identifier(
        self,
        queryset: InstanceConfigQuerySet,
        identifier: str,
        directive: DirectiveNode | None = None,
    ) -> InstanceConfig:
        from nodes.models import InstanceConfig

        try:
            if identifier.isnumeric():
                instance = queryset.get(id=identifier)
            else:
                instance = queryset.get(identifier=identifier)
        except InstanceConfig.DoesNotExist:
            raise GraphQLError('Instance with identifier %s not found' % identifier, directive) from None
        return instance

    def get_instance_by_hostname(
        self,
        queryset: InstanceConfigQuerySet,
        hostname: str,
        directive: DirectiveNode | None = None,
    ) -> InstanceConfig:
        from nodes.models import InstanceConfig

        ctx = self.get_context()
        try:
            instance = queryset.for_hostname(hostname, wildcard_domains=ctx.wildcard_domains).get()
        except InstanceConfig.DoesNotExist:
            logger.warning(f'No instance found for hostname {hostname} (wildcard domains: {ctx.wildcard_domains})')
            raise GraphQLError('Instance matching hostname %s not found' % hostname, directive) from None
        return instance

    def process_instance_directive(self, directive: DirectiveNode) -> InstanceConfig:
        from .schema import instance_directive as instance_directive_def

        assert instance_directive_def.graphql_name is not None
        qs = self.get_ic_queryset()
        exec_ctx = self.execution_context
        directive_ast = exec_ctx.schema._schema.get_directive(instance_directive_def.graphql_name)
        assert directive_ast is not None
        arguments = get_argument_values(directive_ast, directive, exec_ctx.variables)
        identifier = arguments.get('identifier')
        hostname = arguments.get('hostname')
        if identifier:
            ic = self.get_instance_by_identifier(qs, identifier, directive)
        elif hostname:
            ic = self.get_instance_by_hostname(qs, hostname, directive)
        else:
            raise GraphQLError('Invalid instance directive', directive)
        self._apply_preview_and_version(
            arguments.get('preview'),
            arguments.get('version'),
            arguments.get('tolerate_node_failures', False),
        )
        return ic

    def process_context_directive(self, directive: DirectiveNode) -> tuple[InstanceConfig | None, str | None]:
        from .schema import context_directive

        assert context_directive.graphql_name is not None
        exec_ctx = self.execution_context
        directive_ast = exec_ctx.schema._schema.get_directive(context_directive.graphql_name)
        assert directive_ast is not None
        ctx = get_argument_values(directive_ast, directive, exec_ctx.variables).get('input')
        if ctx is None:
            return None, None
        # FIXME: Filter by user permissions
        qs = self.get_ic_queryset()
        identifier = ctx.get('identifier')
        hostname = ctx.get('hostname')
        if identifier:
            ic = self.get_instance_by_identifier(qs, identifier)
        elif hostname:
            ic = self.get_instance_by_hostname(qs, hostname)
        else:
            return None, None
        locale = ctx.get('locale')
        if not locale:
            locale = ic.primary_language
        elif locale not in ic.supported_languages:
            raise GraphQLError('unsupported language: %s. Did you run --update-instance?' % locale, directive)
        self._apply_preview_and_version(
            ctx.get('preview'),
            ctx.get('version'),
            ctx.get('tolerate_node_failures', False),
        )
        return ic, locale

    def _apply_preview_and_version(
        self,
        preview: Any,
        version: Any,
        tolerate_node_failures: bool = False,
    ) -> None:
        """
        Stash the directive's ``preview`` / ``version`` / fault-tolerance args on the context.

        ``preview`` reaches us as the ``PreviewMode`` enum instance (via
        ``get_argument_values``) or ``None``. ``version`` is a ``UUID`` or
        ``None``. Editing mutations later read ``expected_version`` to gate
        the stale-check. ``tolerate_node_failures`` opts into fault-tolerant
        computation (see docs/architecture/fault-tolerance.md).
        """
        ctx = self.get_context()
        ctx.preview_mode = preview
        ctx.expected_version = version
        ctx.tolerate_node_failures = bool(tolerate_node_failures)

    def process_instance_headers(self) -> InstanceConfig | None:
        headers = self.get_request_headers()
        identifier = headers.get(settings.INSTANCE_IDENTIFIER_HEADER)
        hostname = headers.get(settings.INSTANCE_HOSTNAME_HEADER)

        qs = self.get_ic_queryset()
        if identifier:
            return self.get_instance_by_identifier(qs, identifier)
        if hostname:
            return self.get_instance_by_hostname(qs, hostname)
        return None

    def determine_instance_and_locale(self, operation: OperationDefinitionNode) -> tuple[InstanceConfig, str] | None:
        instance_config: InstanceConfig | None = None
        locale: str | None = None
        user = self.get_user()

        for directive in operation.directives or []:
            directive_name = directive.name.value
            if directive_name == 'context':
                instance_config, locale = self.process_context_directive(directive)
                if instance_config is not None:
                    break
            elif directive_name == 'instance':
                instance_config = self.process_instance_directive(directive)
                break
        else:
            instance_config = self.process_instance_headers()

        if instance_config is None:
            return None

        if instance_config.is_protected and user is None:
            raise GraphQLError('Instance is protected', extensions=dict(code='instance_protected'))

        if locale is None:
            for directive in operation.directives or []:
                directive_name = directive.name.value
                if directive_name != 'locale':
                    continue
                locale = self.process_locale_directive(instance_config, directive)
                break
            else:
                locale = instance_config.primary_language

        ctx = self.get_context()
        ctx.graphql_query_language = locale
        ctx.instance_config = instance_config
        return instance_config, locale

    def on_execute(self) -> Generator[None]:
        doc = self.execution_context.graphql_document
        if doc:
            op = get_first_operation(doc)
        else:
            op = None

        if not op or self.execution_context.result:
            yield
            return

        self.determine_instance_and_locale(op)
        yield


class ActivateInstanceContextExtension(PathsSchemaExtension):
    @contextmanager
    def activate_language(self, lang: str):
        with translation.override(lang), set_i18n_context(lang, other_languages=[]):
            yield

    def set_instance_scope(self) -> None:
        scope = sentry_sdk.get_current_scope()
        ic = self.get_context().instance_config
        if ic is None:
            return
        scope.set_tag('instance_id', ic.identifier)
        scope.set_tag('instance_uuid', str(ic.uuid))
        if ic.has_framework_config():
            fw = self.get_context().cache.for_framework_id(ic.framework_config.framework_id)
            if fw is not None:
                scope.set_tag('framework_id', fw.identifier)

    def activate_instance(self, instance: Instance):
        context = instance.context
        session = self.get_session()
        assert session is not None
        context.setting_storage = storage = SessionStorage(instance=instance, session=session)
        active_scenario_id = storage.get_active_scenario()
        scenario = None
        if active_scenario_id:
            try:
                scenario = context.get_scenario(active_scenario_id)
            except KeyError:
                storage.set_active_scenario(None)

        # Tell the custom scenario about the user setting so that
        # it can locate the customized parameters.
        if context.custom_scenario is not None:
            context.custom_scenario.set_storage(storage)

        if scenario is None:
            scenario = context.get_default_scenario()

        # Activate normalization
        if context.setting_storage.has_option('normalizer'):
            val = context.setting_storage.get_option('normalizer')
            context.set_option('normalizer', val)
        else:
            for n in context.normalizations.values():
                if n.spec.default:
                    context.active_normalization = n
                    break
            else:
                context.active_normalization = None

        context.activate_scenario(scenario)

    def _resolve_preview_source(
        self,
        ic: InstanceConfig,
        ctx: PathsGraphQLContext[Any],
    ) -> PreferredInstanceSource:
        """
        Translate ``ctx.preview_mode`` into a ``PreferredInstanceSource``.

        Rules:

        * Non-DB (YAML / framework) instances always serve DRAFT, ignoring
          the directive. They have no live revision and the ORM-overlay on
          YAML configs *is* the editable state — preserves pre-Phase-4
          behavior for non-migrated instances end-to-end.
        * Explicit ``DRAFT`` on a DB-sourced instance requires ``change``
          permission.
        * Explicit ``PUBLISHED`` serves the live revision; if none exists,
          ``_create_from_config`` silently falls back to DRAFT.
        * Default (no ``preview`` arg): if a live revision exists, serve
          PUBLISHED; otherwise DRAFT (bootstrap path for brand-new DB
          instances that haven't been published yet). No perm check on
          the default path — anonymous public reads land here.
        """
        from paths.schema import PreviewMode

        from nodes.models import PreferredInstanceSource

        # Non-DB sources: directive is advisory, DRAFT wins.
        if ic.config_source != 'database':
            return PreferredInstanceSource.DRAFT

        mode = ctx.preview_mode
        if mode == PreviewMode.DRAFT:
            from users.models import User

            user = ctx.get_user()
            if not isinstance(user, User) or user.is_anonymous:
                raise GraphQLError(
                    'Draft preview requires authentication.',
                    extensions={'code': 'permission_denied'},
                )
            if not ic.permission_policy().user_can_preview_draft(user, ic):
                raise GraphQLError(
                    'Draft preview requires editor permission on this instance.',
                    extensions={'code': 'permission_denied'},
                )
            return PreferredInstanceSource.DRAFT
        if mode == PreviewMode.PUBLISHED:
            return PreferredInstanceSource.PUBLISHED
        # Default: publish-first. Bootstrap to DRAFT for instances that
        # have never been published; once a revision lands, the default
        # serves that.
        if ic.live_revision_id is not None:
            return PreferredInstanceSource.PUBLISHED
        return PreferredInstanceSource.DRAFT

    @contextmanager
    def request_context(self, _operation: OperationDefinitionNode):
        ctx = self.get_context()
        perf = ctx.graphql_perf
        ic = ctx.instance_config
        source = self._resolve_preview_source(ic, ctx) if ic is not None else None
        with ExitStack() as stack:
            stack.enter_context(paths_object_cache.activate(ctx.cache))
            if ic is not None:
                assert ctx.graphql_query_language is not None
                with perf.exec_node(GraphQLPerfNode('prepare instance "%s"' % ic.identifier)):
                    stack.enter_context(self.activate_language(ctx.graphql_query_language))
                    stack.enter_context(sentry_sdk.new_scope())
                    stack.enter_context(logger.contextualize(instance=ic.identifier))
                    self.set_instance_scope()
            ctx.instance_resources = InstanceRequestResources(
                default_config=ic,
                default_source=source,
                default_tolerate_node_failures=ctx.tolerate_node_failures,
                stack=stack,
                extension=self,
                object_cache=ctx.cache,
            )
            try:
                yield
            finally:
                ctx.instance_resources = None

    def on_execute(self) -> Generator[None]:
        doc = self.execution_context.graphql_document
        if doc:
            op = get_first_operation(doc)
        else:
            op = None

        if not op or self.execution_context.result:
            yield
            return

        with self.request_context(op):
            yield


class PathsExecutionCacheExtension(ExecutionCacheExtension[PathsGraphQLContext]):
    context_class: type[PathsGraphQLContext[Instance | None]] = PathsGraphQLContext

    def get_cache_key_parts(self) -> list[str] | None:
        exec_ctx = self.get_context()
        ic = exec_ctx.instance_config
        if ic is None:
            self.set_reason('no instance config')
            return None

        parts = [str(ic.uuid), ic.cache_invalidated_at.isoformat()]
        session = self.get_session()
        if session is not None:
            session_key = SessionStorage.get_cache_key(session, ic.identifier)
            if session_key is None:
                self.set_reason('user session has custom parameters')
                return None
            parts.append(session_key)
        return parts


class PathsAuthenticationExtension(AuthenticationExtension[PathsGraphQLContext]):
    context_class: type[PathsGraphQLContext[Instance | None]] = PathsGraphQLContext
