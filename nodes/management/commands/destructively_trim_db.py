from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

from django.apps import apps
from django.conf import settings
from django.contrib.admin.models import LogEntry
from django.contrib.contenttypes.models import ContentType
from django.contrib.sessions.models import Session
from django.core.management import CommandError, call_command
from django.core.management.base import BaseCommand
from django.db import transaction
from django.db.models import Exists, OuterRef, Q
from django.db.models.functions import Cast
from django.db.models.signals import post_delete
from wagtail.images import get_image_model
from wagtail.models import DraftStateMixin, ModelLogEntry, Page, PageLogEntry, Revision as WagtailRevision

import factory
from oauth2_provider.models import AccessToken, RefreshToken
from social_django.models import Association, Code, Nonce, Partial

from kausal_common.datasets.models import (
    Dataset,
    DatasetSchema,
    DatasetSchemaScope,
    DataSource,
    Dimension,
    DimensionScope,
)

from nodes.models import InstanceConfig
from orgs.models import Organization
from request_log.models import LoggedRequest
from users.models import User

if TYPE_CHECKING:
    from django.db.models import Field, Model, QuerySet


@dataclass(frozen=True)
class ScopeIds:
    """
    Ids of the instances/orgs being deleted vs. retained, captured before deletion.

    Used by the scoped-dataset cleanup to resolve generic-FK ``scope`` columns once the
    instances/orgs themselves are gone, and to keep datasets shared with a retained scope.

    The ``*_schema_ids`` are the ids of schemas made available to those scopes via a
    ``DatasetSchemaScope``. They must be captured *before* deletion: ``InstanceConfig`` has a
    ``GenericRelation`` to ``DatasetSchemaScope`` (``InstanceConfig.dataset_schema_scopes``), so
    deleting an instance already cascade-deletes its schema-scope rows — querying them afterwards
    would miss every instance-scoped schema.
    """

    instance_ct: ContentType
    org_ct: ContentType
    deleted_instance_ids: list[int]
    deleted_org_ids: list[int]
    kept_instance_ids: list[int]
    kept_org_ids: list[int]
    deleted_schema_ids: list[int]
    kept_schema_ids: list[int]


class Command(BaseCommand):
    help = 'Delete instances and related data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--exclude-instance',
            metavar='IDENTIFIER',
            action='append',
            help='Exclude the instance with the specified identifier from deletion',
        )
        parser.add_argument(
            '--exclude-organization',
            metavar='UUID',
            action='append',
            help='Exclude the organization with the specified UUID from deletion',
        )
        parser.add_argument(
            '--no-confirm',
            action='store_true',
            help='Do not ask for confirmation but delete right away',
        )
        parser.add_argument(
            '--thorough',
            action='store_true',
            help='Delete more data, including revision history and audit logs',
        )

    def handle(self, *args, **options):
        if not settings.DEBUG or settings.DEPLOYMENT_TYPE == 'production':
            raise CommandError(
                'Sorry, for preventing accidents, this management command only works if DEBUG is true and '
                "DEPLOYMENT_TYPE is not 'production'.",
            )

        instances_to_delete, instances_to_keep = self._determine_instances_to_delete(options)
        orgs_to_delete = self._determine_organizations_to_delete(instances_to_keep, options)

        self._print_deletion_summary()
        if not self._confirm_deletion(options):
            return

        self.delete_data(
            instances_to_delete,
            orgs_to_delete,
            thorough=options['thorough'],
        )
        self.stdout.write("Rebuilding Wagtail's reference index...")
        call_command('rebuild_references_index')

    def _determine_instances_to_delete(self, options) -> tuple[QuerySet[InstanceConfig], QuerySet[InstanceConfig]]:
        all_identifiers = list(InstanceConfig.objects.values_list('identifier', flat=True))
        exclude_instances = options.get('exclude_instance') or []
        for identifier in exclude_instances:
            if identifier not in all_identifiers:
                raise CommandError(f"No instance with identifier '{identifier}' exists.")
        instances_to_delete = InstanceConfig.objects.exclude(identifier__in=exclude_instances)
        instances_to_keep = InstanceConfig.objects.exclude(id__in=instances_to_delete)
        delete_identifiers = list(instances_to_delete.values_list('identifier', flat=True))
        if exclude_instances:
            self.stdout.write(f'The following instances will not be deleted: {", ".join(exclude_instances)}')
        if delete_identifiers:
            self.stdout.write(f'The following instances will be deleted with all related data: {", ".join(delete_identifiers)}')
        return instances_to_delete, instances_to_keep

    def _validate_exclude_organizations(self, exclude_organizations: list[str]) -> None:
        if not exclude_organizations:
            return
        all_uuids = {str(u) for u in Organization.objects.values_list('uuid', flat=True)}
        for uuid_str in exclude_organizations:
            if uuid_str not in all_uuids:
                raise CommandError(f"No organization with UUID '{uuid_str}' exists.")

    def _determine_organizations_to_delete(self, instances_to_keep, options) -> QuerySet[Organization]:
        exclude_organizations = options.get('exclude_organization') or []
        self._validate_exclude_organizations(exclude_organizations)
        orgs_to_keep = Organization.objects.none()
        for instance in instances_to_keep:
            orgs_to_keep |= Organization.objects.qs.available_for_instance(instance)
            # available_for_instance() only returns the instance's organization and its descendants.
            # Also keep its ancestors, or deleting the other organizations would remove a retained
            # instance's parent/root and corrupt its (treebeard) organization hierarchy.
            org = getattr(instance, 'organization', None)
            if org is not None:
                orgs_to_keep |= org.get_ancestors()
        if exclude_organizations:
            orgs_to_keep |= Organization.objects.filter(uuid__in=exclude_organizations)
        orgs_to_delete = Organization.objects.exclude(id__in=orgs_to_keep)
        num_delete_suborgs = {}
        for org in orgs_to_delete.filter(depth=1):
            num_delete_suborgs[org] = orgs_to_delete.filter(id__in=org.get_descendants()).count()
        if num_delete_suborgs:
            strings = []
            for org, n in num_delete_suborgs.items():
                string = org.name
                if n == 1:
                    string += ' (and 1 suborganization)'
                elif n > 1:
                    string += f' (and {n} suborganizations)'
                strings.append(string)
            self.stdout.write(f'The following organizations will be deleted: {", ".join(strings)}')
        return orgs_to_delete

    def _print_deletion_summary(self) -> None:
        self.stdout.write('Moreover, the following data will be deleted:')
        self.stdout.write(
            '- all datasets, data sources, dimensions and schemas belonging to the deleted instances or organizations',
        )
        self.stdout.write("- all User instances that don't have a corresponding Person anymore")
        self.stdout.write('- all Wagtail Revision instances whose target object no longer exists')
        self.stdout.write('- all Wagtail ModelLogEntry instances whose target object no longer exists')
        self.stdout.write('- all Wagtail PageLogEntry instances whose target page no longer exists')
        self.stdout.write('- all logged requests')
        self.stdout.write('- all sessions')
        self.stdout.write('- all image renditions')

    def _confirm_deletion(self, options) -> bool:
        if not options['no_confirm']:
            confirmation = input('Do you want to proceed? [y/N] ').lower()
            if confirmation != 'y':
                self.stdout.write(self.style.WARNING('Aborted by user.'))
                return False
        return True

    def delete_all(self, model: type[Model]) -> None:
        self.stdout.write(f'Deleting {model.__name__} instances...')
        _, by_type = model._default_manager.all().delete()
        self.print_deleted_instances_by_model(by_type)

    def delete_scoped_datasets(self, scope_ids: ScopeIds) -> None:
        """
        Delete dataset-related rows owned by deleted instances/orgs via the generic ``scope`` FK.

        ``Dataset``, ``DataSource``, ``DimensionScope`` and ``DatasetSchemaScope`` reference their scope
        (an ``InstanceConfig`` or ``Organization``) through a ``GenericForeignKey``. Django only cascades
        a generic relation when the *target* declares a matching ``GenericRelation``; ``InstanceConfig``
        does so only for ``DatasetSchemaScope`` and ``Organization`` not at all. Without this explicit
        cleanup the bulk data — notably the ``DataPoint`` rows that cascade from each ``Dataset`` — would
        survive as orphans pointing at deleted scopes, defeating the purpose of the trim.

        Must run *after* the instances/orgs (and their cascade-deleted ``NodeDataset``/``DatasetPort``
        rows) are gone: those rows reference ``Dataset`` with ``on_delete=PROTECT``, so deleting datasets
        first would raise ``ProtectedError``. The scope of each row is resolved from the ids captured
        before deletion (``scope_ids``).
        """

        def scope_q(instance_ids: list[int], org_ids: list[int]) -> Q:
            return Q(scope_content_type=scope_ids.instance_ct, scope_id__in=instance_ids) | Q(
                scope_content_type=scope_ids.org_ct, scope_id__in=org_ids
            )

        deleted_scope = scope_q(scope_ids.deleted_instance_ids, scope_ids.deleted_org_ids)
        kept_scope = scope_q(scope_ids.kept_instance_ids, scope_ids.kept_org_ids)

        # Paths treats a dataset as owned by an instance either when it is directly scoped to it OR
        # when its schema is made available there via a DatasetSchemaScope (see
        # instance_serialization._datasets_for_instance_export). Delete in two tiers:
        #
        # 1. Datasets directly scoped to a deleted instance/org hold that customer's data, so they go
        #    unconditionally — even if their schema is also shared with a retained instance.
        # 2. Datasets owned only via schema scope (placeholders / not directly scoped to anything
        #    being deleted) are deleted only when no retained instance/org still has them, i.e. the
        #    schema isn't shared with a kept scope and the dataset isn't directly scoped to a kept one.
        belongs_to_kept = kept_scope | Q(schema_id__in=scope_ids.kept_schema_ids)
        schema_scoped_orphan = Q(schema_id__in=scope_ids.deleted_schema_ids) & ~belongs_to_kept
        to_delete = deleted_scope | schema_scoped_orphan

        # Capture the schemas/dimensions this trim may orphan *before* deleting anything, so the
        # orphan cleanups below only ever touch objects this trim affected — never a pre-existing
        # unscoped library/draft schema or dimension that has no datasets and was never part of a
        # deleted scope. A schema is affected if it lost a deleted DatasetSchemaScope
        # (deleted_schema_ids) or backed a dataset we're about to delete (e.g. a bare auto-created
        # schema with no scope of its own); a dimension is affected if it lost a deleted
        # DimensionScope.
        affected_schema_ids = set(scope_ids.deleted_schema_ids)
        affected_schema_ids.update(
            sid for sid in Dataset._default_manager.filter(to_delete).values_list('schema_id', flat=True) if sid is not None
        )
        affected_dimension_ids = set(
            DimensionScope._default_manager.filter(deleted_scope).values_list('dimension_id', flat=True),
        )

        # Deleting datasets cascades DataPoint, DatasetSourceReference and dataset revisions.
        _, by_type = Dataset._default_manager.filter(to_delete).delete()
        self.print_deleted_instances_by_model(by_type)

        # DataSources are only directly scoped. Delete them after the datasets, so references through
        # deleted datasets/data points have already been cascade-deleted. If a reference remains, it
        # belongs to retained data, so keep the source to avoid corrupting the retained dataset graph.
        _, by_type = DataSource._default_manager.filter(deleted_scope, references__isnull=True).delete()
        self.print_deleted_instances_by_model(by_type)

        # Scope link rows for the deleted instances/orgs.
        _, by_type = DimensionScope._default_manager.filter(deleted_scope).delete()
        self.print_deleted_instances_by_model(by_type)
        _, by_type = DatasetSchemaScope._default_manager.filter(deleted_scope).delete()
        self.print_deleted_instances_by_model(by_type)

        # Orphaned schemas: a schema is "owned" only through its DatasetSchemaScope links, so one
        # scoped solely to a now-deleted instance/org (including its placeholder datasets, removed
        # above) is left with no scopes and no datasets. Deleting it cascades its metrics and
        # schema-dimension links. Restrict to schemas this trim affected so a pre-existing unscoped,
        # dataset-less schema is never collateral; shared schemas keep at least one scope or dataset.
        _, by_type = DatasetSchema._default_manager.filter(
            pk__in=affected_schema_ids, scopes__isnull=True, datasets__isnull=True
        ).delete()
        self.print_deleted_instances_by_model(by_type)

        # Orphaned dimensions: deleting a DimensionScope does not delete its Dimension (the cascade
        # runs the other way), so dimensions scoped only to deleted instances/orgs would keep their
        # customer-specific category labels. Restrict to dimensions whose deleted scope we just
        # removed (so a pre-existing unscoped library dimension is never collateral) and, among
        # those, delete the ones that now have no scopes left, no surviving schema-dimension link
        # (DatasetSchemaDimension cascades from Dimension, so a kept schema would otherwise be
        # corrupted), and no surviving data point referencing their categories
        # (DataPointDimensionCategory.dimension_category is PROTECT). Deleting such a dimension
        # cascades its categories.
        _, by_type = (
            Dimension._default_manager
            .filter(pk__in=affected_dimension_ids, scopes__isnull=True, schemas__isnull=True)
            .exclude(categories__data_point_links__isnull=False)
            .delete()
        )
        self.print_deleted_instances_by_model(by_type)

    def _get_object_id_cast_field(self, model: type[Model]) -> Field[Any, Any]:
        pk_field: Field[Any, Any] = model._meta.pk
        target_field = getattr(pk_field, 'target_field', None)
        while target_field is not None:
            pk_field = target_field
            target_field = getattr(pk_field, 'target_field', None)
        return pk_field.clone()

    def delete_entries_for_missing_objects(self, model: type[Model]) -> None:
        # Delete rows of a content_type/object_id-keyed model (Wagtail Revision and ModelLogEntry)
        # whose referenced object no longer exists. We key on object existence rather than on the
        # `user` FK: a revision or log entry for a deleted instance/page/dataset can still have a
        # surviving author, and it carries serialized object data, so retaining it would leak the
        # deleted customer's data. Conversely, entries for *kept* objects are always preserved here
        # (their object still exists), so a kept page never loses its live/latest revision.
        #
        # Done per content type with a NOT EXISTS subquery, so we never materialise the live-object
        # ID set in Python or pass it as a giant IN predicate on production-sized dumps.
        content_type_ids = list(model._default_manager.values_list('content_type_id', flat=True).distinct())
        aggregated: dict[str, int] = {}

        for content_type_id in content_type_ids:
            target = ContentType.objects.get_for_id(content_type_id).model_class() if content_type_id is not None else None
            queryset = model._default_manager.filter(content_type_id=content_type_id)
            if target is None:
                # Stale content type (model removed from the codebase): the object cannot exist.
                _, by_type = queryset.delete()
            else:
                existing_object_subquery = target._default_manager.filter(
                    pk=Cast(OuterRef('object_id'), output_field=self._get_object_id_cast_field(target)),
                )
                _, by_type = (
                    queryset.annotate(has_live_object=Exists(existing_object_subquery)).filter(has_live_object=False).delete()
                )
            for model_name, n in by_type.items():
                aggregated[model_name] = aggregated.get(model_name, 0) + n

        self.print_deleted_instances_by_model(aggregated)

    def delete_page_log_entries_for_missing_pages(self) -> None:
        # PageLogEntry references its object through a constraint-less `page` FK rather than
        # content_type/object_id, so it needs its own orphan check. Same rationale as
        # delete_entries_for_missing_objects: drop entries for pages that no longer exist
        # (regardless of author), keep entries for surviving pages.
        live_pages = Page.objects.filter(pk=OuterRef('page_id'))
        _, by_type = PageLogEntry.objects.annotate(has_live_page=Exists(live_pages)).filter(has_live_page=False).delete()
        self.print_deleted_instances_by_model(by_type)

    def repair_has_unpublished_changes(self) -> None:
        # `latest_revision` is `on_delete=SET_NULL`, so deleting Revision rows leaves DraftStateMixin
        # instances with `latest_revision=NULL` while `has_unpublished_changes` stays at whatever it
        # was — possibly producing the impossible state `has_unpublished_changes=True AND
        # latest_revision IS NULL`. Fix the discrepancy here.
        for model in apps.get_models():
            if not issubclass(model, DraftStateMixin):
                continue
            updated_count = model._default_manager.filter(
                latest_revision__isnull=True,
                has_unpublished_changes=True,
            ).update(has_unpublished_changes=False)
            if updated_count:
                self.stdout.write(
                    f'Reset has_unpublished_changes on {updated_count} {model.__name__} instances with no latest_revision.',
                )

    def delete_thoroughly(self):
        self.delete_all(WagtailRevision)
        # The full revision purge nulls latest_revision on every DraftStateMixin instance, so repair
        # the has_unpublished_changes invariant for the pages/instances we're keeping.
        self.repair_has_unpublished_changes()
        self.delete_all(ModelLogEntry)
        self.delete_all(PageLogEntry)
        self.delete_all(LogEntry)

        self.delete_all(Association)
        self.delete_all(Nonce)
        self.delete_all(Code)
        self.delete_all(Partial)

        # oauth2_provider is a hard dependency (imported at module level), so its tokens always
        # exist and must be cleared regardless of whether the extensions package is installed.
        self.delete_all(RefreshToken)
        self.delete_all(AccessToken)

        if find_spec('kausal_paths_extensions') is not None:
            from kausal_paths_extensions.auth.models import AuthGrant, AuthIDToken

            self.delete_all(AuthIDToken)
            self.delete_all(AuthGrant)

    def _capture_scope_ids(self, instances_to_delete, orgs_to_delete) -> ScopeIds:
        # Capture the ids of the instances/orgs being deleted vs. retained *before* any deletion, so
        # the scoped-dataset cleanup (which runs after they are gone) can still resolve generic-FK
        # scopes and tell deleted-customer datasets apart from shared ones.
        instance_ct = ContentType.objects.get_for_model(InstanceConfig)
        org_ct = ContentType.objects.get_for_model(Organization)
        deleted_instance_ids = list(instances_to_delete.values_list('id', flat=True))
        deleted_org_ids = list(orgs_to_delete.values_list('id', flat=True))
        kept_instance_ids = list(
            InstanceConfig.objects.exclude(id__in=deleted_instance_ids).values_list('id', flat=True),
        )
        kept_org_ids = list(Organization.objects.exclude(id__in=deleted_org_ids).values_list('id', flat=True))

        # Schemas made available to a scope via DatasetSchemaScope, captured now because deleting an
        # instance cascade-deletes its scope rows through the GenericRelation (see ScopeIds docstring).
        def schema_ids_for(instance_ids: list[int], org_ids: list[int]) -> list[int]:
            scope_q = Q(scope_content_type=instance_ct, scope_id__in=instance_ids) | Q(
                scope_content_type=org_ct, scope_id__in=org_ids
            )
            return list(DatasetSchemaScope.objects.filter(scope_q).values_list('schema_id', flat=True))

        return ScopeIds(
            instance_ct=instance_ct,
            org_ct=org_ct,
            deleted_instance_ids=deleted_instance_ids,
            deleted_org_ids=deleted_org_ids,
            kept_instance_ids=kept_instance_ids,
            kept_org_ids=kept_org_ids,
            deleted_schema_ids=schema_ids_for(deleted_instance_ids, deleted_org_ids),
            kept_schema_ids=schema_ids_for(kept_instance_ids, kept_org_ids),
        )

    @transaction.atomic
    def delete_data(
        self,
        instances_to_delete,
        orgs_to_delete,
        thorough: bool = False,
    ):
        scope_ids = self._capture_scope_ids(instances_to_delete, orgs_to_delete)

        # Phase 1: Delete instances individually (signals active, custom delete() handles cleanup).
        # This cascades the NodeDataset/DatasetPort rows that PROTECT datasets, so the dataset
        # cleanup below can run without hitting ProtectedError.
        for instance in instances_to_delete:
            instance.delete()
            self.stdout.write(f'Deleted instance {instance.identifier}; information on deleted related rows not available.')

        # Phase 2: Bulk deletions with muted signals
        with factory.django.mute_signals(post_delete):
            # Delete organizations
            num_orgs = orgs_to_delete.count()
            orgs_to_delete.delete()
            self.stdout.write(f'Deleted {num_orgs} organizations; information on deleted related rows not available.')
            # Delete the dataset graph (datasets, sources, dimensions, schemas) owned by the now-deleted
            # instances/orgs. Runs after Phase 1 so the PROTECT-ing NodeDataset/DatasetPort rows are gone.
            self.delete_scoped_datasets(scope_ids)
            # Delete users without persons
            _, by_type = User.objects.filter(person__isnull=True).delete()
            self.print_deleted_instances_by_model(by_type)
            # Delete Wagtail revisions for objects that no longer exist (regardless of author), then
            # repair the draft-state invariant on any kept page whose latest_revision was affected.
            self.delete_entries_for_missing_objects(WagtailRevision)
            self.repair_has_unpublished_changes()
            # Delete Wagtail model/page log entries for objects that no longer exist
            self.delete_entries_for_missing_objects(ModelLogEntry)
            self.delete_page_log_entries_for_missing_pages()
            # Delete all logged requests
            self.delete_all(LoggedRequest)
            # Delete all sessions
            self.delete_all(Session)
            # Delete all image renditions (derived cache files, regenerated on demand)
            self.delete_all(get_image_model().get_rendition_model())

            # Phase 3: Thorough mode
            if thorough:
                self.delete_thoroughly()

    def print_deleted_instances_by_model(self, by_type):
        for model_name, n in by_type.items():
            self.stdout.write(f'Deleted {n} instances of {model_name}.')
