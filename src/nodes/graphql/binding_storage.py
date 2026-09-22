"""Edit local-node bindings in an effective graph without writing through template definitions."""

from typing import TYPE_CHECKING

from django.db import transaction
from graphql import GraphQLError

from kausal_common.strawberry.errors import GraphQLValidationError

from nodes.change_ops import gql_change_operation, record_change
from nodes.constraints.validation import InstanceConstraintError
from nodes.graphql.types.constraints import ConstraintViolationsType
from nodes.instance_graph import NodeMeta
from nodes.instance_serialization import build_instance_snapshot
from nodes.models import InstanceConfig, NodeConfig, PreferredInstanceSource
from nodes.template_graph import replace_input_port_bindings

if TYPE_CHECKING:
    from uuid import UUID

    from paths import gql

    from nodes.instance_serialization import InputBindingSnapshot


class LocalBindingEditor:
    """Legacy per-binding mutations projected onto the complete effective port selection."""

    @staticmethod
    def _lock(instance: InstanceConfig) -> None:
        # Incremental edits must read their members after taking the instance
        # lock; otherwise concurrent appends could overwrite each other.
        locked = InstanceConfig.objects.select_for_update().get(pk=instance.pk)
        instance.template_revision = locked.template_revision
        instance.cache_invalidated_at = locked.cache_invalidated_at
        instance.is_locked = locked.is_locked

    @classmethod
    def require_node(cls, info: gql.Info, instance: InstanceConfig, node: NodeConfig) -> None:
        resources = info.context.instance_resources
        assert resources is not None
        context = resources.node_edit_context(info, instance, PreferredInstanceSource.DRAFT)
        if not NodeMeta.can_edit(context, inherited=node.instance_id != instance.pk, node_is_editable=node.is_editable):
            raise GraphQLValidationError(
                info, 'Cannot edit bindings on a template-sourced node; use setInputPortBindings for local inputs'
            )

    @classmethod
    def find(cls, info: gql.Info, instance: InstanceConfig, binding_id: str) -> tuple[NodeConfig, InputBindingSnapshot]:
        snapshot = info.context.require_instance_snapshot(instance, source=PreferredInstanceSource.DRAFT)
        binding = next((b for b in snapshot.bindings if str(b.uuid) == binding_id), None)
        if binding is None:
            raise GraphQLError('Binding not found')
        node = instance.nodes.filter(uuid=binding.node_id).first()
        if node is None:
            raise GraphQLValidationError(info, 'Cannot edit a template-sourced node through the local binding editor')
        cls.require_node(info, instance, node)
        return node, binding

    @classmethod
    def members(cls, instance: InstanceConfig, node: NodeConfig, port_id: UUID) -> list[InputBindingSnapshot]:
        snapshot = build_instance_snapshot(instance)
        return [b for b in snapshot.bindings if b.node_id == node.uuid and b.port_id == port_id]

    @classmethod
    def save(
        cls,
        info: gql.Info,
        instance: InstanceConfig,
        node: NodeConfig,
        port_id: UUID,
        bindings: list[InputBindingSnapshot],
    ) -> ConstraintViolationsType | None:
        cls.require_node(info, instance, node)
        bindings = [binding.model_copy(update={'position': pos}) for pos, binding in enumerate(bindings)]
        before = cls.members(instance, node, port_id)
        try:
            with gql_change_operation(info, instance, action='node.input_bindings.set'):
                replace_input_port_bindings(instance, node.uuid, port_id, bindings)
                record_change(
                    instance,
                    action='node.input_bindings.set',
                    before={'bindings': [b.model_dump(mode='json') for b in before]},
                    after={'bindings': [b.model_dump(mode='json') for b in bindings]},
                )
        except InstanceConstraintError as exc:
            return ConstraintViolationsType.from_conflicts(exc.conflicts)
        return None

    @classmethod
    @transaction.atomic
    def add(
        cls,
        info: gql.Info,
        instance: InstanceConfig,
        node: NodeConfig,
        binding: InputBindingSnapshot,
        *,
        replace: bool,
    ) -> ConstraintViolationsType | None:
        cls._lock(instance)
        assert node.spec is not None
        port = node.spec.input_port_by_id[binding.port_id]
        members = cls.members(instance, node, binding.port_id)
        if replace:
            if port.multi:
                raise GraphQLValidationError(info, 'replace is ambiguous on a port accepting multiple bindings')
            members = []
        elif members and not port.multi:
            raise GraphQLValidationError(info, 'Input port already has a binding')
        return cls.save(info, instance, node, binding.port_id, [*members, binding])

    @classmethod
    @transaction.atomic
    def update(
        cls,
        info: gql.Info,
        instance: InstanceConfig,
        node: NodeConfig,
        binding: InputBindingSnapshot,
    ) -> ConstraintViolationsType | None:
        cls._lock(instance)
        members = cls.members(instance, node, binding.port_id)
        return cls.save(info, instance, node, binding.port_id, [binding if b.uuid == binding.uuid else b for b in members])

    @classmethod
    @transaction.atomic
    def delete(cls, info: gql.Info, instance: InstanceConfig, node: NodeConfig, binding: InputBindingSnapshot) -> None:
        cls._lock(instance)
        members = cls.members(instance, node, binding.port_id)
        violations = cls.save(info, instance, node, binding.port_id, [b for b in members if b.uuid != binding.uuid])
        if violations is not None:
            raise GraphQLValidationError(info, 'Removing this binding introduces structural conflicts')
