"""
Legacy port-role inference contract.

Migration-only machinery: specs synced before explicit roles carry one
anonymous port per binding, and ``Node.infer_legacy_port_roles()`` lets the
class that knows its input algebra classify those ports into its declared
roles. Implementing this hook for a *new* node class is always wrong — new
classes declare roles at port creation. The hook, its implementations, and
this module all go away once persisted ports carry explicit roles.

The framework (``NodeMeta``) computes the candidate ports — authored roles
and declaration-identifier matches are filtered out before the class sees
anything — formats uniform diagnostics from the structured result, and
validates that every inferred role exists in the class declarations.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

    from nodes.defs.port_def import InputPortDef


@dataclass(frozen=True, slots=True)
class InferredPortRole:
    port_id: UUID
    role: str
    basis: str
    """Human-readable justification, e.g. "binding tag 'impute'"."""


@dataclass(frozen=True, slots=True)
class UnclassifiedPort:
    port_id: UUID
    reason: str


@dataclass
class PortRoleInferenceResult:
    inferred: list[InferredPortRole] = field(default_factory=list)
    unclassified: list[UnclassifiedPort] = field(default_factory=list)

    def classify(self, port: InputPortDef, role: str, basis: str) -> None:
        self.inferred.append(InferredPortRole(port_id=port.id, role=role, basis=basis))

    def refuse(self, port: InputPortDef, reason: str) -> None:
        self.unclassified.append(UnclassifiedPort(port_id=port.id, reason=reason))
