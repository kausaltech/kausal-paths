"""Rekeying a document by the uuid kinds of its fields."""

from typing import Annotated, ClassVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

import pytest

from paths.rekey import rekeyed
from paths.uuid_kinds import Identity, Provenance, Ref, Token, unmarked_uuid_fields

pytestmark = pytest.mark.django_db

NodeId = Annotated[UUID, Identity('node')]
NodeRef = Annotated[UUID, Ref('node')]
NodeCopyOf = Annotated[UUID, Provenance('node')]
PortId = Annotated[UUID, Identity('port')]
DimensionId = Annotated[UUID, Identity('dimension')]
DimensionRef = Annotated[UUID, Ref('dimension')]


class Port(BaseModel):
    id: PortId


class Node(BaseModel):
    uuid: NodeId
    copy_of: NodeCopyOf | None = None
    parent: NodeRef | None = None
    ports: list[Port] = Field(default_factory=list)
    label: str = ''


class Dimension(BaseModel):
    id: DimensionId
    shared: bool = False

    def rekey_owns(self, field: str, owned: bool) -> bool:
        return owned and not self.shared


class Template(BaseModel):
    nodes: list[Node] = Field(default_factory=list)


class Document(BaseModel):
    __rekey_foreign__: ClassVar[frozenset[str]] = frozenset({'template'})

    nodes: list[Node] = Field(default_factory=list)
    dimensions: list[Dimension] = Field(default_factory=list)
    values: dict[DimensionRef, float] = Field(default_factory=dict)
    token: Annotated[UUID, Token()] | None = None
    template: Template | None = None


def test_owned_identities_get_new_uuids_and_references_follow() -> None:
    parent, child = Node(uuid=uuid4(), ports=[Port(id=uuid4())]), Node(uuid=uuid4())
    child = child.model_copy(update={'parent': parent.uuid})
    copy, rekeying = rekeyed(Document(nodes=[parent, child]))

    new_parent, new_child = copy.nodes
    assert new_parent.uuid == rekeying.mapping[parent.uuid] != parent.uuid
    assert new_child.parent == new_parent.uuid
    assert new_parent.ports[0].id != parent.ports[0].id
    assert rekeying.outside() == {}


def test_provenance_points_at_the_source_and_tokens_are_dropped() -> None:
    node = Node(uuid=uuid4(), copy_of=uuid4())
    copy, _ = rekeyed(Document(nodes=[node], token=uuid4()))
    assert copy.nodes[0].copy_of == node.uuid
    assert copy.token is None


def test_provenance_is_left_alone_when_asked() -> None:
    node = Node(uuid=uuid4(), copy_of=uuid4())
    copy, _ = rekeyed(Document(nodes=[node]), set_provenance=False)
    assert copy.nodes[0].copy_of == node.copy_of


def test_a_reference_to_something_not_owned_is_kept_and_reported() -> None:
    elsewhere = uuid4()
    copy, rekeying = rekeyed(Document(nodes=[Node(uuid=uuid4(), parent=elsewhere)]))
    assert copy.nodes[0].parent == elsewhere
    assert rekeying.outside() == {'node': {elsewhere}}


def test_dict_keys_are_references_too() -> None:
    dimension = Dimension(id=uuid4())
    copy, rekeying = rekeyed(Document(dimensions=[dimension], values={dimension.id: 1.0}))
    assert copy.values == {rekeying.mapping[dimension.id]: 1.0}


def test_a_model_can_disown_its_identities() -> None:
    """A framework's dimension inside an instance catalog is referred to, not copied."""
    shared, own = Dimension(id=uuid4(), shared=True), Dimension(id=uuid4())
    copy, rekeying = rekeyed(Document(dimensions=[shared, own], values={shared.id: 1.0, own.id: 2.0}))
    assert [d.id for d in copy.dimensions] == [shared.id, rekeying.mapping[own.id]]
    assert copy.values == {shared.id: 1.0, rekeying.mapping[own.id]: 2.0}
    assert rekeying.outside() == {'dimension': {shared.id}}


def test_foreign_fields_pass_through_and_their_identities_stay_theirs() -> None:
    """An identity a bundled template defines is an override where the document repeats it."""
    inherited = Node(uuid=uuid4(), label='template')
    override = inherited.model_copy(update={'label': 'closed locally'})
    local = Node(uuid=uuid4(), parent=inherited.uuid)
    template = Template(nodes=[inherited])
    copy, rekeying = rekeyed(Document(nodes=[override, local], template=template))

    assert copy.template is template
    assert copy.nodes[0].uuid == inherited.uuid
    assert copy.nodes[1].parent == inherited.uuid
    assert copy.nodes[1].uuid != local.uuid
    assert rekeying.outside() == {}, 'bundled identities need not exist where the copy goes'


def test_seed_fixes_new_uuids_in_advance() -> None:
    node = Node(uuid=uuid4())
    target = uuid4()
    copy, _ = rekeyed(Document(nodes=[node]), seed={node.uuid: target})
    assert copy.nodes[0].uuid == target


def test_unchanged_parts_are_not_rebuilt() -> None:
    template = Template(nodes=[Node(uuid=uuid4())])
    document = Document(template=template, dimensions=[Dimension(id=uuid4(), shared=True)])
    copy, _ = rekeyed(document)
    assert copy.dimensions[0] is document.dimensions[0]


def test_a_uuid_without_a_kind_is_reported() -> None:
    class Loose(BaseModel):
        id: UUID
        node: Node

    assert unmarked_uuid_fields(Loose) == ['Loose.id']


def test_two_kinds_on_one_uuid_are_refused() -> None:
    from paths.uuid_kinds import shape_of

    with pytest.raises(TypeError, match='more than one kind'):
        shape_of(Annotated[UUID, Identity('node'), Ref('node')])
