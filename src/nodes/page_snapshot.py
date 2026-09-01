"""
Serialize an instance's Wagtail page tree into a comparable snapshot.

This is **export/verification only** — it is not used to import or copy pages
(that goes through Wagtail's ``Page.copy``). Its purpose is to let a source and
its copy be compared structurally: node references (the ``OutcomePage`` FK and
``NodeChooserBlock`` PKs in StreamField bodies) are expressed by node
*identifier* rather than pk, so equivalent trees serialize identically across
instances.
"""

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from wagtail.blocks import ListBlock, StreamBlock, StructBlock
from wagtail.fields import StreamField

from nodes.blocks import NodeChooserBlock

if TYPE_CHECKING:
    from wagtail.models import Page

    from nodes.models import InstanceConfig


class PageSnapshot(BaseModel):
    """Structural snapshot of one Wagtail page (with its descendants)."""

    type: str  # specific page model name, e.g. 'OutcomePage'
    slug: str
    title: str
    outcome_node: str | None = None  # node identifier, for OutcomePage
    body: dict[str, Any] = Field(default_factory=dict)  # streamfield name → raw, node refs by identifier
    children: list['PageSnapshot'] = Field(default_factory=list)


def _streamfield_to_identifiers(block: Any, raw: Any, pk_to_id: dict[int, str]) -> Any:  # noqa: C901
    """Walk a StreamField raw value, replacing NodeChooser pks with node identifiers."""
    if isinstance(block, NodeChooserBlock):
        if raw in (None, ''):
            return raw
        try:
            return pk_to_id.get(int(raw), raw)
        except TypeError, ValueError:
            return raw
    if isinstance(block, StructBlock) and isinstance(raw, dict):
        return {
            name: (_streamfield_to_identifiers(child, raw[name], pk_to_id) if name in raw else None)
            for name, child in block.child_blocks.items()
            if name in raw
        }
    if isinstance(block, StreamBlock) and isinstance(raw, list):
        out = []
        for item in raw:
            child = block.child_blocks.get(item.get('type'))
            new_item = dict(item)
            new_item.pop('id', None)  # block ids are random per-copy; drop for comparability
            if child is not None:
                new_item['value'] = _streamfield_to_identifiers(child, item.get('value'), pk_to_id)
            out.append(new_item)
        return out
    if isinstance(block, ListBlock) and isinstance(raw, list):
        out = []
        for item in raw:
            if isinstance(item, dict) and 'value' in item:
                new_item = {k: v for k, v in item.items() if k != 'id'}
                new_item['value'] = _streamfield_to_identifiers(block.child_block, item['value'], pk_to_id)
                out.append(new_item)
            else:
                out.append(_streamfield_to_identifiers(block.child_block, item, pk_to_id))
        return out
    return raw


def build_instance_page_snapshots(ic: InstanceConfig) -> list[PageSnapshot]:
    """Serialize the instance's Wagtail page subtree (empty if it has no root page)."""
    if ic.root_page is None:
        return []
    from pages.models import OutcomePage

    pk_to_id = {nc.pk: nc.identifier for nc in ic.nodes.all()}

    def serialize(page: Page) -> PageSnapshot:
        specific = page.specific
        body: dict[str, Any] = {}
        for field in specific._meta.get_fields():
            if isinstance(field, StreamField):
                sv = getattr(specific, field.name)
                body[field.name] = _streamfield_to_identifiers(field.stream_block, sv.get_prep_value(), pk_to_id)
        outcome_node = pk_to_id.get(specific.outcome_node_id) if isinstance(specific, OutcomePage) else None
        return PageSnapshot(
            type=type(specific).__name__,
            slug=specific.slug,
            title=specific.title,
            outcome_node=outcome_node,
            body=body,
            children=[serialize(child) for child in page.get_children()],
        )

    return [serialize(ic.root_page)]
