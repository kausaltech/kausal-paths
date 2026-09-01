from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodes.defs.transform_def import EdgeTransformOp

    from .dimensions import DimensionCategory
    from .node import Node


@dataclass
class EdgeDimension:
    categories: list[DimensionCategory]
    exclude: bool
    flatten: bool


@dataclass
class Edge:
    input_node: Node
    output_node: Node
    tags: list[str] = field(default_factory=list)
    from_dimensions: dict[str, EdgeDimension] = field(default_factory=dict)
    to_dimensions: dict[str, EdgeDimension] | None = None
    metrics: list[str] = field(default_factory=list)

    source_transforms: list[EdgeTransformOp] | None = field(default=None, repr=False)
    """The stored typed pipeline this edge was built from, when one exists.

    Snapshot-built edges keep their authored op order here; the dimension maps
    cannot carry it (their key order serves the exporter's port-dimension
    derivation, which follows declaration order, not op order)."""

    # These are used only temporarily at export time to store the port IDs.
    _to_port_ids: list[str] = field(default_factory=list, init=False, repr=False)
    _from_output_metric_ids: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        self.tags = self.tags.copy()
        self.from_dimensions = self.from_dimensions.copy()
        if self.to_dimensions is not None:
            self.to_dimensions = self.to_dimensions.copy()
        self.metrics = self.metrics.copy()

    def to_transforms(self) -> list[EdgeTransformOp]:
        """Return the edge's transformation pipeline (stored ops, or derived from the dimension maps)."""
        from nodes.defs.transform_def import AssignDimensionOp, FilterDimensionOp

        if self.source_transforms is not None:
            return list(self.source_transforms)

        transforms: list[EdgeTransformOp] = []

        for dim_id, ed in self.from_dimensions.items():
            cat_refs = [cat.id for cat in ed.categories]
            transforms.append(
                FilterDimensionOp(
                    dimension=dim_id,
                    categories=cat_refs,
                    flatten=ed.flatten,
                    exclude=ed.exclude,
                )
            )

        if self.to_dimensions:
            for dim_id, ed in self.to_dimensions.items():
                if not ed.categories:
                    # A bare to-dimension is a declaration on the consuming
                    # port. It has no executable edge operation.
                    continue
                if len(ed.categories) != 1:
                    raise ValueError(f'to_dimensions can have only one category for now (got {len(ed.categories)} for {dim_id})')
                transforms.append(
                    AssignDimensionOp(
                        dimension=dim_id,
                        category=ed.categories[0].id,
                    )
                )

        return transforms
