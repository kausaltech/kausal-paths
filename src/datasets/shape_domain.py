"""
A dataset's entry form, resolved from its shape reference in the dataset's own instance.

The reference is stored on the dataset row (`Dataset.spec['shape']`); the combinations are not
stored anywhere. Municipalities share their schemas with the template, and each one's record of an
extension point may add different combinations, so the result depends on the instance and is
computed on read. See `docs/architecture/shapes.md`.
"""

from functools import lru_cache
from typing import TYPE_CHECKING, Final
from uuid import UUID

from django.contrib.contenttypes.models import ContentType
from wagtail.models import Revision

from kausal_common.datasets.category_domain import DatasetCategoryCombination, DatasetCategoryDomain

from frameworks.catalogue import dimension_scopes
from nodes.defs.shape_defs import ShapeSpec
from nodes.shapes import EffectiveShape, resolve_shapes
from nodes.template_spec import compose_shapes

if TYPE_CHECKING:
    from kausal_common.datasets.models import Dataset

    from nodes.models import InstanceConfig

SHAPE_SPEC_KEY: Final = 'shape'


class ShapeDomainError(ValueError):
    pass


def dataset_shape_id(dataset: Dataset) -> UUID | None:
    value = (dataset.spec or {}).get(SHAPE_SPEC_KEY)
    return UUID(value) if value else None


@lru_cache(maxsize=64)
def _template_shapes(revision_id: int) -> tuple[ShapeSpec, ...]:
    """Read a published revision's own shapes; revisions are immutable, so this caches safely."""
    spec = Revision.objects.get(pk=revision_id).content['model_snapshot']['structured']['spec']
    return tuple(ShapeSpec.model_validate(item) for item in spec.get('shapes', []))


def instance_shapes(instance: InstanceConfig) -> dict[UUID, EffectiveShape]:
    """
    Resolve an instance's effective shapes from its local spec and its pinned template revision.

    Composes only the shapes, so it needs neither the runtime nor the template's whole snapshot.
    """
    local = list(instance.ensure_spec().shapes)
    if instance.template_revision_id is not None:
        local = compose_shapes(list(_template_shapes(instance.template_revision_id)), local)
    return resolve_shapes(local)


class CategoryDomainResolver:
    """
    Resolve datasets' entry domains, caching each instance's shapes and dimension catalogue.

    Use one resolver for a batch of datasets; a dataset without a shape reference keeps its
    schema's stored domain.
    """

    def __init__(self) -> None:
        self._instances: dict[int, InstanceConfig] = {}
        self._shapes: dict[int, dict[UUID, EffectiveShape]] = {}
        self._categories: dict[int, dict[str, tuple[UUID, dict[str, UUID]]]] = {}

    def for_dataset(self, dataset: Dataset) -> DatasetCategoryDomain:
        shape_id = dataset_shape_id(dataset)
        if shape_id is None:
            schema = dataset.schema
            return schema.category_domain if schema is not None else DatasetCategoryDomain()
        return self.for_shape(dataset.scope_instance, shape_id, where=f'Dataset {dataset.identifier or dataset.uuid}')

    def instance(self, scope_type_id: int, scope_id: int) -> InstanceConfig:
        """Look up a shaped dataset's instance by its scope columns, once per instance."""
        # nodes.models imports the dataset validation that imports this module.
        from nodes.models import InstanceConfig

        if scope_type_id != ContentType.objects.get_for_model(InstanceConfig).pk:
            raise ShapeDomainError('Only datasets scoped to an instance can refer to a shape')
        instance = self._instances.get(scope_id)
        if instance is None:
            instance = self._instances[scope_id] = InstanceConfig.objects.get(pk=scope_id)
        return instance

    def for_shape(self, instance: InstanceConfig, shape_id: UUID, *, where: str = 'A dataset') -> DatasetCategoryDomain:
        shapes = self._shapes.get(instance.pk)
        if shapes is None:
            shapes = self._shapes[instance.pk] = instance_shapes(instance)
        shape = shapes.get(shape_id)
        if shape is None:
            raise ShapeDomainError(f'{where} refers to shape {shape_id}, which {instance.identifier} does not declare')
        return self._compile(instance, shape, where)

    def _compile(self, instance: InstanceConfig, shape: EffectiveShape, where: str) -> DatasetCategoryDomain:
        categories = self._categories.get(instance.pk)
        if categories is None:
            scopes = dimension_scopes(instance).select_related('dimension').prefetch_related('dimension__categories')
            categories = self._categories[instance.pk] = {
                scope.identifier: (
                    scope.dimension.uuid,
                    {category.identifier: category.uuid for category in scope.dimension.categories.all() if category.identifier},
                )
                for scope in scopes
                if scope.identifier
            }
        combinations: list[DatasetCategoryCombination] = []
        for combination in shape.combinations:
            coordinates: dict[UUID, UUID] = {}
            for dimension_id, category_id in combination.categories.items():
                dimension = categories.get(dimension_id)
                if dimension is None or category_id not in dimension[1]:
                    msg = (
                        f'{where}: shape {shape.spec.label} names {dimension_id}:{category_id}, '
                        f'which {instance.identifier} has no row for; sync the instance'
                    )
                    raise ShapeDomainError(msg)
                coordinates[dimension[0]] = dimension[1][category_id]
            combinations.append(
                DatasetCategoryCombination(
                    id=combination.uuid,
                    identifier=combination.identifier or str(combination.uuid),
                    categories=coordinates,
                )
            )
        return DatasetCategoryDomain(mode='closed' if shape.closed else 'open', combinations=combinations)


def dataset_category_domain(dataset: Dataset, resolver: CategoryDomainResolver | None = None) -> DatasetCategoryDomain:
    """Return the dataset's entry domain: its shape's combinations, or its schema's stored domain."""
    return (resolver or CategoryDomainResolver()).for_dataset(dataset)
