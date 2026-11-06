"""
Compatibility boundary for the two BISKO certification entrypoints frozen at 136c060e.

Their YAML retains the reviewed model with municipal overlap disabled in both entrypoints. Translate
only authoring syntax here; retain each original requirement and its enforcement tier.
New models continue to use the strict shape-based YAML parser.
"""

from copy import deepcopy
from typing import TYPE_CHECKING, Any
from uuid import uuid3

from kausal_common.datasets.category_domain import DatasetCategoryDomainSpec

if TYPE_CHECKING:
    from uuid import UUID

CERTIFICATION_INSTANCES = frozenset({'bisko', 'pruefstadt-bisko'})


def certification_yaml(config: dict[str, Any], instance_uuid: UUID) -> dict[str, Any]:
    """Compile the frozen YAML's domains and named rule members to native shape identities."""
    if config.get('id') not in CERTIFICATION_INSTANCES:
        raise ValueError('Expected a frozen BISKO certification entrypoint')
    result = deepcopy(config)
    for parameter in result.get('params', []):
        if parameter['id'] == 'municipal_facilities_included_in_commerce':
            parameter['type'] = 'bool'
    for dataset in result.get('datasets', []):
        if 'category_domain' in dataset:
            shape = _compile_domain(dataset, instance_uuid)
            if shape is not None:
                result.setdefault('shapes', []).append(shape)
    return result


def _compile_domain(dataset: dict[str, Any], instance_uuid: UUID) -> dict[str, Any] | None:
    domain = DatasetCategoryDomainSpec.model_validate(dataset.pop('category_domain'))
    if not domain.combinations:
        return None
    identifier = f'certification/{dataset["id"]}'
    dataset['shape'] = identifier
    members = {item.id: str(uuid3(instance_uuid, f'shape:{identifier}:combination:{item.id}')) for item in domain.combinations}
    for metric in dataset.get('metrics', []):
        for rule in metric.get('validation_rules', []):
            if rule['kind'] != 'required_combinations':
                continue
            for group in rule['groups']:
                group['combinations'] = [members[member] for member in group['combinations']]
    return {
        'id': identifier,
        'dimensions': list(domain.combinations[0].categories),
        'closed': domain.mode == 'closed',
        'combinations': [item.model_dump() for item in domain.combinations],
    }
