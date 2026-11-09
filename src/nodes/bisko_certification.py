"""
Compatibility boundary for the two BISKO certification entrypoints frozen at d5ed3448.

Their YAML retains the reviewed model with municipal overlap disabled in both entrypoints. Translate
only authoring syntax here; retain each original requirement and its enforcement tier.
New models continue to use the strict shape-based YAML parser.

The municipal models in `FROZEN_MODULE_INSTANCES` include the same frozen module
(`configs/modules/bisko/model.yaml`), so their YAML passes through the same translation. They
were moved to `bisko-base` on 6 Nov 2026 and moved back: its data-quality nodes read
`bisko_quality` qualifiers, which exist only for members of the BISKO framework, so every
stand-alone city failed `total_weighted_data_quality`. They return to `bisko-base` once that
works for them or once they join the framework. They are listed explicitly, never inferred, and
they are not certification instances: `setup_bisko` refuses only `CERTIFICATION_INSTANCES`.
"""

from copy import deepcopy
from typing import TYPE_CHECKING, Any
from uuid import uuid3

from kausal_common.datasets.category_domain import DatasetCategoryDomainSpec

if TYPE_CHECKING:
    from uuid import UUID

CERTIFICATION_INSTANCES = frozenset({'bisko', 'pruefstadt-bisko'})

FROZEN_MODULE_INSTANCES = CERTIFICATION_INSTANCES | frozenset({
    'augsburg-bisko',
    'bayreuth-bisko',
    'duesseldorf-bisko',
    'mainz-bisko',
    'mainz-dev',
    'reutlingen-klimabilanz',
    'schwerin-bisko',
    'testcity-bisko',
})
"""Entrypoints whose model is the frozen module, and whose YAML `certification_yaml` translates."""


def certification_yaml(config: dict[str, Any], instance_uuid: UUID) -> dict[str, Any]:
    """Compile the frozen YAML's domains and named rule members to native shape identities."""
    if config.get('id') not in FROZEN_MODULE_INSTANCES:
        raise ValueError('Expected an entrypoint of the frozen BISKO module')
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
