"""Resolve model qualifier catalogs and attach framework-owned evidence."""

from collections import defaultdict
from typing import TYPE_CHECKING

from django.db.models import Q

import polars as pl

from common import qualifiers
from frameworks.models import DataQualityScheme, Framework

if TYPE_CHECKING:
    from common.polars import PathsDataFrame
    from datasets.snapshot import DataPointEvidenceSnapshot, QualityLevelRef
    from nodes.models import InstanceConfig


def qualifier_catalog_for_instance(instance: InstanceConfig | None) -> qualifiers.QualifierCatalog:
    """Build the catalog for a member instance or its framework's template."""
    if instance is None:
        return qualifiers.BUILTIN_QUALIFIERS
    framework = Framework.objects.filter(Q(configs__instance_config=instance) | Q(template_instance=instance)).distinct().first()
    if framework is None:
        return qualifiers.BUILTIN_QUALIFIERS
    return qualifier_catalog_for_framework(framework)


def qualifier_catalog_for_framework(framework: Framework) -> qualifiers.QualifierCatalog:
    """
    Expose each scheme family, retaining exact grades from every version.

    The most recently created scheme version supplies defaults. Version strings
    are labels, not assumed to have lexical or semantic-version ordering.
    """
    families: dict[str, list[qualifiers.QualitySchemeDefinition]] = defaultdict(list)
    for scheme in DataQualityScheme.objects.filter(framework=framework).order_by('pk').prefetch_related('levels'):
        families[f'{framework.identifier}_{scheme.identifier}'].append(
            qualifiers.QualitySchemeDefinition(
                uuid=str(scheme.uuid),
                version=scheme.version,
                levels=tuple(
                    qualifiers.QualityLevelDefinition(str(level.uuid), level.identifier, float(level.score))
                    for level in scheme.levels.all()
                ),
            )
        )
    definitions = tuple(
        qualifiers.QualifierDefinition(
            name,
            qualifiers.Propagation.COVERED_SCORE,
            tuple(schemes),
            scheme_identifier=name.removeprefix(f'{framework.identifier}_'),
        )
        for name, schemes in sorted(families.items())
    )
    return qualifiers.QualifierCatalog((*qualifiers.BUILTIN_QUALIFIERS.definitions, *definitions))


def _portable_grade(
    ref: QualityLevelRef,
    catalog: qualifiers.QualifierCatalog,
) -> tuple[str, float] | None:
    for definition in catalog.assessments:
        identifier = ref.scheme
        if identifier == 'bisko' and definition.identifier == 'bisko_quality':
            identifier = 'quality'
        if definition.scheme_identifier != identifier:
            continue
        for scheme in definition.schemes:
            if scheme.version != ref.scheme_version:
                continue
            for level in scheme.levels:
                if level.identifier == ref.level:
                    return definition.identifier, level.score
    return None


def attach_evidence_qualifiers(
    frame: PathsDataFrame,
    evidence: list[DataPointEvidenceSnapshot],
    catalog: qualifiers.QualifierCatalog,
    *,
    portable: bool = False,
) -> PathsDataFrame:
    """Attach assessments by immutable grade UUID and the payload's natural cell key."""
    levels = {
        level.uuid: (definition.identifier, level.score)
        for definition in catalog.assessments
        for scheme in definition.schemes
        for level in scheme.levels
    }
    by_cell: dict[tuple[int, str, tuple[str, ...]], tuple[str, float]] = {}
    for item in evidence:
        if item.quality_level is None:
            continue
        resolved = levels.get(item.quality_level.uuid)
        if resolved is None and portable:
            resolved = _portable_grade(item.quality_level, catalog)
        if resolved is not None:
            by_cell[(item.point.year, item.point.metric, tuple(sorted(item.point.categories)))] = resolved
    keys = [int(row[0]) for row in frame.select('Year').iter_rows()]
    categories = (
        [tuple(sorted(str(c) for c in row if c is not None)) for row in frame.select(frame.dim_ids).iter_rows()]
        if frame.dim_ids
        else [()] * frame.height
    )
    for metric in frame.metric_cols:
        values = [by_cell.get((year, metric, cats)) for year, cats in zip(keys, categories, strict=True)]
        assessments = {}
        for definition in catalog.assessments:
            score = pl.Series(
                [entry[1] if entry is not None and entry[0] == definition.identifier else None for entry in values],
                dtype=pl.Float64,
            )
            # No evidence for this scheme is unknown, not an assessment of a different scheme.
            assessments[definition.identifier] = (
                pl
                .when(pl.lit(score).is_not_null())
                .then(qualifiers.covered_score(pl.lit(score), pl.lit(1.0)))
                .otherwise(pl.lit(None, dtype=qualifiers.COVERED_SCORE_DTYPE))
            )
        frame = frame.with_columns(
            qualifiers.make(
                reported=pl.col(metric).is_not_null(),
                catalog=catalog,
                assessments=assessments,
            ).alias(qualifiers.qualifier_column(metric))
        )
    return frame
