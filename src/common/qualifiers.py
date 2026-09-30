"""
Per-metric assessments and reporting status, carried with values.

The metric/qualifier pairing belongs to the dataframe contract. Covered scores
are reduced as a pair; their weighting and combination rules belong to the
assessment. Reporting status is independent of grading and uses three-valued
AND over contributing data. See docs/architecture/metric-dataframe.md.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, TypedDict

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import JsonValue

    from common.polars import PathsDataFrame


QUALIFIER_SUFFIX = '__qual'
QUALITY = 'quality'
SCORE = 'score'
COVERAGE = 'coverage'
REPORTED = 'reported'
# Change this when either the stored shape or propagation semantics change.
QUALIFIER_VERSION = 5


class CoveredScore(TypedDict):
    score: float | None
    coverage: float | None


COVERED_SCORE_DTYPE = pl.Struct({SCORE: pl.Float64, COVERAGE: pl.Float64})


class Propagation(StrEnum):
    REPORTED = 'reported'
    COVERED_SCORE = 'covered_score'


@dataclass(frozen=True)
class QualityLevelDefinition:
    uuid: str
    identifier: str
    score: float


@dataclass(frozen=True)
class QualitySchemeDefinition:
    uuid: str
    version: str
    levels: tuple[QualityLevelDefinition, ...]


@dataclass(frozen=True)
class QualifierDefinition:
    identifier: str
    propagation: Propagation
    schemes: tuple[QualitySchemeDefinition, ...] = ()
    scheme_identifier: str | None = None

    @property
    def dtype(self) -> pl.DataType:
        return pl.Boolean() if self.propagation == Propagation.REPORTED else COVERED_SCORE_DTYPE

    def hash_data(self) -> dict[str, JsonValue]:
        return {
            'identifier': self.identifier,
            'scheme_identifier': self.scheme_identifier,
            'propagation': self.propagation,
            'schemes': [
                {
                    'uuid': s.uuid,
                    'version': s.version,
                    'levels': [[level.uuid, level.identifier, level.score] for level in s.levels],
                }
                for s in self.schemes
            ],
        }


@dataclass(frozen=True)
class QualifierCatalog:
    definitions: tuple[QualifierDefinition, ...] = ()

    def __post_init__(self) -> None:
        names = [d.identifier for d in self.definitions]
        if len(names) != len(set(names)):
            raise ValueError('Qualifier identifiers must be unique')

    def __getitem__(self, identifier: str) -> QualifierDefinition:
        for definition in self.definitions:
            if definition.identifier == identifier:
                return definition
        raise KeyError(identifier)

    @property
    def dtype(self) -> pl.Struct:
        return pl.Struct({d.identifier: d.dtype for d in self.definitions})

    @property
    def assessments(self) -> tuple[QualifierDefinition, ...]:
        return tuple(d for d in self.definitions if d.propagation == Propagation.COVERED_SCORE)

    def hash_data(self) -> list[JsonValue]:
        return [d.hash_data() for d in self.definitions]

    @classmethod
    def from_dtype(cls, dtype: pl.DataType) -> QualifierCatalog:
        """Resolve the fixed propagation mechanisms encoded by a qualifier payload."""
        if not isinstance(dtype, pl.Struct):
            raise TypeError('A qualifier column must be a Struct')
        definitions = []
        for field in dtype.fields:
            if field.name == REPORTED and field.dtype == pl.Boolean:
                mechanism = Propagation.REPORTED
            elif field.dtype == COVERED_SCORE_DTYPE:
                mechanism = Propagation.COVERED_SCORE
            else:
                raise ValueError(f'Unknown qualifier payload: {field}')
            definitions.append(QualifierDefinition(field.name, mechanism))
        return cls(tuple(definitions))


BUILTIN_QUALIFIERS = QualifierCatalog((QualifierDefinition(REPORTED, Propagation.REPORTED),))
# Only built-ins are global; quality fields come from Context.qualifiers.
QUALIFIER_DTYPE = BUILTIN_QUALIFIERS.dtype


def catalog_for_frames(*frames: PathsDataFrame) -> QualifierCatalog:
    definitions: dict[str, QualifierDefinition] = {}
    for frame in frames:
        for col in frame.qualifier_cols.values():
            for definition in QualifierCatalog.from_dtype(frame.schema[col]).definitions:
                previous = definitions.get(definition.identifier)
                if previous is not None and previous.propagation != definition.propagation:
                    raise ValueError(f'Conflicting qualifier {definition.identifier}')
                definitions[definition.identifier] = definition
    return QualifierCatalog(tuple(definitions.values()))


def align_frame(frame: PathsDataFrame, catalog: QualifierCatalog) -> PathsDataFrame:
    expressions = []
    for col in frame.qualifier_cols.values():
        if frame.schema[col] == catalog.dtype:
            continue
        present = {d.identifier for d in QualifierCatalog.from_dtype(frame.schema[col]).definitions}
        expressions.append(
            pl
            .struct([
                (pl.col(col).struct.field(d.identifier) if d.identifier in present else pl.lit(None, dtype=d.dtype)).alias(
                    d.identifier
                )
                for d in catalog.definitions
            ])
            .cast(catalog.dtype)
            .alias(col)
        )
    return frame.with_columns(expressions) if expressions else frame


def qualifier_column(metric_col: str) -> str:
    return f'{metric_col}{QUALIFIER_SUFFIX}'


def qualified_metric(col: str) -> str | None:
    if not col.endswith(QUALIFIER_SUFFIX) or col == QUALIFIER_SUFFIX:
        return None
    return col[: -len(QUALIFIER_SUFFIX)]


def covered_score(score: pl.Expr, coverage: pl.Expr) -> pl.Expr:
    """Construct an assessment atomically; no assessed weight means no score."""
    return pl.struct(
        pl.when(coverage > 0).then(score).otherwise(pl.lit(None, dtype=pl.Float64)).cast(pl.Float64).alias(SCORE),
        coverage.cast(pl.Float64).alias(COVERAGE),
    )


def make(
    quality: pl.Expr | None = None,
    reported: pl.Expr | None = None,
    *,
    supplied: pl.Expr | None = None,
    catalog: QualifierCatalog | None = None,
    assessments: dict[str, pl.Expr] | None = None,
) -> pl.Expr:
    """
    Construct a source qualifier in the resolved model catalog.

    ``quality`` and ``supplied`` support the previous construction API; runtime
    bindings use named assessments and an explicit catalog.
    """
    if supplied is not None:
        if reported is not None:
            raise ValueError('Specify reported, not both reported and supplied')
        reported = supplied
    assessments = dict(assessments or {})
    if quality is not None:
        q = quality.cast(pl.Float64)
        assessments[QUALITY] = covered_score(q, q.is_not_null().cast(pl.Float64))
    if catalog is None:
        catalog = QualifierCatalog((
            *BUILTIN_QUALIFIERS.definitions,
            *(QualifierDefinition(name, Propagation.COVERED_SCORE) for name in assessments),
        ))
    fields = []
    for definition in catalog.definitions:
        if definition.propagation == Propagation.REPORTED:
            expr = reported if reported is not None else pl.lit(None, dtype=pl.Boolean)
        else:
            expr = assessments.get(definition.identifier, pl.lit(None, dtype=COVERED_SCORE_DTYPE))
        fields.append(expr.alias(definition.identifier))
    return pl.struct(fields).cast(catalog.dtype)


def _fields(qual: str | None, name: str = QUALITY) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    if qual is None:
        return pl.lit(None, dtype=pl.Float64), pl.lit(0.0), pl.lit(None, dtype=pl.Boolean)
    col = pl.col(qual)
    assessment = col.struct.field(name)
    score = assessment.struct.field(SCORE)
    coverage = pl.when(score.is_null()).then(0.0).otherwise(assessment.struct.field(COVERAGE).fill_null(0.0))
    return score, coverage, col.struct.field(REPORTED)


def _weight(value: str) -> pl.Expr:
    return pl.col(value).cast(pl.Float64).abs().fill_nan(0.0).fill_null(0.0)


def _reported_all(reported: pl.Expr) -> pl.Expr:
    """Reduce reporting flags without discarding unknown contributions."""
    return (
        pl
        .when((reported == False).any())  # noqa: E712
        .then(pl.lit(value=False))
        .when((reported.len() == 0) | reported.is_null().any())
        .then(pl.lit(None, dtype=pl.Boolean))
        .otherwise(reported.all())
    )


def _reduce_score(value: str, qual: str, name: str) -> pl.Expr:
    """
    Energy-weighted assessment and reporting status of contributing cells.

    Reported zeros count for reporting status. Null values do not contribute.
    An all-zero group has no energy-weighted assessment.
    """
    q, g, _r = _fields(qual, name)
    w = _weight(value)
    wg = w * g
    coverage = pl.when(w.sum() > 0).then(wg.sum() / w.sum()).otherwise(pl.lit(None, dtype=pl.Float64))
    score = pl.when(wg.sum() > 0).then((wg * q.fill_null(0.0)).sum() / wg.sum()).otherwise(pl.lit(None, dtype=pl.Float64))
    return covered_score(score, coverage).alias(name)


def _combine_score(left_value: str, left_qual: str | None, right_value: str, right_qual: str | None, name: str) -> pl.Expr:
    """Assessment of a sum; absent outer-join sides contribute nothing."""
    lq, lg, _lr = _fields(left_qual, name)
    rq, rg, _rr = _fields(right_qual, name)
    lw = _weight(left_value)
    rw = _weight(right_value)
    lwg, rwg = lw * lg, rw * rg
    coverage = pl.when(lw + rw > 0).then((lwg + rwg) / (lw + rw)).otherwise(pl.lit(None, dtype=pl.Float64))
    score = (
        pl
        .when(lwg + rwg > 0)
        .then((lwg * lq.fill_null(0.0) + rwg * rq.fill_null(0.0)) / (lwg + rwg))
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return covered_score(score, coverage).alias(name)


def _field(qual: str | None, definition: QualifierDefinition) -> pl.Expr:
    return pl.col(qual).struct.field(definition.identifier) if qual else pl.lit(None, dtype=definition.dtype)


def reduce_sum(value: str, qual: str, catalog: QualifierCatalog) -> pl.Expr:
    return pl.struct([
        _reported_all(pl.col(qual).struct.field(REPORTED).filter(pl.col(value).is_not_null())).alias(REPORTED)
        if d.propagation == Propagation.REPORTED
        else _reduce_score(value, qual, d.identifier)
        for d in catalog.definitions
    ]).alias(qual)


def combine_sum(
    out: str,
    left_value: str,
    left_qual: str | None,
    right_value: str,
    right_qual: str | None,
    catalog: QualifierCatalog,
) -> pl.Expr:
    fields = []
    for definition in catalog.definitions:
        if definition.propagation == Propagation.REPORTED:
            left = pl.when(pl.col(left_value).is_null()).then(pl.lit(value=True)).otherwise(_field(left_qual, definition))
            right = pl.when(pl.col(right_value).is_null()).then(pl.lit(value=True)).otherwise(_field(right_qual, definition))
            expr = left & right
        else:
            expr = _combine_score(left_value, left_qual, right_value, right_qual, definition.identifier)
        fields.append(expr.alias(definition.identifier))
    return pl.struct(fields).alias(out)


def remove_subset(
    out: str,
    total_value: str,
    total_qual: str | None,
    subset_value: str,
    subset_qual: str | None,
    catalog: QualifierCatalog,
) -> pl.Expr:
    """
    Keep the parent's assessment for a fully assessed sector separation.

    Partial assessments do not determine the assessed share of the remainder.
    A zero/absent subset changes nothing; a zero remainder has no weighted grade.
    """
    fields = []
    unchanged = pl.col(subset_value).fill_null(0.0) == 0
    remaining = pl.col(total_value) - pl.col(subset_value).fill_null(0.0)
    for definition in catalog.definitions:
        parent = _field(total_qual, definition)
        child = _field(subset_qual, definition)
        if definition.propagation == Propagation.REPORTED:
            expr = pl.when(unchanged).then(parent).otherwise(parent & child)
        else:
            score, coverage, _reported = _fields(total_qual, definition.identifier)
            _child_score, child_coverage, _reported = _fields(subset_qual, definition.identifier)
            known = (coverage == 1) & (child_coverage == 1) & (remaining > 0)
            expr = (
                pl
                .when(unchanged)
                .then(parent)
                .when(known)
                .then(covered_score(score, pl.lit(1.0)))
                .otherwise(pl.lit(None, dtype=COVERED_SCORE_DTYPE))
            )
        fields.append(expr.alias(definition.identifier))
    return pl.struct(fields).alias(out)


def combine_product(out: str, left_qual: str | None, right_qual: str | None, catalog: QualifierCatalog) -> pl.Expr:
    """Carry a sole assessment; retain the provisional conservative product policy."""
    fields = []
    for d in catalog.definitions:
        left, right = _field(left_qual, d), _field(right_qual, d)
        if d.propagation == Propagation.REPORTED:
            expr = left & right
        else:
            score = pl.min_horizontal(left.struct.field(SCORE), right.struct.field(SCORE))
            coverage = pl.min_horizontal(left.struct.field(COVERAGE), right.struct.field(COVERAGE))
            expr = (
                pl
                .when(left.struct.field(SCORE).is_null())
                .then(right)
                .when(right.struct.field(SCORE).is_null())
                .then(left)
                .otherwise(covered_score(score, coverage))
            )
        fields.append(expr.alias(d.identifier))
    return pl.struct(fields).alias(out)


def choose(
    out: str,
    take_left: pl.Expr,
    left_qual: str | None,
    right_qual: str | None,
    catalog: QualifierCatalog,
) -> pl.Expr:
    left = pl.col(left_qual) if left_qual is not None else pl.lit(None, dtype=catalog.dtype)
    right = pl.col(right_qual) if right_qual is not None else pl.lit(None, dtype=catalog.dtype)
    return pl.when(take_left).then(left).otherwise(right).alias(out)


def _fill_score(value: str, qual: str, definition: QualifierDefinition, fill: str, dims: list[str]) -> pl.Expr:
    assessment = pl.col(qual).struct.field(definition.identifier)
    score = assessment.struct.field(SCORE)
    coverage = assessment.struct.field(COVERAGE)
    # A real endpoint without an assessment has zero assessed weight.
    endpoint = pl.col(value).is_not_null() & pl.col(qual).is_not_null()
    covered = pl.when(endpoint).then(coverage.fill_null(0.0))
    numerator = pl.when(endpoint).then(coverage.fill_null(0.0) * score.fill_null(0.0))
    if fill in {'interpolate', 'interpolate_backfill', 'all'}:
        covered, numerator = covered.interpolate_by('Year'), numerator.interpolate_by('Year')
    if fill in {'backfill', 'interpolate_backfill', 'both', 'all'}:
        covered, numerator = covered.backward_fill(), numerator.backward_fill()
    if fill in {'extend', 'both', 'all'}:
        covered, numerator = covered.forward_fill(), numerator.forward_fill()
    if fill == 'zero':
        covered, numerator = pl.lit(0.0), pl.lit(0.0)
    if dims:
        covered, numerator = covered.over(dims), numerator.over(dims)
    return covered_score(numerator / covered, covered)


def carry_over(
    before: PathsDataFrame,
    after: PathsDataFrame,
    *,
    start: bool = False,
    catalog: QualifierCatalog | None = None,
    fill: str = 'zero',
) -> PathsDataFrame:
    """
    Give the result of a fill operation the qualifiers of the frame it filled.

    A fill operation adds rows or replaces nulls, and most of them go through a wide pivot that
    drops every column it does not know. Rather than teaching each of them about qualifiers,
    the qualifiers are carried over afterwards by key: a cell that held a value before keeps its
    qualifier, and a cell that did not -- a new row, or a null that is now a number -- is marked
    not reported. The assessment policy is selected by the fill mechanism.

    ``start`` makes a qualifier for a frame that had none, recording only what was reported.
    ``empty_to_zero`` needs it: a zero it wrote is otherwise indistinguishable from a zero
    someone reported, which is the one thing a consumer choosing between sources has to know.
    Other fills only keep a record that already exists, so a frame nobody qualified stays as
    light as it was.
    """
    catalog = catalog or catalog_for_frames(before)
    if not catalog.definitions:
        catalog = BUILTIN_QUALIFIERS
    metrics = [m for m in after.metric_cols if m in before.columns]
    quals = {m: qualifier_column(m) for m in metrics if qualifier_column(m) in before.columns}
    if not quals and start:
        before = before.with_columns([
            make(reported=pl.col(m).is_not_null(), catalog=catalog).alias(qualifier_column(m)) for m in metrics
        ])
        quals = {m: qualifier_column(m) for m in metrics}
    if not quals:
        return after
    keys = [key for key in before.primary_keys if key in after.columns]
    if not keys or set(keys) != set(after.primary_keys):
        return after
    from common import polars as ppl

    meta = after.get_meta()
    carried = pl.DataFrame(after).drop([q for q in quals.values() if q in after.columns])
    source = pl.DataFrame(before).select([
        *[pl.col(key).cast(carried.schema[key]) for key in keys],
        *[
            pl.when(pl.col(m).is_not_null()).then(pl.col(q)).otherwise(pl.lit(None, dtype=catalog.dtype)).alias(q)
            for m, q in quals.items()
        ],
    ])
    joined = carried.join(source, on=keys, how='left', nulls_equal=True)
    joined = joined.sort(keys)
    dims = [k for k in keys if k != 'Year']
    for m, q in quals.items():
        fields = []
        for d in catalog.definitions:
            if d.propagation == Propagation.REPORTED:
                expr = pl.lit(value=False)
            else:
                expr = _fill_score(m, q, d, fill, dims)
            fields.append(expr.alias(d.identifier))
        filled = pl.struct(fields).cast(catalog.dtype)
        joined = joined.with_columns(
            pl.when(pl.col(q).is_null() & pl.col(m).is_not_null()).then(filled).otherwise(pl.col(q)).alias(q)
        )
    return ppl.to_ppdf(joined, meta=meta)


def qualifier_columns(columns: Sequence[str], metric_cols: Sequence[str]) -> dict[str, str]:
    """Map each metric in ``metric_cols`` that has a qualifier in ``columns`` to that qualifier."""
    present = set(columns)
    return {m: qualifier_column(m) for m in metric_cols if qualifier_column(m) in present}
