from __future__ import annotations

import re
import typing
from collections import OrderedDict
from typing import Any, Self, overload

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

import polars as pl
import xxhash

from common.i18n import I18nBaseModel, I18nStringInstance, TranslatedString
from common.types import Identifier  # noqa: TC001

if typing.TYPE_CHECKING:
    import pandas as pd


class DimensionCategoryGroup(BaseModel):
    id: Identifier
    label: I18nStringInstance
    color: str | None = None
    order: int | None = None
    id_match: str | None = None

    @field_validator('id_match')
    @classmethod
    def validate_id_match(cls, v) -> str | None:
        if v is None:
            return v
        if not len(v):
            raise ValueError('zero length regex supplied')
        # Try to parse the regex
        re.match(v, '')
        return v

class DimensionCategory(I18nBaseModel):
    id: Identifier
    label: I18nStringInstance
    color: str | None = None
    group: str | None = None
    order: int | None = None
    aliases: list[str] = Field(default_factory=list)
    _group: DimensionCategoryGroup | None = PrivateAttr(default=None)

    def all_labels(self) -> set[str]:
        labels = {str(self.id)}
        if isinstance(self.label, TranslatedString):
            labels.update(self.label.all())
        elif isinstance(self.label, str):
            labels.add(self.label)
        if self.aliases:
            labels.update(self.aliases)
        return labels


class Dimension(I18nBaseModel):
    id: Identifier
    label: I18nStringInstance
    help_text: I18nStringInstance | None = None
    groups: list[DimensionCategoryGroup] = Field(default_factory=list)
    categories: list[DimensionCategory] = Field(default_factory=list)
    is_internal: bool = False
    mtime_hash: str | None = None

    _hash: bytes | None = PrivateAttr(default=None)
    _cat_map: OrderedDict[str, DimensionCategory] = PrivateAttr()
    _group_map: OrderedDict[str, DimensionCategoryGroup] = PrivateAttr()
    _pl_dt: pl.Enum = PrivateAttr()

    def model_post_init(self, validation_context: Any) -> None:
        super().model_post_init(validation_context)

        cat_map = OrderedDict([(str(cat.id), cat) for cat in self.categories])
        self._cat_map = cat_map
        group_map = OrderedDict([(str(g.id), g) for g in self.groups])
        self._group_map = group_map
        self._pl_dt = pl.Enum(str(cat.id) for cat in self.categories)
        for cat in self.categories:
            if cat.group is not None:
                cat._group = group_map[cat.group]
        for g in self.groups:
            if g.id_match is None:
                continue
            cats = [cat for cat in self.categories if re.match(g.id_match, cat.id)]
            if not cats:
                raise Exception("No categories match the regex '%s'" % g.id_match)
            for cat in cats:
                if cat._group is None:
                    cat._group = g
                    cat.group = g.id

    @property
    def cat_map(self):
        return self._cat_map.copy()

    def get(self, cat_id: str) -> DimensionCategory:
        if cat_id not in self._cat_map:
            raise KeyError("Dimension %s: category %s not found" % (self.id, cat_id))
        return self._cat_map[cat_id]

    def get_cats_for_group(self, group_id: str) -> list[DimensionCategory]:
        if group_id not in self._group_map:
            raise KeyError("Dimension %s: group %s not found" % (self.id, group_id))
        grp = self._group_map[group_id]
        return [cat for cat in self.categories if cat._group == grp]

    def get_cat_ids(self) -> set[str]:
        return set(self._cat_map.keys())

    def get_cat_ids_ordered(self) -> list[str]:
        return list(self._cat_map.keys())

    def labels_to_ids(self) -> dict[str, Identifier]:
        all_labels = {}
        for cat in self.categories:
            for label in cat.all_labels():
                if label in all_labels:
                    raise Exception("Dimension %s: duplicate label %s" % (self.id, label))
                all_labels[label] = cat.id
        return all_labels

    def series_to_ids(self, s: pd.Series) -> pd.Series:
        if s.hasnans:
            raise Exception("Series contains NaNs")
        cat_map = self.labels_to_ids()
        s = s.str.strip()
        cs = s.map(cat_map)
        if cs.hasnans:
            missing_cats = s[~s.isin(cat_map)].unique()
            raise Exception("Some dimension categories failed to convert (%s)" % ', '.join(missing_cats))
        return cs

    @overload
    def ids_to_groups(self, expr: pl.Series) -> pl.Series: ...

    @overload
    def ids_to_groups(self, expr: pl.Expr) -> pl.Expr: ...

    def ids_to_groups(self, expr: pl.Expr | pl.Series) -> pl.Expr | pl.Series:
        id_map = {}
        for cat in self.categories:
            if not cat._group:
                raise Exception("Category %s does not have a group" % cat.id)
            id_map[cat.id] = cat._group.id
        return expr.cast(pl.Utf8).replace(id_map)

    def series_to_ids_pl(self, s: pl.Series, allow_null=False) -> pl.Series:
        name = s.name
        if not allow_null and s.null_count():
            raise Exception(f"Series {self.id} contains NaNs: {s}")
        s = s.cast(str).str.strip_chars()
        cat_map = self.labels_to_ids()
        labels = list(cat_map.keys())
        ids = list(cat_map.values())
        map_df = pl.DataFrame(dict(label=labels, id=ids))
        df = pl.DataFrame(dict(cat=s))
        df = df.join(map_df, left_on='cat', right_on='label', how='left')
        if df['id'].null_count() > s.null_count():
            missing_cats = df.filter(pl.col('id').is_null())['cat'].unique()
            # FIXME: `missing_cats` should not contain those items that were already in `s`.
            raise Exception(f"Some dimension {self.id} categories failed to convert: `{'`, `'.join(missing_cats)}`")
        ret = df['id'].cast(pl.Categorical)
        if name:
            ret = ret.alias(name)
        return ret

    def calculate_hash(self) -> bytes:
        if self._hash is not None:
            return self._hash
        if self.mtime_hash is not None:
            self._hash = self.mtime_hash.encode('ascii')
            return self._hash
        data = self.model_dump_json(exclude={
            'label': True,
            'categories': {'__all__': {'label'}},
            'groups': {'__all__': {'label'}},
        })
        h = xxhash.xxh64()
        h.update(data.encode('utf8'))
        self._hash = h.digest()
        return self._hash

    @model_validator(mode='after')
    def validate_category_groups(self) -> Self:
        for cat in self.categories:
            if cat.group is None:
                continue
            for g in self.groups:
                if g.id == cat.group:
                    break
            else:
                raise KeyError('group %s not found' % cat.group)
        return self
