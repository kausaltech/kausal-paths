import re
import hashlib
import typing
from typing import Any, List, OrderedDict
import polars as pl

from pydantic import BaseModel, Field, PrivateAttr, ValidationError, root_validator, validator

from common.i18n import I18nString, TranslatedString, get_default_language
from common.types import Identifier

if typing.TYPE_CHECKING:
    import pandas as pd


class DimensionCategoryGroup(BaseModel):
    id: Identifier
    label: I18nString
    id_match: str | None = None

    @validator('id_match')
    def validate_id_match(cls, v):
        if v is None:
            return v
        if not len(v):
            raise ValueError('zero length regex supplied')
        # Try to parse the regex
        re.match(v, '')
        return v

class DimensionCategory(BaseModel):
    id: Identifier
    label: I18nString
    group: str | None = None
    aliases: List[str] = Field(default_factory=list)
    _group: DimensionCategoryGroup | None = PrivateAttr(default=None)

    def all_labels(self) -> set[str]:
        labels = set([str(self.id)])
        if isinstance(self.label, TranslatedString):
            labels.update(self.label.all())
        elif isinstance(self.label, str):
            labels.add(self.label)
        if self.aliases:
            labels.update(self.aliases)
        return labels


class Dimension(BaseModel):
    id: Identifier
    label: I18nString | None = Field(default=None)
    groups: List[DimensionCategoryGroup] = Field(default_factory=list)
    categories: List[DimensionCategory] = Field(default_factory=list)
    is_internal: bool = False
    _hash: bytes | None = PrivateAttr(default=None)
    _cat_map: OrderedDict[str, DimensionCategory] = PrivateAttr(default_factory=dict)
    _group_map: OrderedDict[str, DimensionCategoryGroup] = PrivateAttr(default_factory=dict)

    def __init__(self, **data) -> None:
        super().__init__(**data)
        cat_map = OrderedDict([(str(cat.id), cat) for cat in self.categories])
        self._cat_map = cat_map
        group_map = OrderedDict([(str(g.id), g) for g in self.groups])
        self._group_map = group_map
        for cat in self.categories:
            if cat.group is not None:
                cat._group = group_map[cat.group]
        for g in self.groups:
            if g.id_match is None:
                continue
            cats = [cat for cat in self.categories if re.match(g.id_match, cat.id)]
            for cat in cats:
                if cat._group is not None:
                    cat._group = g

    def get(self, cat_id: str) -> DimensionCategory:
        if cat_id not in self._cat_map:
            raise KeyError("Dimension %s: category %s not found" % (self.id, cat_id))
        return self._cat_map[cat_id]

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

    def series_to_ids(self, s: 'pd.Series') -> 'pd.Series':
        if s.hasnans:
            raise Exception("Series contains NaNs")
        cat_map = self.labels_to_ids()
        s = s.str.strip()
        cs = s.map(cat_map)
        if cs.hasnans:
            missing_cats = s[~s.isin(cat_map)].unique()
            raise Exception("Some dimension categories failed to convert (%s)" % ', '.join(missing_cats))
        return cs

    def series_to_ids_pl(self, s: pl.Series) -> pl.Series:
        name = s.name
        if s.null_count():
            raise Exception("Series contains NaNs")
        s = s.cast(str).str.strip()
        cat_map = self.labels_to_ids()
        label = cat_map.keys()
        id = cat_map.values()
        map_df = pl.DataFrame(dict(label=label, id=id))
        df = pl.DataFrame(dict(cat=s))
        df = df.join(map_df, left_on='cat', right_on='label', how='left')
        if df['id'].null_count():
            missing_cats = df.filter(pl.col('id').is_null())['cat'].unique()
            raise Exception("Some dimension categories failed to convert (%s)" % ', '.join(missing_cats))
        ret = df['id'].cast(pl.Categorical)
        if name:
            ret = ret.alias(name)
        return ret

    def calculate_hash(self) -> bytes:
        if self._hash is not None:
            return self._hash
        h = hashlib.md5()
        h.update(self.json(exclude={
            'label': True, 
            'categories': {'__all__': {'label'}},
            'groups': {'__all__': {'label'}},
        }).encode('utf8'))
        self._hash = h.digest()
        return self._hash

    @classmethod
    def validate_translated_string(cls, field_name: str, obj: dict) -> TranslatedString:
        f = cls.__fields__[field_name]
        field_val = obj.get(field_name)
        langs: dict[str, str] = {}
        if isinstance(field_val, TranslatedString):
            return field_val
        elif isinstance(field_val, str):
            default_language = get_default_language()
            assert default_language is not None
            langs[default_language] = field_val
        else:
            raise TypeError('%s: Expecting str, got %s' % (field_name, type(field_val)))
        # FIXME: how to get default language?
        for key, val in obj.items():
            if '_' not in key or not key.startswith(field_name):
                continue
            parts = key.split('_')
            lang = parts.pop(-1)
            fn = '_'.join(parts)
            if fn != key:
                continue
            if not isinstance(val, str):
                raise TypeError('%s: Expecting str, got %s' % (key, type(val)))
            obj.pop(key)
            langs[lang] = val

        return TranslatedString(default_language=default_language, **langs)

    @validator('categories', each_item=True)
    def validate_category_groups(cls, v, values):
        if v.group is not None:
            for g in values['groups']:
                if g.id == v.group:
                    break
            else:
                raise KeyError('group %s not found' % v.group)
        return v

    @root_validator
    def validate_translated_fields(cls, val: dict):
        for fn, f in cls.__fields__.items():
            t = f.type_
            if (typing.get_origin(t) == typing.Union and TranslatedString in typing.get_args(t)):
                val[fn] = cls.validate_translated_string(fn, val)
        return val
