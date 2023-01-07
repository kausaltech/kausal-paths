import hashlib
import typing
from typing import List

from pydantic import BaseModel, Field, PrivateAttr

from common.i18n import I18nString, TranslatedString
from common.types import Identifier

if typing.TYPE_CHECKING:
    import pandas as pd


class DimensionCategory(BaseModel):
    id: Identifier
    label: I18nString | None
    aliases: List[str] = Field(default_factory=list)

    def all_labels(self) -> set[str]:
        labels = set()
        if isinstance(self.label, TranslatedString):
            labels.update(self.label.all())
        elif isinstance(self.label, str):
            labels.add(self.label)
        if self.aliases:
            labels.update(self.aliases)
        return labels


class Dimension(BaseModel):
    id: Identifier
    categories: List[DimensionCategory] = Field(default_factory=list)
    _hash: bytes | None = PrivateAttr(default=None)

    def __init__(self, **data) -> None:
        super().__init__(**data)

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
        cs = s.map(cat_map)
        if cs.hasnans:
            missing_cats = s[~s.isin(cat_map)].unique()
            print(missing_cats)
            raise Exception("Some dimension categories failed to convert (%s)" % ', '.join(missing_cats))
        return cs

    def calculate_hash(self) -> bytes:
        if self._hash is not None:
            return self._hash
        h = hashlib.md5()
        h.update(self.json(exclude={'categories': {'__all__': {'label'}}}).encode('utf8'))
        self._hash = h.digest()
        return self._hash

