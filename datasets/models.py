from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any, ClassVar

from django.contrib.postgres.fields import ArrayField
from django.db import models, transaction
from django.utils.translation import gettext_lazy as _
from modelcluster.fields import ParentalKey
from modelcluster.models import ClusterableModel
from modeltrans.fields import TranslationField
from modeltrans.manager import MultilingualQuerySet
from wagtail.admin.panels import FieldPanel, InlinePanel

import polars as pl

from kausal_common.models.types import FK, M2M, MLModelManager

from paths.utils import IdentifierField, OrderedModel, UnitField, UserModifiableModel, UUIDIdentifierField

from common import polars as ppl
from common.i18n import get_modeltrans_attrs_from_str, get_translated_string_from_modeltrans
from nodes.constants import YEAR_COLUMN
from nodes.datasets import JSONDataset
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wagtail.admin.panels.base import Panel

    import pandas  # noqa: ICN001

    from kausal_common.models.types import RevMany

    from nodes.dimensions import Dimension as NodeDimension
    from users.models import User
