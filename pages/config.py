from __future__ import annotations

from typing import Any

from pydantic import Field

from kausal_common.i18n.pydantic import I18nBaseModel, I18nStringInstance


class Page(I18nBaseModel):
    id: str
    name: I18nStringInstance
    path: str = Field(min_length=1, pattern=r'^/[a-z_0-9-/]*$')
    show_in_menus: bool = False
    show_in_footer: bool = False


class OutcomePage(Page):
    outcome_node: str | None = None
    lead_title: I18nStringInstance | None = None
    lead_paragraph: I18nStringInstance | None = None


def pages_from_config(conf: list[dict[str, Any]]) -> list[OutcomePage]:
    return [OutcomePage.from_yaml_config(pc) for pc in conf]
