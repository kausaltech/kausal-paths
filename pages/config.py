from pydantic import BaseModel, Field

from common.i18n import I18nBaseModel, I18nStringInstance


class Page(I18nBaseModel):
    id: str
    name: I18nStringInstance
    path: str = Field(min_length=1, pattern=r'^/[a-z_0-9-/]*$')
    show_in_menus: bool = False
    show_in_footer: bool = False


class OutcomePage(Page):
    outcome_node: str
    lead_title: I18nStringInstance | None = None
    lead_paragraph: I18nStringInstance | None = None


def pages_from_config(conf: list[dict]):
    pages = []
    for pc in conf:
        page = OutcomePage.model_validate(pc)
        pages.append(page)
    return pages
