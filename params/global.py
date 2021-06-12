from .param import StringParameter
from common.i18n import gettext_lazy as _


class MunicipalityName(StringParameter):
    name = _('Municipality name')
    id = 'municipality_name'
