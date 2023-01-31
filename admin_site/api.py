from django.utils.translation import gettext as _

from rest_framework.decorators import (
    api_view, throttle_classes, schema, authentication_classes, permission_classes
)
from rest_framework.exceptions import ValidationError, NotFound
from rest_framework.response import Response
from rest_framework.throttling import UserRateThrottle

from users.models import User


class LoginMethodThrottle(UserRateThrottle):
    rate = '5/m'


@api_view(['POST'])
@authentication_classes([])
@permission_classes([])
@schema(None)
@throttle_classes([LoginMethodThrottle])
def check_login_method(request):
    d = request.data
    if not d or not isinstance(d, dict):
        msg = _("Invalid email address")
        raise ValidationError(dict(detail=msg, code="invalid_email"))

    email = d.get('email', '').strip().lower()
    if not email:
        msg = _("Invalid email address")
        raise ValidationError(dict(detail=msg, code="invalid_email"))
    user = User.objects.filter(email=email).first()
    if user is None:
        msg = _("No user found with this email address. Ask your administrator to create an account for you.")
        raise ValidationError(dict(detail=msg, code="no_user"))

    if not user.can_access_admin():
        msg = _("This user does not have access to admin.")
        raise ValidationError(dict(detail=msg, code="no_admin_access"))

    if user.has_usable_password():
        method = 'password'
    else:
        method = 'azure_ad'
    return Response({"method": method})
