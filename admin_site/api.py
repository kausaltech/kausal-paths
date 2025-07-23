from __future__ import annotations

from urllib.parse import urlparse

from django.urls import resolve
from django.utils.translation import gettext as _
from rest_framework.decorators import api_view, authentication_classes, permission_classes, schema, throttle_classes
from rest_framework.exceptions import ValidationError
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
    user = User.objects.filter(email__iexact=email).first()
    if user is None:
        msg = _("No user found with this email address. Ask your administrator to create an account for you.")
        raise ValidationError(dict(detail=msg, code="no_user"))

    next_url_input = d.get('next')
    resolved = None
    if next_url_input:
        next_url = urlparse(next_url_input)
        resolved = resolve(next_url.path)

    destination_is_public_site = resolved and (
        resolved.url_name == 'authorize' and 'oauth2_provider' in resolved.app_names
    )

    if not destination_is_public_site and not user.can_access_admin():
        msg = _("This user does not have access to admin.")
        raise ValidationError(dict(detail=msg, code="no_admin_access"))

    if user.has_usable_password():
        method = 'password'
    else:
        method = 'azure_ad'
    return Response({"method": method})
