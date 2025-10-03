from __future__ import annotations

from urllib.parse import urlparse

from django.conf import settings
from django.urls import resolve
from django.utils.translation import gettext as _
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, authentication_classes, permission_classes, schema, throttle_classes
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from rest_framework.throttling import UserRateThrottle

import requests

from users.models import User


class LoginMethodThrottle(UserRateThrottle):
    rate = '5/m'


def check_user_in_other_clusters(email, request):
    """Check if user exists in other regional clusters."""
    current_host = request.get_host()
    cluster_endpoints = getattr(settings, 'PATHS_BACKEND_REGION_URLS', [])

    # Check that the current host is not a regional endpoint
    if any(current_host == urlparse(endpoint).hostname for endpoint in cluster_endpoints):
        return None

    for endpoint in cluster_endpoints:
        try:
            response = requests.post(
                f"{endpoint}/admin/login/check/",
                json={'email': email},
                timeout=5,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                result = response.json()
                result['cluster_url'] = endpoint
                return result

        except requests.exceptions.RequestException:
            continue

    return None

@csrf_exempt
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
        cluster_result = check_user_in_other_clusters(email, request)
        if cluster_result:
            return Response({
                'method': cluster_result.get('method'),
                'cluster_redirect': True,
                'cluster_url': cluster_result.get('cluster_url')
            })

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
