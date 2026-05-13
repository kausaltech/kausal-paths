"""Email notifications related to instance lifecycle."""

from pathlib import Path
from typing import TYPE_CHECKING

from django.conf import settings
from django.core.mail import EmailMultiAlternatives
from django.utils.html import strip_tags

from kausal_common.notifications.mjml import render_mjml_from_template

if TYPE_CHECKING:
    from nodes.models import InstanceInvitation


_EMAIL_TEMPLATE_DIR = str(Path(__file__).parent / 'email_templates')


def _resolve_accept_url(invitation: InstanceInvitation) -> str:
    fw_config = getattr(invitation.instance_config, 'framework_config', None)
    if fw_config is None:
        identifier = invitation.instance_config.identifier
        msg = f'Instance "{identifier}" has no framework config; cannot resolve invitation acceptance URL.'
        raise RuntimeError(msg)
    url_template = fw_config.framework.accept_invitation_url
    if not url_template:
        raise RuntimeError(f'Framework "{fw_config.framework.identifier}" has no accept_invitation_url configured.')
    return url_template.replace('{code}', invitation.token)


def send_instance_invitation(invitation: InstanceInvitation) -> None:
    instance = invitation.instance_config
    accept_url = _resolve_accept_url(invitation)
    inviter = invitation.created_by
    inviter_name = ''
    if inviter is not None:
        inviter_name = inviter.get_full_name() or inviter.email

    subject = f'You have been invited to {instance.get_name()}'
    context = {
        'subject': subject,
        'instance_name': instance.get_name(),
        'inviter_name': inviter_name,
        'accept_url': accept_url,
        'expires_at': invitation.expires_at,
    }
    html_body = render_mjml_from_template(
        'instance_invitation',
        context,
        template_dirs=[_EMAIL_TEMPLATE_DIR],
    )
    text_body = strip_tags(html_body)
    msg = EmailMultiAlternatives(
        subject=subject,
        body=text_body,
        from_email=getattr(settings, 'DEFAULT_FROM_EMAIL', None),
        to=[invitation.email],
    )
    msg.attach_alternative(html_body, 'text/html')
    msg.send()
