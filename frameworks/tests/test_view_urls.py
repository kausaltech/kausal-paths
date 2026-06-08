from __future__ import annotations

from django.test import RequestFactory, override_settings

import pytest

from paths.const import WILDCARD_DOMAINS_HEADER

from frameworks.tests.factories import FrameworkConfigFactory, FrameworkFactory
from nodes.tests.factories import InstanceConfigFactory

pytestmark = pytest.mark.django_db


def _instance_config(identifier: str):
    return InstanceConfigFactory.create(identifier=identifier, name=identifier.replace('-', ' ').title())


def test_framework_config_view_url_uses_subdomain_by_default() -> None:
    fwc = FrameworkConfigFactory.create(
        framework=FrameworkFactory.create(public_base_fqdn='framework.example.com'),
        instance_config=_instance_config('city'),
    )

    assert fwc.get_view_url() == 'https://city.framework.example.com'


def test_framework_config_view_url_uses_uuid_path_when_subdomains_are_disabled() -> None:
    fwc = FrameworkConfigFactory.create(
        framework=FrameworkFactory.create(
            public_base_fqdn='framework.example.com',
            use_instance_subdomains=False,
        ),
        instance_config=_instance_config('city'),
    )

    assert fwc.get_view_url() == f'https://framework.example.com/{fwc.instance_config.uuid}'


@override_settings(HOSTNAME_INSTANCE_DOMAINS=['localhost'])
def test_framework_config_view_url_preserves_local_subdomain_routing() -> None:
    fwc = FrameworkConfigFactory.create(
        framework=FrameworkFactory.create(public_base_fqdn='framework.example.com'),
        instance_config=_instance_config('city'),
    )
    request = RequestFactory().post('/v1/graphql/', headers={'origin': 'http://landing.localhost:3000'})

    assert fwc.get_view_url(request=request) == 'http://city.localhost:3000'


@override_settings(HOSTNAME_INSTANCE_DOMAINS=['localhost'])
def test_framework_config_view_url_uses_landing_host_for_local_path_routing() -> None:
    landing_ic = _instance_config('cads-landing')
    city_ic = _instance_config('city')
    framework = FrameworkFactory.create(
        public_base_fqdn='cads.kausal.tech',
        use_instance_subdomains=False,
        root_instance=landing_ic,
    )
    fwc = FrameworkConfigFactory.create(framework=framework, instance_config=city_ic)
    request = RequestFactory().post('/v1/graphql/', HTTP_ORIGIN='http://cads-landing.localhost:3000')

    assert fwc.get_view_url(request=request) == f'http://cads-landing.localhost:3000/{city_ic.uuid}'


def test_framework_config_view_url_uses_landing_host_from_request_wildcards() -> None:
    landing_ic = _instance_config('cads-landing')
    city_ic = _instance_config('city')
    framework = FrameworkFactory.create(
        public_base_fqdn='cads.kausal.tech',
        use_instance_subdomains=False,
        root_instance=landing_ic,
    )
    fwc = FrameworkConfigFactory.create(framework=framework, instance_config=city_ic)
    request = RequestFactory().post(
        '/v1/graphql/',
        headers={
            'origin': 'https://cads-landing.paths-ui.dev',
            WILDCARD_DOMAINS_HEADER: 'paths-ui.dev',
        },
    )

    assert fwc.get_view_url(request=request) == f'https://cads-landing.paths-ui.dev/{city_ic.uuid}'
