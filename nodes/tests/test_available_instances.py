import pytest

from paths.tests.graphql import PathsTestClient

from frameworks.tests.factories import FrameworkConfigFactory, FrameworkFactory
from nodes.defs.instance_defs import InstanceFeatures
from nodes.models import YAML_SPEC_VERSION, InstanceConfig, InstanceHostname
from nodes.tests.factories import InstanceConfigFactory

pytestmark = pytest.mark.django_db


def test_available_instances_uses_instance_config_metadata(graphql_client_query_data, monkeypatch):
    hostname = 'paths.example.test'
    instance_config = InstanceConfigFactory.create(
        identifier='quick-instance',
        name='Quick Instance',
        is_protected=True,
        primary_language='fi',
        other_languages=['en', 'sv'],
    )
    assert instance_config.spec is not None
    instance_config.spec.theme_identifier = 'fast-theme'
    instance_config.spec.features = InstanceFeatures(requires_authentication=True)
    instance_config.save(update_fields=['spec'])
    InstanceHostname.objects.create(instance=instance_config, hostname=hostname, base_path='/quick')

    def fail_get_instance(self, *args, **kwargs):
        raise AssertionError('availableInstances must not hydrate the computation instance')

    monkeypatch.setattr(InstanceConfig, 'get_instance', fail_get_instance)

    data = graphql_client_query_data(
        """
        query AvailableInstances($hostname: String!) {
          availableInstances(hostname: $hostname) {
            identifier
            isProtected
            requiresAuthentication
            defaultLanguage
            supportedLanguages
            themeIdentifier
            hostname {
              basePath
            }
          }
        }
        """,
        variables={'hostname': hostname},
    )

    assert data == {
        'availableInstances': [
            {
                'identifier': 'quick-instance',
                'isProtected': True,
                'requiresAuthentication': True,
                'defaultLanguage': 'fi',
                'supportedLanguages': ['fi', 'en', 'sv'],
                'themeIdentifier': 'fast-theme',
                'hostname': {'basePath': '/quick'},
            }
        ]
    }


def test_available_instances_backfills_missing_yaml_spec(graphql_client_query_data, monkeypatch, settings, tmp_path):
    hostname = 'legacy.example.test'
    configs_dir = tmp_path / 'configs'
    configs_dir.mkdir()
    settings.BASE_DIR = tmp_path
    (configs_dir / 'legacy-instance.yaml').write_text(
        """id: legacy-instance
default_language: fi
supported_languages: [en]
name: Legacy Instance
owner: Kausal
target_year: 2035
theme_identifier: fast-theme
features:
  requires_authentication: true
""",
        encoding='utf8',
    )
    instance_config = InstanceConfigFactory.create(
        identifier='legacy-instance',
        name='Legacy Instance',
        config_source='yaml',
        spec=None,
    )
    InstanceHostname.objects.create(instance=instance_config, hostname=hostname, base_path='/legacy')

    def fail_get_instance(self, *args, **kwargs):
        raise AssertionError('availableInstances must not hydrate the computation instance')

    monkeypatch.setattr(InstanceConfig, 'get_instance', fail_get_instance)

    data = graphql_client_query_data(
        """
        query AvailableInstances($hostname: String!) {
          availableInstances(hostname: $hostname) {
            identifier
            requiresAuthentication
            defaultLanguage
            supportedLanguages
            themeIdentifier
          }
        }
        """,
        variables={'hostname': hostname},
    )

    assert data == {
        'availableInstances': [
            {
                'identifier': 'legacy-instance',
                'requiresAuthentication': True,
                'defaultLanguage': 'fi',
                'supportedLanguages': ['fi', 'en'],
                'themeIdentifier': 'fast-theme',
            }
        ]
    }
    instance_config.refresh_from_db()
    assert instance_config.spec is not None
    assert instance_config.yaml_mtime_hash is not None


def test_available_instances_refreshes_stale_yaml_spec(graphql_client_query_data, monkeypatch, settings, tmp_path):
    hostname = 'stale.example.test'
    configs_dir = tmp_path / 'configs'
    configs_dir.mkdir()
    settings.BASE_DIR = tmp_path
    (configs_dir / 'stale-instance.yaml').write_text(
        """id: stale-instance
default_language: fi
supported_languages: [en]
name: Stale Instance
owner: Kausal
target_year: 2035
reference_year: 1990
minimum_historical_year: 1990
maximum_historical_year: 2024
model_end_year: 2050
theme_identifier: current-theme
features:
  requires_authentication: true
""",
        encoding='utf8',
    )
    instance_config = InstanceConfigFactory.create(
        identifier='stale-instance',
        name='Stale Instance',
        config_source='yaml',
        yaml_mtime_hash='stale-hash',
    )
    assert instance_config.spec is not None
    instance_config.spec.years.reference = 2010
    instance_config.spec.years.model_end = 2035
    instance_config.spec.theme_identifier = 'stale-theme'
    instance_config.spec.features = InstanceFeatures(requires_authentication=False)
    instance_config.save(update_fields=['spec'])
    InstanceHostname.objects.create(instance=instance_config, hostname=hostname)

    def fail_get_instance(self, *args, **kwargs):
        raise AssertionError('availableInstances must not hydrate the computation instance')

    monkeypatch.setattr(InstanceConfig, 'get_instance', fail_get_instance)

    data = graphql_client_query_data(
        """
        query AvailableInstances($hostname: String!) {
          availableInstances(hostname: $hostname) {
            identifier
            requiresAuthentication
            themeIdentifier
          }
        }
        """,
        variables={'hostname': hostname},
    )

    assert data == {
        'availableInstances': [
            {
                'identifier': 'stale-instance',
                'requiresAuthentication': True,
                'themeIdentifier': 'current-theme',
            }
        ]
    }
    instance_config.refresh_from_db()
    assert instance_config.config_source == 'yaml'
    assert instance_config.yaml_mtime_hash not in (None, 'stale-hash')
    assert instance_config.spec is not None
    assert instance_config.spec.years.reference == 1990
    assert instance_config.spec.years.model_end == 2050


def test_available_instances_expands_path_routed_framework_from_root_hostname(client, settings):
    settings.HOSTNAME_INSTANCE_DOMAINS = ['localhost']
    root_config = InstanceConfigFactory.create(identifier='cads-landing', name='CADS')
    city_config = InstanceConfigFactory.create(identifier='city', name='City')
    framework = FrameworkFactory.create(
        public_base_fqdn='cads.kausal.tech',
        use_instance_subdomains=False,
        root_instance=root_config,
    )
    FrameworkConfigFactory.create(framework=framework, instance_config=root_config)
    FrameworkConfigFactory.create(framework=framework, instance_config=city_config)
    gql_client = PathsTestClient(client)

    data = gql_client.query_data(
        """
        query AvailableInstances($hostname: String!) {
          availableInstances(hostname: $hostname) {
            identifier
            hostname {
              hostname
              basePath
            }
          }
        }
        """,
        variables={'hostname': 'cads-landing.localhost'},
    )

    assert data == {
        'availableInstances': [
            {
                'identifier': 'cads-landing',
                'hostname': {'hostname': 'cads-landing.localhost', 'basePath': ''},
            },
            {
                'identifier': 'city',
                'hostname': {'hostname': 'cads-landing.localhost', 'basePath': f'/{city_config.uuid}'},
            },
        ]
    }


def test_ensure_spec_backfills_from_yaml(settings, tmp_path):
    configs_dir = tmp_path / 'configs'
    configs_dir.mkdir()
    settings.BASE_DIR = tmp_path
    (configs_dir / 'quick-instance.yaml').write_text(
        """id: quick-instance
default_language: fi
supported_languages: [en, sv]
name: Quick Instance from YAML
owner: Kausal
target_year: 2035
reference_year: 2020
minimum_historical_year: 2018
maximum_historical_year: 2024
model_end_year: 2040
theme_identifier: fast-theme
features:
  requires_authentication: true
action_groups:
- id: energy
  name: Energia
  name_en: Energy
  name_sv: Energi
  color: '#123456'
""",
        encoding='utf8',
    )
    instance_config = InstanceConfigFactory.create(
        identifier='quick-instance',
        name='Quick Instance',
        config_source='yaml',
        spec=None,
    )

    spec = instance_config.ensure_spec()

    instance_config.refresh_from_db()
    assert instance_config.primary_language == 'fi'
    assert instance_config.other_languages == ['en', 'sv']
    assert instance_config.yaml_mtime_hash is not None
    assert instance_config.yaml_spec_version == YAML_SPEC_VERSION
    assert spec.theme_identifier == 'fast-theme'
    assert spec.features.requires_authentication is True
    assert spec.years.target == 2035
    assert spec.years.model_end == 2040
    assert [(group.id, str(group.name), group.color) for group in spec.action_groups] == [('energy', 'Energy', '#123456')]

    yaml_mtime_hash = instance_config.yaml_mtime_hash
    instance_config.spec = spec.model_copy(update={'action_groups': []})
    instance_config.yaml_spec_version = 0
    instance_config.save(update_fields=['spec', 'yaml_spec_version'])
    instance_config.refresh_from_db()

    regenerated_spec = instance_config.ensure_spec()

    instance_config.refresh_from_db()
    assert instance_config.yaml_mtime_hash == yaml_mtime_hash
    assert instance_config.yaml_spec_version == YAML_SPEC_VERSION
    assert [group.id for group in regenerated_spec.action_groups] == ['energy']
