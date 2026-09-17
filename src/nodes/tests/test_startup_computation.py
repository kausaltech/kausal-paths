import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.django_db

SCRIPT = Path(__file__).parents[3] / 'docker' / 'pre-entry' / '10-compute-instances.sh'


@pytest.mark.parametrize('behavior', ['exit 0', 'exit 1', 'sleep 10'])
def test_warmup_script_allows_startup_after_success_failure_or_timeout(tmp_path: Path, behavior: str) -> None:
    binary = tmp_path / 'python'
    binary.write_text(f'#!/bin/bash\necho "$*"\n{behavior}\n')
    binary.chmod(0o755)
    result = subprocess.run(  # noqa: S603 - fixed startup script with a test-owned Python substitute
        ['/bin/bash', str(SCRIPT)],
        env={**os.environ, 'PATH': f'{tmp_path}:{os.environ["PATH"]}', 'COMPUTE_INSTANCES_TIMEOUT': '0.1'},
        capture_output=True,
        text=True,
        timeout=3,
        check=True,
    )
    assert 'manage.py compute_instances --in-customer-use' in result.stdout
    if behavior == 'exit 0':
        assert 'warm-up completed' in result.stdout
    else:
        assert 'continuing server startup' in result.stderr
