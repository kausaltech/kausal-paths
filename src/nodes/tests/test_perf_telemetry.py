from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import orjson
import pytest
import sentry_sdk
from sentry_sdk.transport import Transport

if TYPE_CHECKING:
    from collections.abc import Generator

    from sentry_sdk.envelope import Envelope

    from nodes.instance import Instance

pytestmark = pytest.mark.django_db


class NullTransport(Transport):
    def capture_envelope(self, envelope: Envelope) -> None:
        pass


@pytest.fixture
def capture_spans(request: pytest.FixtureRequest) -> Generator[list[dict[str, Any]]]:
    """Initialise Sentry with the given trace sample rate and collect every span it would send."""
    sample_rate: float = request.param
    spans: list[dict[str, Any]] = []

    def before_send_span(span: Any, hint: Any) -> Any:
        spans.append(span)
        return span

    global_scope = sentry_sdk.get_global_scope()
    previous_client = global_scope.client
    sentry_sdk.init(
        dsn='https://public@example.com/1',
        transport=NullTransport(),
        default_integrations=False,
        traces_sample_rate=sample_rate,
        before_send_span=before_send_span,
    )
    try:
        yield spans
    finally:
        sentry_sdk.get_client().close()
        global_scope.set_client(previous_client)


@pytest.mark.parametrize(('capture_spans', 'sampled'), [(1.0, True), (0.0, False)], indirect=['capture_spans'])
@pytest.mark.parametrize('failed', [True, False])
def test_model_performance_summary(instance: Instance, capture_spans: list[dict[str, Any]], sampled: bool, failed: bool) -> None:
    context = instance.context
    context.perf_context.enabled = False
    with (
        sentry_sdk.start_span(name='perf-test', parent_span=None),
        pytest.raises(ValueError, match='test failure') if failed else nullcontext(),
        context.run(),
    ):
        for i in range(30):
            with context.start_perf_span('load', kind='dataset', id=str(i), op='get'):
                pass
        if failed:
            raise ValueError('test failure')
    sentry_sdk.flush()

    assert context.perf_run is None
    assert context.perf_context.aggregate_only is False
    if not sampled:
        assert capture_spans == []
        return
    span = next(span for span in capture_spans if span['attributes'].get('sentry.op') == 'model.calculate')
    attributes = span['attributes']
    if failed:
        assert 'model.perf' not in attributes
        return
    breakdown = orjson.loads(attributes['model.perf'])
    assert breakdown['operation_group_count'] == 1
    assert breakdown['top_operations'][0]['count'] == 30
    assert breakdown['top_operations'][0]['operation'] == 'get'
    assert breakdown['omitted_own_total_ms'] == 0
