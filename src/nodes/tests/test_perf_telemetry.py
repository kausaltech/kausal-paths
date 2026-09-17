from contextlib import nullcontext
from typing import TYPE_CHECKING

import pytest
import sentry_sdk
from sentry_sdk.transport import Transport

if TYPE_CHECKING:
    from sentry_sdk.envelope import Envelope

    from nodes.instance import Instance

pytestmark = pytest.mark.django_db


@pytest.mark.parametrize('sampled', [True, False])
@pytest.mark.parametrize('failed', [True, False])
def test_model_performance_summary(instance: Instance, sampled: bool, failed: bool) -> None:
    envelopes: list[Envelope] = []

    class CaptureTransport(Transport):
        def capture_envelope(self, envelope: Envelope) -> None:
            envelopes.append(envelope)

    context = instance.context
    context.perf_context.enabled = False
    with sentry_sdk.init(
        dsn='https://public@example.com/1',
        transport=CaptureTransport(),
        default_integrations=False,
        traces_sample_rate=1.0 if sampled else 0.0,
        _experiments={'max_spans': 3},
    ):
        with (
            sentry_sdk.start_transaction(name='perf-test'),
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
    transactions = [item.payload.json for envelope in envelopes for item in envelope.items if item.type == 'transaction']
    if not sampled:
        assert transactions == []
        return
    assert len(transactions) == 1
    transaction = transactions[0]
    assert transaction is not None
    spans = transaction['spans']
    assert len(spans) < 30
    span = next(span for span in spans if span['op'] == 'model.calculate')
    data = span.get('data', {})
    if failed:
        assert 'model.perf' not in data
        return
    breakdown = data['model.perf']
    assert breakdown['operation_group_count'] == 1
    assert breakdown['top_operations'][0]['count'] == 30
    assert breakdown['top_operations'][0]['operation'] == 'get'
    assert breakdown['omitted_own_total_ms'] == 0
