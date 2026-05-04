from decimal import Decimal

from agentkit.metrics import InMemoryMetricsSink, MetricEvent, MetricsSink, NullMetricsSink


def test_null_sink_accepts_all_metrics_silently():
    sink: MetricsSink = NullMetricsSink()
    sink.record(MetricEvent(name="t", value=1.0, tags={"k": "v"}))


def test_in_memory_sink_collects_metrics():
    sink = InMemoryMetricsSink()
    sink.record(MetricEvent(name="turn.duration_ms", value=123.0, tags={"phase": "streaming"}))
    sink.record(MetricEvent(name="turn.cost_usd", value=Decimal("0.0005"), tags={}))
    assert len(sink.records) == 2
    assert sink.records[0].name == "turn.duration_ms"
