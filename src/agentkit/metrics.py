"""MetricsSink protocol + in-memory and null implementations.

Consumers plug in OTLP / Prometheus / StatsD adapters. Library calls
``sink.record(MetricEvent(...))`` from key points in the loop.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class MetricEvent:
    name: str
    value: float | Decimal
    tags: dict[str, str] = field(default_factory=dict)  # type: ignore[reportUnknownVariableType]


@runtime_checkable
class MetricsSink(Protocol):
    def record(self, event: MetricEvent) -> None: ...


class NullMetricsSink(MetricsSink):
    def record(self, event: MetricEvent) -> None:
        pass


class InMemoryMetricsSink(MetricsSink):
    def __init__(self) -> None:
        self.records: list[MetricEvent] = []

    def record(self, event: MetricEvent) -> None:
        self.records.append(event)
