"""Sampling and Async-Metrik-Sammlung."""

import time
from typing import List
from loguru import logger

from opentelemetry import metrics as otel_metrics
from opentelemetry import trace


class Sampler:
    """
    Sampling (10%) der Traces für asynchrone Metriken-Sammlung.
    """

    def __init__(self):
        self.latency_samples: List[float] = []
        self.async_sample_count = 0
        self.async_sample_ignored = 0

    def should_sample(self, trace_id: str) -> bool:
        """
        Bestimmt, ob ein Trace gesampelt werden soll (10% Rate).
        """
        if not trace_id:
            return False

        # Deterministisch basier auf trace_id
        sample_value = hash(trace_id) % 10
        return sample_value < 1

    def record_latency(self, latency_ms: float):
        """Recordiert Latency für Sampling."""
        self.latency_samples.append(latency_ms)

        if len(self.latency_samples) > 100:
            avg = sum(self.latency_samples) / len(self.latency_samples)
            logger.debug("[SAMPLER] Avg latency over last 100 samples: {}ms", avg)

    def record_async_sample(self):
        """Zählt asynchrone Samples."""
        self.async_sample_count += 1

    def record_async_ignored(self):
        """Zählt ignorierte Samples."""
        self.async_sample_ignored += 1

    def get_statistics(self) -> dict:
        """Gibt Sampling-Statistiken zurück."""
        if not self.latency_samples:
            return {}

        return {
            "sample_count": len(self.latency_samples),
            "avg_latency_ms": sum(self.latency_samples) / len(self.latency_samples),
            "async_sample_count": self.async_sample_count,
            "async_sample_ignored": self.async_sample_ignored,
        }


# Globale OTEL-Metriken
llm_call_histogram = otel_metrics.Histogram(
    name="nanobot.llm.call_duration", description="LLM Aufruf-Dauer", unit="ms"
)

active_subagents_gauge = otel_metrics.Gauge(
    name="nanobot.subagent.active", description="Aktive Subagents"
)

llm_total_tokens_counter = otel_metrics.Counter(
    name="nanobot.llm.tokens.total", description="Gesamte LLM-Tokens"
)
