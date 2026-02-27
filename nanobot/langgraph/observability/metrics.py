"""Observability Metrics for Nanobot LangGraph."""

import time
import json
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field

from nanobot.config.loader import get_data_dir


@dataclass
class AgentMetrics:
    """High-Level Agent Metrics."""

    subagent_spawned_total: int = 0
    subagent_running_count: int = 0
    subagent_completed_total: int = 0
    subagent_failed_total: int = 0

    def spawned(self):
        self.subagent_spawned_total += 1

    def running(self, delta: int = 1):
        self.subagent_running_count += delta

    def completed(self):
        self.subagent_completed_total += 1
        self.subagent_running_count -= 1

    def failed(self):
        self.subagent_failed_total += 1
        self.subagent_running_count -= 1


@dataclass
class LLMCallMetrics:
    """LLM-Aufruf Metriken (High-Level)."""

    call_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    error_count: int = 0

    def record(self, input_tokens: int, output_tokens: int, latency_ms: float):
        self.call_count += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_latency_ms += latency_ms
        logger.info(
            "[METRICS] LLM call | {} tokens in | {}ms", input_tokens + output_tokens, latency_ms
        )


class MetricsCollector:
    """Sammelt Metriken und schreibt sie in Dateien."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.metrics_dir = get_data_dir() / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.metrics_dir / f"{session_id}.json"

        self.agent = AgentMetrics()
        self.llm = LLMCallMetrics()

    def record_subagent_spawn(self, task_id: str, task: str, context_size: int):
        """Subagent spawn Event."""
        self.agent.spawned()
        logger.info(
            "[METRICS] Subagent spawned: {} | {} | context: {} bytes", task_id, task, context_size
        )
        self._append_event(
            {
                "event": "subagent_spawn",
                "task_id": task_id,
                "task": task,
                "context_size": context_size,
                "timestamp": time.time(),
            }
        )

    def record_llm_call(
        self, agent_type: str, model: str, input_tokens: int, output_tokens: int, latency_ms: float
    ):
        """LLM-Aufruf Event (High-Level)."""
        self.llm.record(input_tokens, output_tokens, latency_ms)
        logger.info(
            "[METRICS] LLM call: {} | {} | {}ms | {} tokens",
            agent_type,
            model,
            latency_ms,
            input_tokens + output_tokens,
        )
        self._append_event(
            {
                "event": "llm_call",
                "agent_type": agent_type,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "timestamp": time.time(),
            }
        )

    def get_summary(self) -> dict:
        """Gibt Summary der Metriken zurück."""
        return {
            "subagent_spawned_total": self.agent.subagent_spawned_total,
            "subagent_running_count": self.agent.subagent_running_count,
            "subagent_completed_total": self.agent.subagent_completed_total,
            "subagent_failed_total": self.agent.subagent_failed_total,
            "llm_calls": self.llm.call_count,
            "total_input_tokens": self.llm.total_input_tokens,
            "total_output_tokens": self.llm.total_output_tokens,
            "total_tokens": self.llm.total_input_tokens + self.llm.total_output_tokens,
            "avg_latency_ms": self.llm.total_latency_ms / max(1, self.llm.call_count),
            "error_count": self.llm.error_count,
        }

    def _append_event(self, event: dict):
        """Hängt Event an Datei an."""
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(event) + "\n")
