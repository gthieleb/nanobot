"""OpenTelemetry Setup for Nanobot LangGraph."""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from loguru import logger

from nanobot.config.schema import ObservabilitySettings


def setup_opentelemetry(session_id: str, otlp_endpoint: str | None = None):
    """Richtet OpenTelemetry ein (High-Level)."""

    if not otlp_endpoint:
        logger.warning("[OBSERVABILITY] No OTLP endpoint configured - metrics to file only")
        return None

    # Trace-Provider konfigurieren
    resource = Resource.create(schema_url="https://opentelemetry.io/schemas/1.31.1")

    provider = TracerProvider(
        resource=resource,
        service_name="nanobot",
        exporters=[OTLPSpanExporter(endpoint=otlp_endpoint)],
    )

    tracer = provider.get_tracer(__name__)
    logger.info("[OBSERVABILITY] OpenTelemetry tracer initialized for session: {}", session_id)

    return tracer
