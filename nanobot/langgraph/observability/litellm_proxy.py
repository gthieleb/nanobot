"""Litellm Proxy for Self-Hosting."""

import httpx
from loguru import logger
from typing import Optional, Dict, Any

from nanobot.config.schema import LitellmProxySettings


class LitellmProxyProvider:
    """
    Litellm Proxy Provider for OpenAI-compatible APIs.
    Ermöglicht Self-Hosting und Token-Usage Tracking.
    """

    def __init__(self, config: LitellmProxySettings):
        self.config = config
        if config.enabled:
            self.client = httpx.Client(base_url=config.proxy_url, timeout=30.0)
            logger.info("[PROXY] Litellm proxy initialized: {}", config.proxy_url)
        else:
            self.client = None
            logger.warning("[PROXY] Litellm proxy not enabled - direct calls")

    async def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: list[dict] | None = None,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Führt Chat über Proxy oder direkt durch.
        """
        if self.config.enabled and self.client:
            return await self._proxy_chat(
                messages, model, temperature, max_tokens, tools, stream, **kwargs
            )
        else:
            return await self._fallback_chat(
                messages, model, temperature, max_tokens, tools, stream, **kwargs
            )

    async def _proxy_chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Chat über Litellm Proxy."""
        try:
            # Token-Usage Tracking
            response = await self.client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "tools": tools,
                    "stream": stream,
                },
                timeout=60.0,
            )

            # Metriken aufzeichnen
            input_tokens = response.get("usage", {}).get("prompt_tokens", 0)
            output_tokens = response.get("usage", {}).get("completion_tokens", 0)

            latency = response.headers.get("x-response-time", 0)

            logger.info(
                "[PROXY] Proxy chat completed | in: {} | out: {} | {}ms",
                input_tokens,
                output_tokens,
                latency,
            )

            return {
                "choices": response.get("choices", []),
                "content": response.get("content"),
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            }

        except Exception as e:
            logger.error("[PROXY] Proxy chat failed: {}", str(e))
            return {
                "content": f"Error: {str(e)}",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

    async def _fallback_chat(
        self, messages, model, temperature, max_tokens, tools, stream, **kwargs
    ):
        """Fallback: Direkte Aufrufe ohne Proxy."""
        logger.warning("[PROXY] Fallback to direct provider calls")
        raise NotImplementedError("Direct fallback not implemented yet")
