"""Main entry point for LangGraph-based nanobot agent."""

import asyncio
from pathlib import Path

from nanobot.config.loader import load_config
from nanobot.config.loader import get_config_path, get_data_dir
from nanobot.langgraph.config.settings import LangGraphSettings
from nanobot.langgraph.graph.main_graph import create_main_graph
from nanobot.langgraph.graph.state import AgentState
from nanobot.langgraph.bus.state_adapter import StateMessageBusAdapter
from nanobot.bus.queue import MessageBus
from loguru import logger


def get_langgraph_settings() -> LangGraphSettings:
    """Load LangGraph settings from config file."""
    config_path = get_config_path()

    if config_path.exists():
        import json

        try:
            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)

            langgraph_data = data.get("langgraph", {})
            return LangGraphSettings(**langgraph_data)
        except (json.JSONDecodeError, ValueError):
            pass

    return LangGraphSettings()


async def main(mode: str = "gateway", message: str | None = None):
    """
    Main entry point for LangGraph-based agent.

    Args:
        mode: "gateway" or "agent"
        message: Optional single message to process
    """
    settings = get_langgraph_settings()

    if not settings.enabled:
        logger.warning("LangGraph is not enabled. Set langgraph.enabled=true in config.json")
        return

    logger.info("Starting LangGraph-based nanobot agent in {} mode", mode)

    config = load_config()

    # Graph erstellen
    try:
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError as e:
        logger.error(
            "Failed to import LangGraph: {}. Install with: pip install langgraph langgraph-checkpoint",
            e,
        )
        return

    graph = create_main_graph(config)

    # Checkpointer für Persistenz
    checkpointer = MemorySaver()

    # Graph kompilieren
    app = graph.compile(checkpointer=checkpointer)

    # Message Bus erstellen
    bus = MessageBus()

    # Message Bus Adapter für Koexistenz mit Nanobot
    bus_adapter = StateMessageBusAdapter(graph_app=app, nanobot_bus=bus, config=config)

    if mode == "gateway":
        await _run_gateway(bus_adapter)
    elif mode == "agent" and message:
        await _run_single_message(bus_adapter, message)
    elif mode == "agent":
        await _run_interactive(bus_adapter)
    else:
        logger.error("Unknown mode: {}", mode)


async def _run_gateway(bus_adapter: StateMessageBusAdapter):
    """Run in gateway mode, continuously processing messages from the bus."""
    logger.info("LangGraph gateway mode: waiting for messages...")

    try:
        while True:
            msg = await bus_adapter.consume_from_bus()
            if msg:
                await bus_adapter.process_message(msg)
    except KeyboardInterrupt:
        logger.info("Gateway mode stopped by user")
    except Exception as e:
        logger.error("Error in gateway mode: {}", e)


async def _run_single_message(bus_adapter: StateMessageBusAdapter, message: str):
    """Process a single message and exit."""
    from nanobot.bus.events import InboundMessage

    msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content=message)

    logger.info("Processing single message: {}", message[:100])
    await bus_adapter.process_message(msg)


async def _run_interactive(bus_adapter: StateMessageBusAdapter):
    """Run in interactive CLI mode."""
    from nanobot.bus.events import InboundMessage

    logger.info("LangGraph interactive mode (Ctrl+C to exit)")

    try:
        while True:
            import sys

            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "/exit", "/quit", ":q"}:
                logger.info("Exiting interactive mode")
                break

            msg = InboundMessage(
                channel="cli", sender_id="user", chat_id="direct", content=user_input
            )

            await bus_adapter.process_message(msg)

    except KeyboardInterrupt:
        logger.info("Interactive mode stopped by user")
    except EOFError:
        logger.info("Interactive mode ended")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="nanobot LangGraph-based agent")
    parser.add_argument(
        "--mode",
        choices=["gateway", "agent"],
        default="gateway",
        help="Run mode: gateway (continuous) or agent (interactive)",
    )
    parser.add_argument("-m", "--message", help="Single message to process")

    args = parser.parse_args()

    asyncio.run(main(mode=args.mode, message=args.message))
