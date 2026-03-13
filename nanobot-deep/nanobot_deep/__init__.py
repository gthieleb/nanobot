"""Nanobot-deep: LangGraph/DeepAgents integration for nanobot."""

__version__ = "0.1.0"

from nanobot_deep.agent import DeepAgent, is_deepagents_available
from nanobot_deep.langgraph import (
    LangGraphBridge,
    SessionCheckpointer,
    translate_inbound_to_state,
    translate_result_to_outbound,
    extract_reply_context,
    should_delegate_task,
)
from nanobot_deep.config.schema import DeepAgentsConfig
from nanobot_deep.config.loader import (
    load_deepagents_config,
    save_deepagents_config,
    merge_with_nanobot_config,
)

__all__ = [
    "DeepAgent",
    "is_deepagents_available",
    "LangGraphBridge",
    "SessionCheckpointer",
    "translate_inbound_to_state",
    "translate_result_to_outbound",
    "extract_reply_context",
    "should_delegate_task",
    "DeepAgentsConfig",
    "load_deepagents_config",
    "save_deepagents_config",
    "merge_with_nanobot_config",
]
