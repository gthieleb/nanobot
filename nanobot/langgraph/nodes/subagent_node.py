"""Subagent spawn node."""

import asyncio
import uuid
from datetime import datetime
from typing import Any

from nanobot.langgraph.graph.state import AgentState


def spawn_subagent_node(state: AgentState, config) -> dict[str, Any]:
    """
    Spawn Subagent - erzeugt Background Task mit initialen Kontext.
    """
    last_message = state["messages"][-1]
    
    if not last_message:
        return state
    
    # Prüfe, ob tool_calls vorhanden sind
    tool_calls = last_message.get("tool_calls") if isinstance(last_message, dict) else []
    if not tool_calls and hasattr(last_message, "tool_calls"):
        tool_calls = last_message.tool_calls
    
    if not tool_calls:
        return state
    
    # Spawn Tool-Aufruf finden
    def is_spawn_call(tc):
        if isinstance(tc, dict):
            return tc.get("name") == "spawn"
        return getattr(tc, "name", None) == "spawn"

    spawn_call = next((tc for tc in tool_calls if is_spawn_call(tc)), None)
    
    if not spawn_call:
        return state
    
    task_id = str(uuid.uuid4())[:8]
    task_args = spawn_call.get("args") if isinstance(spawn_call, dict) else spawn_call.args
    task = task_args.get("task") if isinstance(task_args, dict) else task_args.task
    label = task_args.get("label") if isinstance(task_args, dict) else task_args.label
    
    # Initialen Kontext vom Main Agent übernehmen
    initial_context = {
        "messages": [msg for msg in state["messages"][-10:]],  # Letzte 10 Nachrichten
        "workspace": config["configurable"]["workspace"],
        "current_tools": state["current_tools"]
    }
    
    # Subagent Task persistieren
    subagent_task = {
        "task_id": task_id,
        "task": task,
        "label": label or (task[:30] + "..." if len(task) > 30 else task),
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "initial_context": initial_context
    }
    state["subagent_tasks"].append(subagent_task)
    
    # Background Task starten
    try:
        from nanobot.langgraph.subagent.manager import SubagentManager
        manager = config["configurable"]["subagent_manager"]
        
        asyncio.create_task(
            manager.run_subagent(
                task_id=task_id,
                task=task,
                label=label,
                initial_context=initial_context,
                state_ref=state
            )
        )
    except ImportError as e:
        from loguru import logger
        logger.error("Failed to import SubagentManager: {}", e)
        # Fallback: Task direkt in State erstellen
        pass
    
    # Feedback-Message an den User
    feedback_content = (
        f"Subagent [{label or (task[:30] + '...' if len(task) > 30 else task)}] "
        f"started (id: {task_id}). "
        f"Running in background with {len(state['subagent_tasks'])} active tasks."
    )
    
    feedback_message = {
        "role": "assistant",
        "content": feedback_content
    }
    state["messages"].append(feedback_message)
    
    return state
