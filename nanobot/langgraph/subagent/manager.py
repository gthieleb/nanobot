"""Hybrid Subagent Manager for LangGraph integration."""

import asyncio
from datetime import datetime
from typing import Optional, Any

from nanobot.langgraph.graph.state import SubagentState, AgentState
from nanobot.langgraph.subagent.adapter import SubagentToolAdapter
from nanobot.langgraph.subagent.messenger import SubagentMessenger
from loguru import logger


class SubagentManager:
    """
    Hybrid-Subagent Manager:
    - Unabhängig (eigener Loop)
    - Initialer Kontext vom Main
    - Asynchrone ReAct-Adjustierungen
    """

    def __init__(self, provider, workspace, bus, main_state_ref: dict):
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.main_state_ref = main_state_ref  # Für asynchrone Updates

        # Adapter für Tools (kein spawn/message Tool)
        self.tool_adapter = SubagentToolAdapter(
            workspace=workspace,
            brave_api_key=None,  # Wird später injiziert
            exec_config={"timeout": 60},  # Default
        )

        # Messenger für asynchrone Adjustierungen
        self.messenger = SubagentMessenger(bus=bus, on_adjustment=self.handle_adjustment)

    async def run_subagent(
        self, task_id: str, task: str, label: Optional[str], initial_context: dict, state_ref: dict
    ):
        """
        Führt Subagent mit ReAct-Loop aus.
        """
        logger.info("Subagent [{}] starting: {}", task_id, label)

        # Subagent State initialisieren mit Main-Kontext
        subagent_state: SubagentState = {
            "task_id": task_id,
            "task": task,
            "initial_context": initial_context,
            "messages": [
                {"role": "system", "content": self._build_subagent_prompt(task)},
                {"role": "user", "content": task},
            ],
            "iteration": 0,
            "max_iterations": 15,
            "result": None,
            "status": "running",
        }

        # ReAct Loop
        while subagent_state["iteration"] < subagent_state["max_iterations"]:
            subagent_state["iteration"] += 1

            # LLM-Aufruf
            try:
                response = await self.provider.chat(
                    messages=subagent_state["messages"],
                    tools=self.tool_adapter.get_definitions(),
                    model=self.provider.get_default_model(),
                )
            except Exception as e:
                logger.error("Subagent [{}] LLM call failed: {}", task_id, e)
                subagent_state["status"] = "failed"
                subagent_state["result"] = f"Error: {str(e)}"
                break

            # Tool-Aufrufe?
            if response.tool_calls:
                # Assistant Message anhängen
                tool_calls_data = [
                    {"id": tc.id, "name": tc.name, "args": tc.arguments}
                    for tc in response.tool_calls
                ]

                subagent_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": tool_calls_data,
                    }
                )

                # Tools ausführen
                for tool_call in response.tool_calls:
                    try:
                        result = await self.tool_adapter.execute(
                            tool_call.name, tool_call.arguments
                        )

                        subagent_state["messages"].append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.name,
                                "content": result,
                            }
                        )
                    except Exception as e:
                        logger.error("Subagent [{}] tool {} failed: {}", task_id, tool_call.name, e)
                        subagent_state["messages"].append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.name,
                                "content": f"Error: {str(e)}",
                            }
                        )

                # Asynchrone Adjustierung prüfen (jede 3. Iteration)
                if subagent_state["iteration"] % 3 == 0:
                    await self.messenger.request_adjustment(
                        task_id=task_id, current_context=subagent_state["messages"]
                    )

            else:
                # Keine Tool-Aufrufe → fertig
                subagent_state["result"] = response.content
                subagent_state["status"] = "completed"
                break

        # Ergebnis an Main Agent melden
        await self._announce_completion(task_id, label, task, subagent_state["result"], state_ref)

    async def handle_adjustment(self, task_id: str, adjustment: dict):
        """
        Asynchrone Adjustierung vom Main Agent.
        """
        logger.info("Subagent [{}] received adjustment: {}", task_id, adjustment)

        # Adjustierung in den aktuellen Kontext einarbeiten
        # (hier implementieren je nach Anforderung)
        pass

    def _build_subagent_prompt(self, task: str) -> str:
        """
        System-Prompt für Subagent.
        """
        return f"""# Subagent

You are a subagent spawned by the main agent to complete a specific task.

## Task
{task}

## Rules
1. Stay focused - complete only the assigned task
2. You may request adjustments from the main agent every 3 iterations
3. Be concise but informative

## What You Cannot Do
- Send messages directly to users
- Spawn other subagents
- Access the main agent's conversation history (only initial context provided)"""

    async def _announce_completion(
        self, task_id: str, label: Optional[str], task: str, result: Optional[str], state_ref: dict
    ):
        """
        Meldet Abschluss an Main Agent via Message Bus.
        """
        # Thread-sichere Kopie des State machen
        if "subagent_tasks" not in state_ref:
            state_ref["subagent_tasks"] = []

        # Task-Updates in einer Kopie vornehmen
        updated_tasks = list(state_ref["subagent_tasks"])
        for task_entry in updated_tasks:
            if task_entry.get("task_id") == task_id:
                task_entry["status"] = "completed"
                task_entry["result"] = result
                task_entry["completed_at"] = datetime.now().isoformat()
                break

        # Updates in State zurückschreiben
        state_ref["subagent_tasks"] = updated_tasks

        # System Message an State anhängen
        display_label = label or "subagent"
        result_text = result or "No result"
        announce_content = f"""[Subagent '{display_label}' completed]

Task: {task}

Result:
{result_text}

Summarize this naturally for the user. Keep it brief (1-2 sentences)."""

        if "messages" not in state_ref:
            state_ref["messages"] = []

        # Thread-sicheres Anhängen
        new_messages = list(state_ref["messages"]) + [
            {"role": "system", "content": announce_content}
        ]
        state_ref["messages"] = new_messages

        logger.info("Subagent [{}] completed", task_id)
