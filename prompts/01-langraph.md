# 🎯 LangGraph-Integration für Nanobot - Implementierungsplan

## Zusammenfassung der Anforderungen

| Komponente | Spezifikation |
|------------|---------------|
| **Subagent-Typ** | Unabhängig mit initialer Main-Kontext-Übernahme |
| **Asynchrone Adjustierungen** | ReAct-Loop mit asynchroner Message-Bus-Kommunikation |
| **Persistenz** | MemorySaver (in-memory) |
| **Monitoring** | Basic Logging (`loguru`) |
| **Migration** | Parallel (Coexistenz) |
| **Tool-Registry** | Nanobot-Registry + LangGraph-Adapter |
| **BaseDir** | `--basedir` Option (default `~/.nanobot`) |

---

## 1. Architektur-Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     nanobot/langgraph/ (neuer Ordner)                │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                   Main Graph (Agent Loop)                      │ │
│  │                                                              │ │
│  │  State:                                                      │ │
│  │  - messages: [] (Konversationshistorie)                        │ │
│  │  - subagent_tasks: [] (PERSISTIERT durch MemorySaver)          │ │
│  │  - current_context: {} (Initialer Kontext für Subagents)        │ │
│  │                                                              │ │
│  │  Nodes:                                                      │ │
│  │  ┌─────────┐    ┌─────────┐    ┌──────────┐                 │ │
│  │  │ agent   │───▶│ tools   │───▶│ update   │                 │ │
│  │  │ (LLM)   │◀───│ (Nanobot│    │ state    │                 │ │
│  │  └────┬────┘    │Registry)│    └──────────┘                 │ │
│  │       │         └─────────┘                                   │ │
│  │       ▼                                                      │ │
│  │  should_continue() ──▶ Ja: tools / Nein: END                 │ │
│  └───────┬────────────────────────────────────────────────────────────┘ │
│          │                                                        │
│          │ spawn_tool()                                            │
│          │                                                        │
│          ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │         Subagent Manager (Hybrid Design)                     │ │
│  │                                                              │ │
│  │  spawn(task, context) ──▶ asyncio.create_task()             │ │
│  │       │                                                      │ │
│  │       ├─ Initialer Kontext vom Main Agent übernehmen           │ │
│  │       ├─ Eigener ReAct-Loop (isolierte Tools)                │ │
│  │       └─ Asynchrone Adjustierungen über Message Bus          │ │
│  │              ├─ Request: Main Agent Feedback einholen           │ │
│  │              └─ Response: Adjusted Kontext zurückgeben        │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│          │                                                        │
│          ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │         Message Bus Adapter                                  │ │
│  │  Verbindet LangGraph-State mit existierendem Nanobot-Bus     │ │
│  │  (für Channel-Kompatibilität während Migration)              │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────────────┘

         Koexistenz (Parallel-Migration)
┌──────────────────────┐           ┌──────────────────────┐
│  Nanobot Original   │◀─────────▶│  Nanobot LangGraph  │
│  (bestehend)       │   Shared   │  (neu)            │
│                    │   Bus      │                    │
└──────────────────────┘           └──────────────────────┘
```

---

## 2. Verzeichnisstruktur

```
nanobot/
├── agent/                           # Original (unverändert)
│   ├── loop.py
│   ├── subagent.py
│   └── tools/
├── langgraph/                       # NEU - LangGraph-Implementierung
│   ├── __init__.py
│   ├── main.py                      # Einstiegspunkt für LangGraph-Agent
│   ├── graph/                       # Graph-Definitionen
│   │   ├── __init__.py
│   │   ├── main_graph.py             # Haupt-Graph (Agent Loop)
│   │   ├── subgraph.py              # Subagent-Subgraph
│   │   └── state.py                # State-Definitionen
│   ├── nodes/                       # Graph Nodes
│   │   ├── __init__.py
│   │   ├── agent_node.py            # LLM-Call
│   │   ├── tools_node.py           # Tool-Execution
│   │   ├── state_update_node.py    # State-Management
│   │   └── subagent_node.py       # Subagent-Spawn-Node
│   ├── subagent/                   # Subagent-Management
│   │   ├── __init__.py
│   │   ├── manager.py              # Hybrid-SubagentManager
│   │   ├── adapter.py             # Nanobot Registry Adapter
│   │   └── messenger.py           # Asynchrone Adjustierungs-Logik
│   ├── tools/                      # Tool-Adapter
│   │   ├── __init__.py
│   │   ├── adapter.py             # Nanobot Registry → LangChain
│   │   └── spawn_tool.py         # Spawn Tool für Main Graph
│   ├── bus/                        # Message Bus Adapter
│   │   ├── __init__.py
│   │   ├── state_adapter.py       # State ↔ MessageBus Bridge
│   │   └── event_mapper.py       # Event-Mapping
│   └── config/                    # LangGraph-spezifische Config
│       ├── __init__.py
│       └── settings.py
├── bus/                            # Original MessageBus (wiederverwendet)
└── channels/                       # Channel-kompatibel mit beiden
```

---

## 3. Phase-by-Phase Implementierung

### Phase 1: Grundgerüst & State (ca. 2-3 Stunden)

**1.1 Verzeichnisse erstellen**
```bash
mkdir -p nanobot/langgraph/{graph,nodes,subagent,tools,bus,config}
touch nanobot/langgraph/__init__.py
```

**1.2 State-Definitionen (`langgraph/graph/state.py`)**

```python
from typing import Annotated, Sequence, TypedDict, Any
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    State für den Haupt-Agenten (persistiert durch MemorySaver).
    """
    messages: Annotated[Sequence, add_messages]
    current_tools: list[str]
    subagent_tasks: list[dict[str, Any]]
    current_context: dict[str, Any]

class SubagentState(TypedDict):
    """
    State für Subagents (isoliert, aber initialer Kontext von Main).
    """
    task_id: str
    task: str
    initial_context: dict[str, Any]
    messages: list[dict[str, Any]]
    iteration: int
    max_iterations: int
    result: str | None
    status: str  # "running", "completed", "failed", "awaiting_adjustment"
```

**1.3 Einstiegspunkt (`langgraph/main.py`)**

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from nanobot.langgraph.graph.main_graph import create_main_graph
from nanobot.langgraph.bus.state_adapter import StateMessageBusAdapter
from nanobot.config.loader import load_config

async def main():
    config = load_config()

    # Graph erstellen
    graph = create_main_graph(config)

    # Checkpointer für Persistenz
    checkpointer = MemorySaver()

    # Graph kompilieren
    app = graph.compile(checkpointer=checkpointer)

    # Message Bus Adapter für Koexistenz mit Nanobot
    bus_adapter = StateMessageBusAdapter(app)

    # Warte auf Messages (wie Nanobot's AgentLoop.run())
    while True:
        msg = await bus_adapter.consume_from_bus()
        await bus_adapter.process_message(msg)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

### Phase 2: Main Graph (Agent Loop) (ca. 3-4 Stunden)

**2.1 Haupt-Graph (`langgraph/graph/main_graph.py`)**

```python
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from nanobot.langgraph.graph.state import AgentState
from nanobot.langgraph.nodes.agent_node import call_model
from nanobot.langgraph.nodes.tools_node import execute_tools
from nanobot.langgraph.nodes.state_update_node import update_state
from nanobot.langgraph.nodes.subagent_node import spawn_subagent_node

def should_continue(state: AgentState) -> str:
    """
    Entscheidet, ob der Loop weiterläuft (Tool-Aufrufe) oder endet.
    """
    last_message = state["messages"][-1]

    # Wenn letzte Message Tool-Aufrufe enthält → weiter zu Tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"

    # Keine Tool-Aufrufe → Ende
    return "end"

def create_main_graph(config) -> StateGraph:
    """Erstellt den Haupt-Graph für den Agent Loop."""

    graph = StateGraph(AgentState)

    # Nodes registrieren
    graph.add_node("agent", call_model)
    graph.add_node("tools", execute_tools)
    graph.add_node("update_state", update_state)
    graph.add_node("spawn_subagent", spawn_subagent_node)

    # Kanten definieren
    graph.set_entry_point("agent")

    # Agent → Tools oder Spawn?
    graph.add_conditional_edges(
        "agent",
        should_spawn_subagent,
        {
            "spawn": "spawn_subagent",
            "continue": "tools",
            "end": "update_state"
        }
    )

    # Spawn Subagent → zurück zu Agent
    graph.add_edge("spawn_subagent", "update_state")

    # Tools → State Update → zurück zu Agent
    graph.add_edge("tools", "update_state")
    graph.add_edge("update_state", "agent")

    # Update State → Prüfen, ob fertig
    graph.add_conditional_edges(
        "update_state",
        should_continue,
        {
            "continue": "agent",
            "end": END
        }
    )

    return graph

def should_spawn_subagent(state: AgentState) -> str:
    """
    Prüft, ob spawn tool aufgerufen wurde.
    """
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
            if tool_call.name == "spawn":
                return "spawn"

    # Sonst: normaler Tool-Loop oder Ende
    return should_continue(state)
```

**2.2 Agent Node (`langgraph/nodes/agent_node.py`)**

```python
from langchain_core.messages import AIMessage, HumanMessage

from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.langgraph.graph.state import AgentState

async def call_model(state: AgentState, config) -> dict:
    """
    LLM-Aufruf - äquivalent zu AgentLoop._run_agent_loop()
    """
    # LLM Provider aus config holen (wiederverwendet Nanobot's Provider)
    provider = config["configurable"]["provider"]

    # State in Message-Liste konvertieren
    messages = [
        {"role": msg.type if hasattr(msg, "type") else "assistant",
         "content": msg.content if hasattr(msg, "content") else str(msg)}
        for msg in state["messages"]
    ]

    # Tool-Definitionen über Adapter holen
    from nanobot.langgraph.tools.adapter import get_tool_definitions
    tools = get_tool_definitions()

    # LLM aufrufen
    response: LLMResponse = await provider.chat(
        messages=messages,
        tools=tools,
        model=config["configurable"]["model"],
        temperature=config["configurable"]["temperature"],
        max_tokens=config["configurable"]["max_tokens"]
    )

    # Response in LangChain Message konvertieren
    if response.tool_calls:
        # Tool-Aufrufe
        ai_message = AIMessage(
            content=response.content or "",
            tool_calls=[
                {"id": tc.id, "name": tc.name, "args": tc.arguments}
                for tc in response.tool_calls
            ]
        )
    else:
        # Normale Antwort
        ai_message = AIMessage(content=response.content or "")

    # State aktualisieren
    state["messages"].append(ai_message)
    state["current_tools"] = [tc.name for tc in response.tool_calls]

    return state
```

**2.3 Tools Node (`langgraph/nodes/tools_node.py`)**

```python
from nanobot.langgraph.graph.state import AgentState
from nanobot.langgraph.tools.adapter import execute_nanobot_tool

async def execute_tools(state: AgentState, config) -> dict:
    """
    Führt Tools aus - Nanobot Registry über Adapter.
    """
    last_message = state["messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return state

    # Tool Registry aus config
    tool_registry = config["configurable"]["tool_registry"]

    # Alle Tool-Aufrufe ausführen
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        try:
            # Über Adapter mit Nanobot Registry ausführen
            result = await execute_nanobot_tool(
                tool_registry,
                tool_name,
                tool_args
            )

            # Tool-Result als Message anhängen
            from langchain_core.messages import ToolMessage
            tool_message = ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            )
            state["messages"].append(tool_message)

        except Exception as e:
            error_message = ToolMessage(
                content=f"Error executing {tool_name}: {str(e)}",
                tool_call_id=tool_call["id"]
            )
            state["messages"].append(error_message)

    return state
```

**2.4 State Update Node (`langgraph/nodes/state_update_node.py`)**

```python
from nanobot.langgraph.graph.state import AgentState

def update_state(state: AgentState) -> dict:
    """
    Aktualisiert State für nächsten Loop.
    """
    # Current Context aktualisieren
    # (wird von Subagent initial gefüllt)

    # Subagent-Tasks bereinigen (abgeschlossene entfernen)
    state["subagent_tasks"] = [
        task for task in state["subagent_tasks"]
        if task.get("status") not in ["completed", "failed", "cancelled"]
    ]

    return state
```

**2.5 Spawn Subagent Node (`langgraph/nodes/subagent_node.py`)**

```python
import uuid
from datetime import datetime

from nanobot.langgraph.graph.state import AgentState

def spawn_subagent_node(state: AgentState, config) -> dict:
    """
    Spawn Subagent - erzeugt Background Task mit initialen Kontext.
    """
    # Spawn Tool-Argumente aus letzter Message
    last_message = state["messages"][-1]
    spawn_call = next(
        (tc for tc in last_message.tool_calls if tc["name"] == "spawn"),
        None
    )

    if not spawn_call:
        return state

    task_id = str(uuid.uuid4())[:8]
    task = spawn_call["args"]["task"]
    label = spawn_call["args"].get("label")

    # Initialen Kontext vom Main Agent übernehmen
    initial_context = {
        "messages": [msg for msg in state["messages"][-10:]],  # Letzte 10 Nachrichten
        "workspace": config["configurable"]["workspace"],
        "current_tools": state["current_tools"]
    }

    # Subagent Task persistieren
    state["subagent_tasks"].append({
        "task_id": task_id,
        "task": task,
        "label": label or task[:30],
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "initial_context": initial_context
    })

    # Background Task starten
    import asyncio
    from nanobot.langgraph.subagent.manager import SubagentManager
    manager = config["configurable"]["subagent_manager"]

    asyncio.create_task(
        manager.run_subagent(
            task_id=task_id,
            task=task,
            label=label,
            initial_context=initial_context,
            state_ref=state  # Für asynchrone Adjustierungen
        )
    )

    # Feedback-Message an den User
    from langchain_core.messages import AIMessage
    feedback_msg = AIMessage(
        content=f"Subagent [{label or task[:30]}] started (id: {task_id}). "
                 f"Running in background with {len(state['subagent_tasks'])} active tasks."
    )
    state["messages"].append(feedback_msg)

    return state
```

---

### Phase 3: Subagent Hybrid-Manager (ca. 3-4 Stunden)

**3.1 Subagent Manager (`langgraph/subagent/manager.py`)**

```python
import asyncio
from loguru import logger

from nanobot.langgraph.graph.state import SubagentState, AgentState
from nanobot.langgraph.subagent.adapter import SubagentToolAdapter
from nanobot.langgraph.subagent.messenger import SubagentMessenger

class SubagentManager:
    """
    Hybrid-Subagent Manager:
    - Unabhängig (eigener Loop)
    - Initialer Kontext vom Main
    - Asynchrone ReAct-Adjustierungen
    """

    def __init__(
        self,
        provider,
        workspace,
        bus,
        main_state_ref: AgentState
    ):
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.main_state_ref = main_state_ref  # Für asynchrone Updates

        # Adapter für Tools (kein spawn/message Tool)
        self.tool_adapter = SubagentToolAdapter()

        # Messenger für asynchrone Adjustierungen
        self.messenger = SubagentMessenger(
            bus=bus,
            on_adjustment=self.handle_adjustment
        )

    async def run_subagent(
        self,
        task_id: str,
        task: str,
        label: str | None,
        initial_context: dict,
        state_ref: AgentState
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
                {"role": "user", "content": task}
            ],
            "iteration": 0,
            "max_iterations": 15,
            "result": None,
            "status": "running"
        }

        # ReAct Loop
        while subagent_state["iteration"] < subagent_state["max_iterations"]:
            subagent_state["iteration"] += 1

            # LLM-Aufruf
            response = await self.provider.chat(
                messages=subagent_state["messages"],
                tools=self.tool_adapter.get_definitions(),
                model=self.provider.get_default_model()
            )

            # Tool-Aufrufe?
            if response.tool_calls:
                # Assistant Message anhängen
                subagent_state["messages"].append({
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "args": tc.arguments}
                        for tc in response.tool_calls
                    ]
                })

                # Tools ausführen
                for tool_call in response.tool_calls:
                    result = await self.tool_adapter.execute(tool_call.name, tool_call.arguments)

                    subagent_state["messages"].append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": result
                    })

                # Asynchrone Adjustierung prüfen (jede 3. Iteration)
                if subagent_state["iteration"] % 3 == 0:
                    await self.messenger.request_adjustment(
                        task_id=task_id,
                        current_context=subagent_state["messages"]
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

You are a subagent spawned by main agent to complete a specific task.

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
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        state_ref: AgentState
    ):
        """
        Meldet Abschluss an Main Agent via Message Bus.
        """
        # Subagent Task als completed markieren
        for task_entry in state_ref["subagent_tasks"]:
            if task_entry["task_id"] == task_id:
                task_entry["status"] = "completed"
                task_entry["result"] = result
                task_entry["completed_at"] = datetime.now().isoformat()
                break

        # System Message an State anhängen
        announce_content = f"""[Subagent '{label}' completed]

Task: {task}

Result:
{result}

Summarize this naturally for the user. Keep it brief (1-2 sentences)."""

        state_ref["messages"].append({
            "role": "system",
            "content": announce_content
        })

        logger.info("Subagent [{}] completed", task_id)
```

**3.2 Subagent Messenger (`langgraph/subagent/messenger.py`)**

```python
import asyncio
from loguru import logger

class SubagentMessenger:
    """
    Ermöglicht asynchrone Adjustierungen zwischen Subagent und Main Agent.
    """

    def __init__(self, bus, on_adjustment):
        self.bus = bus
        self.on_adjustment = on_adjustment
        self._pending_adjustments: dict[str, asyncio.Queue] = {}

    async def request_adjustment(
        self,
        task_id: str,
        current_context: list[dict]
    ) -> dict | None:
        """
        Requestet Adjustierung vom Main Agent.
        """
        # Queue für diese Request erstellen
        self._pending_adjustments[task_id] = asyncio.Queue()

        # Request über Message Bus an Main Agent
        await self.bus.publish_inbound(
            InboundMessage(
                channel="system",
                sender_id=f"subagent:{task_id}",
                chat_id="adjustment_request",
                content=f"""[Adjustment Request from Subagent {task_id}]

Current Context:
{current_context[-5:] if len(current_context) > 5 else current_context}

Should I adjust my approach? Please provide feedback.
"""
            )
        )

        # Warte auf Response mit Timeout
        try:
            adjustment = await asyncio.wait_for(
                self._pending_adjustments[task_id].get(),
                timeout=30.0
            )
            return adjustment
        except asyncio.TimeoutError:
            logger.warning("Subagent [{}] adjustment request timed out", task_id)
            return None
        finally:
            self._pending_adjustments.pop(task_id, None)

    def deliver_adjustment(self, task_id: str, adjustment: dict):
        """
        Liefert Adjustierung an wartenden Subagent.
        """
        queue = self._pending_adjustments.get(task_id)
        if queue:
            queue.put_nowait(adjustment)
        else:
            logger.warning("No pending adjustment queue for subagent [{}]", task_id)
```

**3.3 Subagent Tool Adapter (`langgraph/subagent/adapter.py`)**

```python
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool

class SubagentToolAdapter:
    """
    Adapter für Subagent-Tools (ohne spawn/message Tool).
    """

    def __init__(self, workspace=None, brave_api_key=None, exec_config=None):
        self.registry = ToolRegistry()

        # Nur erlaubte Tools registrieren
        self.registry.register(ReadFileTool(workspace=workspace))
        self.registry.register(WriteFileTool(workspace=workspace))
        self.registry.register(EditFileTool(workspace=workspace))
        self.registry.register(ListDirTool(workspace=workspace))
        self.registry.register(ExecTool(working_dir=workspace, **(exec_config or {})))
        self.registry.register(WebSearchTool(api_key=brave_api_key))
        self.registry.register(WebFetchTool())

    def get_definitions(self) -> list[dict]:
        """Gibt Tool-Definitionen für LLM zurück."""
        return self.registry.get_definitions()

    async def execute(self, tool_name: str, args: dict) -> str:
        """Führt Tool aus."""
        return await self.registry.execute(tool_name, args)
```

---

### Phase 4: Tool-Adapter (ca. 2-3 Stunden)

**4.1 Nanobot Registry Adapter (`langgraph/tools/adapter.py`)**

```python
from nanobot.agent.tools.registry import ToolRegistry
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class NanobotToolAdapter(BaseTool):
    """
    Adapter für Nanobot Registry → LangChain Tool.
    """

    def __init__(self, tool_name: str, tool_registry: ToolRegistry):
        self.tool_name = tool_name
        self.registry = tool_registry

        # Nanobot Tool-Definition holen
        self.nanobot_tool = tool_registry.get(tool_name)
        if not self.nanobot_tool:
            raise ValueError(f"Tool {tool_name} not found in registry")

    def _run(self, **kwargs) -> str:
        """
        Synchroner Aufruf (wird von LangChain erwartet).
        """
        import asyncio

        # Async Task ausführen
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            self.registry.execute(self.tool_name, kwargs)
        )

        return result

    async def _arun(self, **kwargs) -> str:
        """Asynchroner Aufruf."""
        return await self.registry.execute(self.tool_name, kwargs)

    @property
    def name(self) -> str:
        return self.tool_name

    @property
    def description(self) -> str:
        return self.nanobot_tool.description or f"Tool: {self.tool_name}"

    @property
    def args_schema(self) -> type[BaseModel]:
        """Pydantic Schema aus Tool-Parametern."""
        fields = {}

        for param_name, param_info in self.nanobot_tool.parameters.get("properties", {}).items():
            fields[param_name] = Field(
                description=param_info.get("description", ""),
                default=... if param_name in self.nanobot_tool.parameters.get("required", [])
                             else None
            )

        return type(f"{self.tool_name}Args", (BaseModel,), fields)

def get_tool_definitions(tool_registry: ToolRegistry) -> list[dict]:
    """
    Konvertiert Nanobot Tool-Registry in OpenAI-Format für LLM.
    """
    return tool_registry.get_definitions()

async def execute_nanobot_tool(
    tool_registry: ToolRegistry,
    tool_name: str,
    args: dict
) -> str:
    """
    Führt Tool über Nanobot Registry aus.
    """
    return await tool_registry.execute(tool_name, args)
```

**4.2 Spawn Tool (`langgraph/tools/spawn_tool.py`)**

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class SpawnToolInput(BaseModel):
    task: str = Field(description="The task for subagent to complete")
    label: str | None = Field(default=None, description="Optional short label for display")

class SpawnTool(BaseTool):
    """
    Spawn Tool für Main Graph.
    """

    name = "spawn"
    description = (
        "Spawn a subagent to handle a task in background. "
        "Use this for complex or time-consuming tasks that can run independently. "
        "The subagent will complete the task and report back when done."
    )
    args_schema = SpawnToolInput

    def _run(self, task: str, label: str | None = None) -> str:
        """
        Wird vom Spawn Subagent Node aufgerufen (nicht direkt hier).
        """
        # Die eigentliche Ausführung passiert im spawn_subagent_node
        # Dieses Tool ist nur für die LLM-Definition
        return f"Spawning subagent for task: {task}"
```

---

### Phase 5: Message Bus Adapter (ca. 2 Stunden)

**5.1 State-Bus-Adapter (`langgraph/bus/state_adapter.py`)**

```python
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.langgraph.graph.state import AgentState

class StateMessageBusAdapter:
    """
    Verbindet LangGraph-State mit existierendem Nanobot Message Bus.
    Ermöglicht parallele Coexistenz.
    """

    def __init__(self, graph_app, nanobot_bus):
        self.graph_app = graph_app
        self.nanobot_bus = nanobot_bus
        self._active_threads: dict[str, Any] = {}

    async def consume_from_bus(self) -> InboundMessage:
        """
        Wartet auf Messages vom Nanobot Bus.
        """
        return await self.nanobot_bus.consume_inbound()

    async def process_message(self, msg: InboundMessage):
        """
        Verarbeitet Message über LangGraph.
        """
        # Thread ID für Session (wie Nanobot Sessions)
        thread_id = f"{msg.channel}:{msg.chat_id}"

        config = {
            "configurable": {
                "thread_id": thread_id,
                # Nanobot Config weitergeben
                "provider": self._get_provider(),
                "tool_registry": self._get_tool_registry(),
                "workspace": self._get_workspace(),
                "subagent_manager": self._get_subagent_manager(),
                "model": self._get_model(),
                "temperature": self._get_temperature(),
                "max_tokens": self._get_max_tokens()
            }
        }

        # Initial State
        initial_state = {
            "messages": [("user", msg.content)],
            "current_tools": [],
            "subagent_tasks": [],
            "current_context": {}
        }

        # Graph ausführen
        result = await self.graph_app.ainvoke(initial_state, config=config)

        # Antwort an Bus senden
        if result and result["messages"]:
            last_message = result["messages"][-1]
            await self.nanobot_bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=last_message.content,
                    metadata=msg.metadata
                )
            )

    def _get_provider(self):
        """Lädt Provider aus Nanobot Config."""
        from nanobot.config.loader import load_config
        config = load_config()
        from nanobot.cli.commands import _make_provider
        return _make_provider(config)

    def _get_tool_registry(self):
        """Lädt Tool Registry."""
        from nanobot.config.loader import load_config
        from nanobot.agent.loop import AgentLoop
        from nanobot.bus.queue import MessageBus

        config = load_config()
        registry = ToolRegistry()
        # Tools registrieren (wie in AgentLoop._register_default_tools)
        # ...
        return registry

    def _get_workspace(self):
        """Lädt Workspace."""
        from nanobot.config.loader import load_config
        return load_config().workspace_path

    def _get_subagent_manager(self):
        """Lädt oder erstellt Subagent Manager."""
        from nanobot.langgraph.subagent.manager import SubagentManager

        provider = self._get_provider()
        workspace = self._get_workspace()

        # State-Referenz muss später injiziert werden
        # Hier placeholder
        return SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=self.nanobot_bus,
            main_state_ref={}
        )

    def _get_model(self):
        from nanobot.config.loader import load_config
        return load_config().agents.defaults.model

    def _get_temperature(self):
        from nanobot.config.loader import load_config
        return load_config().agents.defaults.temperature

    def _get_max_tokens(self):
        from nanobot.config.loader import load_config
        return load_config().agents.defaults.max_tokens
```

---

### Phase 6: CLI-Integration & Testing (ca. 3-4 Stunden)

**6.1 Neuer CLI-Befehl (`langgraph/config/settings.py`)**

```python
# LangGraph-spezifische Config
class LangGraphSettings:
    """Konfiguration für LangGraph-Integration."""

    enabled: bool = False  # Flag für Parallel-Migration
    checkpoint_type: str = "memory"  # memory, postgres, redis
    subagent_max_iterations: int = 15
    adjustment_interval: int = 3  # Alle N Iterationen Adjustierung anfragen
```

**6.2 CLI-Befehl erweitern (`cli/commands.py`)**

```python
# Neuer Befehl
@app.command()
def langgraph(
    message: str = typer.Option(None, "--message", "-m", help="Message to send"),
    mode: str = typer.Option("agent", "--mode", help="agent or gateway"),
    basedir: str = typer.Option(Path.home() / ".nanobot", "--basedir", help="Base directory for config and data")
):
    """Use LangGraph-based agent."""
    from nanobot.langgraph.main import main

    if message:
        # Single message
        # ...
        pass
    else:
        # Gateway mode
        asyncio.run(main())

# Oder bestehenden Befehl erweitern:
@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m"),
    langgraph: bool = typer.Option(False, "--langgraph", help="Use LangGraph"),
    basedir: str = typer.Option(None, "--basedir", help="Base directory for config and data"),
    # ...
):
    """Interact with agent."""
    if langgraph:
        from nanobot.langgraph.main import main
        # ...
    else:
        # Original Nanobot
        # ...
```

---

### Phase 7: Migration & Testing (ca. 4-6 Stunden)

**7.1 Parallele Coexistenz sicherstellen**

Schritte:

1. **Original Nanobot unberührt lassen**
   - Alle Änderungen nur in `nanobot/langgraph/`
   - Keine Änderungen an `agent/`, `bus/`, `channels/`

2. **Nanobot-Bus wiederverwenden**
   - LangGraph verwendet denselben `MessageBus`
   - Beide Agenten können parallel laufen

3. **Channels kompatibel machen**
   - Keine Änderungen an Channel-Implementierung nötig
   - Message Bus Abstraktion reicht

**7.2 Testing-Plan**

```bash
# 1. Unit Tests
pytest tests/langgraph/test_state.py
pytest tests/langgraph/test_subagent.py
pytest tests/langgraph/test_graph.py

# 2. Integration Tests
pytest tests/langgraph/integration/test_agent_loop.py
pytest tests/langgraph/integration/test_subagent_adjustment.py

# 3. Parallel-Migration Tests
# Terminal 1: Original Nanobot
nanobot gateway

# Terminal 2: LangGraph
nanobot langgraph --mode gateway

# Channel Tests:
# - Nachricht an Telegram Bot → Antwort von LangGraph?
# - Spawn Subagent → Background Task läuft?
# - Asynchrone Adjustierung → Feedback vom Main?
```

**7.3 Test-Szenarien**

| Szenario | Erwartetes Verhalten |
|----------|---------------------|
| Einfache Frage | Antwort vom LLM, keine Tools |
| Datei lesen | `read_file` Tool wird aufgerufen |
| Spawn Subagent | Subagent läuft im Hintergrund, Main antwortet mit Task-ID |
| Lange Task | Subagent iteriert, alle 3 Iterationen Adjustierungs-Request |
| Subagent fertig | Ergebnis in State, System Message generiert |
| Parallel laufende Subagents | Alle in `state.subagent_tasks[]` sichtbar |
| Crash/Restart | State verloren (MemorySaver = in-memory) |

---

### Phase 8: Dokumentation & Rollout (ca. 2 Stunden)

**8.1 README erweitern**

```markdown
## LangGraph Integration (Beta)

Nanobot unterstützt jetzt LangGraph als alternative Backend.

### Aktivieren

In `~/.nanobot/config.json`:

```json
{
  "langgraph": {
    "enabled": true,
    "checkpoint": "memory"
  }
}
```

### CLI

```bash
# Mit LangGraph laufen
nanobot agent --langgraph

# Gateway mit LangGraph
nanobot langgraph --mode gateway

# Eigenes Verzeichnis verwenden
nanobot --basedir /path/to/custom/dir
```

### Features

- **ReAct-Loop**: Automatischer Tool-Aufruf Loop
- **Persistierte Subagent-Tasks**: In State gespeichert (siehbar für LLM)
- **Asynchrone Adjustierungen**: Subagents können Feedback vom Main Agent anfragen
- **Parallel-Migration**: Nanobot und LangGraph können gleichzeitig laufen

### Migration Status

| Komponente | Status |
|------------|---------|
| Agent Loop | ✅ Implementiert |
| Subagent Manager | ✅ Implementiert |
| Tool Registry Adapter | ✅ Implementiert |
| Message Bus Adapter | ✅ Implementiert |
| Channels | ✅ Kompatibel |
| Cron | ⏳ TODO |
| MCP | ⏳ TODO |
```

---

## 4. Gesamtaufwand & Timeline

| Phase | Aufwand | Abhängigkeit |
|-------|----------|--------------|
| 1. Grundgerüst & State | 2-3h | - |
| 2. Main Graph (Agent Loop) | 3-4h | Phase 1 |
| 3. Subagent Hybrid-Manager | 3-4h | Phase 1 |
| 4. Tool-Adapter | 2-3h | Phase 1 |
| 5. Message Bus Adapter | 2h | Phase 1 |
| 6. CLI-Integration & Testing | 3-4h | Phase 1-5 |
| 7. Migration & Testing | 4-6h | Phase 1-6 |
| 8. Dokumentation | 2h | Phase 1-7 |
| **Gesamt** | **21-29h** | **~3-4 Tage** |

---

## 5. Risiken & Mitigation

| Risiko | Impact | Mitigation |
|---------|---------|------------|
| State-Kompatibilität | Hoch | Phase 7: Ausführliche Tests |
| Subagent-Async-Komplexität | Mittel | Start mit simplen Sync-Adjustments, dann async |
| MemorySaver Datenverlust | Niedrig | Dokumentiert, später PostgreSQL |
| Parallel-Migration Konflikte | Mittel | Trennung durch `enabled` Flag |
| Performance-Einbußen | Niedrig | Benchmarking in Phase 7 |
| --basedir Kompatibilität | Mittel | Konfiguration-Pfad anpassen |

---

## 6. Nächste Schritte

Implementierung in Phasen:

1. ✅ Plan in `prompts/01-langraph.md` gespeichert
2. ⏳ Phase 1: Grundgerüst & State implementieren
3. ⏳ Phase 2: Main Graph implementieren
4. ⏳ Phase 3: Subagent Hybrid-Manager
5. ⏳ Phase 4: Tool-Adapter
6. ⏳ Phase 5: Message Bus Adapter
7. ⏳ Phase 6: CLI-Integration (--basedir Option)
8. ⏳ Phase 7: Migration & Testing
9. ⏳ Phase 8: Dokumentation
