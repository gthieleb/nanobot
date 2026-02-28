"""Helper functions for LangGraph-based Ralph mode."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from rich.console import Console
from loguru import logger

console = Console()


async def run_ralph_mode(
    task: str,
    max_iterations: int = 0,
    workspace: Path | None = None,
    model: str | None = None,
    sandbox: str = "none",
    stream: bool = True,
) -> None:
    """Run agent in autonomous Ralph mode using deepagents.

    Args:
        task: Declarative description of what to build
        max_iterations: Maximum iterations (0 = unlimited)
        workspace: Working directory
        model: Model specification (e.g., 'anthropic:claude-sonnet-4-5')
        sandbox: Sandbox provider ('none', 'modal', 'daytona', 'runloop')
        stream: Whether to stream output
    """
    workspace = workspace or Path.cwd()

    try:
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
        from deepagents.backends.protocol import BackendProtocol, BackendFactory
    except ImportError as e:
        raise ImportError(
            "deepagents is required for Ralph mode. Install with: pip install deepagents"
        ) from e

    backend = _create_backend(sandbox, workspace)
    agent = create_deep_agent(
        model=model or "anthropic:claude-sonnet-4-5",
        backend=backend,
    )

    iteration = 1
    try:
        while max_iterations == 0 or iteration <= max_iterations:
            separator = "=" * 60
            console.print(f"\n[bold cyan]{separator}[/bold cyan]")
            console.print(f"[bold cyan]RALPH ITERATION {iteration}[/bold cyan]")
            console.print(f"[bold cyan]{separator}[/bold cyan]\n")

            iter_display = f"{iteration}/{max_iterations}" if max_iterations > 0 else str(iteration)
            prompt = (
                f"## Ralph Iteration {iter_display}\n\n"
                f"Your previous work is in the filesystem. "
                f"Check what exists and keep building.\n\n"
                f"TASK:\n{task}\n\n"
                f"Make progress. You'll be called again."
            )

            state = {"messages": [{"role": "user", "content": prompt}]}
            config = {
                "configurable": {
                    "thread_id": f"ralph-{iteration}",
                },
            }

            if stream:
                async for event in agent.astream_events(state, config, version="v2"):
                    if event.get("event") == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            console.print(str(chunk.content), end="")
                    elif event.get("event") == "on_tool_start":
                        tool_name = event.get("name", "unknown")
                        console.print(f"\n[dim]→ {tool_name}()[/dim]")
                console.print()
            else:
                result = await agent.ainvoke(state, config)
                messages = result.get("messages", [])
                if messages:
                    last = messages[-1]
                    if hasattr(last, "content"):
                        console.print(str(last.content))

            console.print(f"\n[dim]...continuing to iteration {iteration + 1}[/dim]")
            iteration += 1

    except KeyboardInterrupt:
        console.print(f"\n[bold yellow]Stopped after {iteration} iterations[/bold yellow]")

    console.print(f"\n[bold]Files in {workspace}:[/bold]")
    for path in sorted(workspace.rglob("*")):
        if path.is_file() and ".git" not in str(path):
            console.print(f"  {path.relative_to(workspace)}", style="dim")


def _create_backend(sandbox: str, workspace: Path) -> BackendProtocol | BackendFactory:
    """Create backend based on sandbox type."""
    if sandbox == "none":
        from deepagents.backends import FilesystemBackend

        return FilesystemBackend(root_dir=workspace)

    elif sandbox == "modal":
        try:
            from langchain_modal import ModalSandboxBackend

            return ModalSandboxBackend()
        except ImportError:
            raise ImportError(
                "Modal sandbox requires langchain-modal. "
                "Install with: pip install nanobot-ai[sandbox]"
            )

    elif sandbox == "daytona":
        try:
            from langchain_daytona import DaytonaSandboxBackend

            return DaytonaSandboxBackend()
        except ImportError:
            raise ImportError(
                "Daytona sandbox requires langchain-daytona. "
                "Install with: pip install nanobot-ai[sandbox]"
            )

    elif sandbox == "runloop":
        try:
            from langchain_runloop import RunloopSandboxBackend

            return RunloopSandboxBackend()
        except ImportError:
            raise ImportError(
                "Runloop sandbox requires langchain-runloop. "
                "Install with: pip install nanobot-ai[sandbox]"
            )

    else:
        raise ValueError(f"Unknown sandbox type: {sandbox}")
