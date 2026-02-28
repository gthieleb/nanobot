"""Backend factory for creating sandbox backends.

Provides a unified interface for creating different backend types:
- local: Run commands locally (default)
- modal: Run commands in Modal sandbox
- daytona: Run commands in Daytona sandbox
- runloop: Run commands in Runloop sandbox
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Callable

from loguru import logger


class BackendType(str, Enum):
    """Available backend types."""

    LOCAL = "local"
    MODAL = "modal"
    DAYTONA = "daytona"
    RUNLOOP = "runloop"
    STATE = "state"


def get_available_backends() -> list[BackendType]:
    """Get list of available backend types.

    Returns:
        List of backend types that can be created
    """
    available = [BackendType.LOCAL, BackendType.STATE]

    try:
        import langchain_modal

        available.append(BackendType.MODAL)
    except ImportError:
        pass

    try:
        import langchain_daytona

        available.append(BackendType.DAYTONA)
    except ImportError:
        pass

    try:
        import langchain_runloop

        available.append(BackendType.RUNLOOP)
    except ImportError:
        pass

    return available


def create_backend(
    backend_type: BackendType | str = BackendType.LOCAL,
    workspace: Path | None = None,
    **kwargs: Any,
) -> Any:
    """Create a backend instance.

    Args:
        backend_type: Type of backend to create
        workspace: Workspace directory for file operations
        **kwargs: Additional arguments passed to backend constructor

    Returns:
        Backend instance

    Raises:
        ImportError: If required backend package is not installed
        ValueError: If backend type is not supported
    """
    if isinstance(backend_type, str):
        backend_type = BackendType(backend_type)

    workspace = workspace or Path.home() / ".nanobot" / "workspace"

    if backend_type == BackendType.LOCAL:
        return _create_local_backend(workspace, **kwargs)
    elif backend_type == BackendType.STATE:
        return _create_state_backend(**kwargs)
    elif backend_type == BackendType.MODAL:
        return _create_modal_backend(workspace, **kwargs)
    elif backend_type == BackendType.DAYTONA:
        return _create_daytona_backend(workspace, **kwargs)
    elif backend_type == BackendType.RUNLOOP:
        return _create_runloop_backend(workspace, **kwargs)
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")


def _create_local_backend(workspace: Path, **kwargs: Any) -> Any:
    """Create a local filesystem backend."""
    try:
        from deepagents.backends import FilesystemBackend

        return FilesystemBackend(root_dir=workspace)
    except ImportError:
        logger.warning("deepagents not installed, returning None")
        return None


def _create_state_backend(**kwargs: Any) -> Any:
    """Create a state backend (ephemeral, in-memory)."""
    try:
        from deepagents.backends import StateBackend

        return lambda runtime: StateBackend(runtime)
    except ImportError:
        logger.warning("deepagents not installed, returning None")
        return None


def _create_modal_backend(workspace: Path, **kwargs: Any) -> Any:
    """Create a Modal sandbox backend."""
    try:
        from langchain_modal import ModalSandboxBackend
    except ImportError as e:
        raise ImportError(
            "Modal backend requires langchain-modal. Install with: pip install nanobot-ai[sandbox]"
        ) from e

    sandbox_id = kwargs.get("sandbox_id")
    setup_script = kwargs.get("setup_script")

    return ModalSandboxBackend(
        sandbox_id=sandbox_id,
        setup_script=setup_script,
    )


def _create_daytona_backend(workspace: Path, **kwargs: Any) -> Any:
    """Create a Daytona sandbox backend."""
    try:
        from langchain_daytona import DaytonaSandboxBackend
    except ImportError as e:
        raise ImportError(
            "Daytona backend requires langchain-daytona. "
            "Install with: pip install nanobot-ai[sandbox]"
        ) from e

    sandbox_id = kwargs.get("sandbox_id")
    setup_script = kwargs.get("setup_script")

    return DaytonaSandboxBackend(
        sandbox_id=sandbox_id,
        setup_script=setup_script,
    )


def _create_runloop_backend(workspace: Path, **kwargs: Any) -> Any:
    """Create a Runloop sandbox backend."""
    try:
        from langchain_runloop import RunloopSandboxBackend
    except ImportError as e:
        raise ImportError(
            "Runloop backend requires langchain-runloop. "
            "Install with: pip install nanobot-ai[sandbox]"
        ) from e

    sandbox_id = kwargs.get("sandbox_id")
    setup_script = kwargs.get("setup_script")

    return RunloopSandboxBackend(
        sandbox_id=sandbox_id,
        setup_script=setup_script,
    )
