"""Backend abstraction for nanobot.

Provides factory methods for creating backends:
- Local (FilesystemBackend): Run commands locally
- Modal: Run commands in Modal sandbox
- Daytona: Run commands in Daytona sandbox
- Runloop: Run commands in Runloop sandbox
"""

from nanobot.backends.factory import (
    BackendType,
    create_backend,
    get_available_backends,
)

__all__ = [
    "BackendType",
    "create_backend",
    "get_available_backends",
]
