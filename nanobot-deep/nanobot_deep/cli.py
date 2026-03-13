"""CLI entry point for nanobot-deep.

This module provides a CLI that extends nanobot with LangGraph/DeepAgents support.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="nanobot-deep",
    help="Nanobot with LangGraph/DeepAgents integration",
    no_args_is_help=True,
)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Nanobot with LangGraph/DeepAgents integration.

    This CLI wraps nanobot and adds deep-specific commands.
    All nanobot commands are available.
    """
    if ctx.invoked_subcommand is None:
        typer.echo("Use 'nanobot-deep --help' for available commands")


@app.command()
def deep():
    """Run with DeepAgents backend (LangGraph-based agent)."""
    from nanobot_deep import is_deepagents_available

    if not is_deepagents_available():
        typer.secho(
            "Error: deepagents is not installed. Install with: pip install deepagents",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    typer.secho(
        "DeepAgents backend is available. Use 'nanobot-deep run' to start.",
        fg=typer.colors.GREEN,
    )


@app.command()
def config():
    """Show or create default deepagents.json configuration."""
    from pathlib import Path
    import json

    from nanobot_deep import DeepAgentsConfig, save_deepagents_config

    config_path = Path.home() / ".nanobot" / "deepagents.json"

    if config_path.exists():
        typer.secho(f"Config exists at {config_path}", fg=typer.colors.GREEN)
        with open(config_path) as f:
            typer.echo(f.read())
    else:
        typer.secho(f"Creating default config at {config_path}", fg=typer.colors.YELLOW)
        default_config = DeepAgentsConfig()
        save_deepagents_config(default_config, config_path)
        typer.secho("Done!", fg=typer.colors.GREEN)


def run_nanobot():
    """Entry point that delegates to nanobot CLI for non-deep commands."""
    import sys

    try:
        from nanobot.cli.commands import app as nanobot_app
    except ImportError:
        typer.secho(
            "Error: nanobot-ai is not installed. Install with: pip install nanobot-ai",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        if first_arg in ["deep", "config"]:
            app()
            return

    nanobot_app()


if __name__ == "__main__":
    run_nanobot()
