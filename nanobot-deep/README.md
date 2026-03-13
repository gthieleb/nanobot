# nanobot-deep

LangGraph/DeepAgents integration for nanobot.

## Installation

```bash
pip install nanobot-deep
```

Or with uv:

```bash
uv tool install nanobot-deep
```

## Usage

```bash
# Show deepagents status
nanobot-deep deep

# Create default config
nanobot-deep config

# All nanobot commands are available
nanobot-deep onboard
nanobot-deep chat
```

## What's Included

This package provides **glue code** that integrates:

- **nanobot-ai**: Multi-channel AI assistant framework
- **deepagents**: LangGraph-based agent framework
- **LangGraph**: State management and checkpointing

### Key Components

| Module | Purpose |
|--------|---------|
| `DeepAgent` | Wrapper around `deepagents.create_deep_agent()` |
| `LangGraphBridge` | Translates messages between nanobot and LangGraph |
| `SessionCheckpointer` | SQLite-based session persistence |
| `DeepAgentsConfig` | Configuration schema for deepagents.json |

## Configuration

Create `~/.nanobot/deepagents.json`:

```json
{
  "recursion_limit": 500,
  "skills": ["~/.nanobot/workspace/skills"],
  "subagents": [
    {
      "name": "reply-handler",
      "description": "Handles replies to messages"
    }
  ],
  "interrupt_on": {
    "edit_file": false,
    "execute": false
  }
}
```

## Dependencies

- `nanobot-ai>=0.1.4`
- `deepagents>=0.4.0`
- `langgraph>=0.2.0`
- `langchain-core>=0.3.0`

## License

MIT
