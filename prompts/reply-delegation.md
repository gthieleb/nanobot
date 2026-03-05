# Reply-To Delegation & Task Routing

**GitHub Issue:** #8

## Goal

Implement intelligent task delegation for the deepagents backend that:
1. Delegates reply-to messages to subagents with full context
2. Distinguishes between control tasks and delegate tasks
3. Provides configuration for routing behavior

## Context

### Current State
- All messages processed by main agent
- No distinction between control and delegated tasks
- Reply-to context not utilized

### Target State
- Reply-to messages delegated to `reply-handler` subagent
- Control commands (`/help`, `/new`, etc.) handled by main agent
- Complex tasks automatically delegated based on configuration

## Implementation Steps

### Step 1: Channel Reply Detection

**Telegram Channel** (`nanobot/channels/telegram.py`):
```python
async def _handle_message(self, update):
    message = update.message
    reply_to = message.reply_to_message
    
    metadata = {}
    if reply_to:
        metadata["reply_to_message"] = {
            "message_id": reply_to.message_id,
            "text": reply_to.text or "",
            "from": {
                "id": reply_to.from_user.id,
                "username": reply_to.from_user.username,
            },
            "date": reply_to.date.isoformat() if reply_to.date else None,
        }
    
    # ... create InboundMessage with metadata
```

**Discord Channel** (`nanobot/channels/discord.py`):
```python
# Similar pattern for message.reference
if message.reference and message.reference.message_id:
    ref_msg = await message.channel.fetch_message(message.reference.message_id)
    metadata["reply_to_message"] = {...}
```

**Slack Channel** (`nanobot/channels/slack.py`):
```python
# Check for thread_ts or message.thread_ts
if message.thread_ts:
    # Fetch parent message and include context
```

### Step 2: DeepAgent Delegation Logic

**File:** `nanobot/agent/deep_agent.py`

```python
async def process(self, msg: InboundMessage, ...) -> OutboundMessage:
    """Process message with intelligent routing."""
    
    # Check for control commands
    if self._is_control_command(msg.content):
        return await self._handle_control(msg)
    
    # Check for reply-to delegation
    if msg.metadata.get("reply_to_message"):
        if self.dg_config.task_routing.auto_delegate_reply_to:
            return await self._delegate_reply(msg)
    
    # Normal processing
    return await self._process_normal(msg)

def _is_control_command(self, content: str) -> bool:
    """Check if message is a control command."""
    content = content.strip().lower()
    return any(
        content.startswith(cmd.lower()) 
        for cmd in self.dg_config.task_routing.control_commands
    )

async def _delegate_reply(self, msg: InboundMessage) -> OutboundMessage:
    """Delegate reply handling to subagent."""
    
    reply_context = msg.metadata.get("reply_to_message", {})
    history = self.get_history(msg.session_key, limit=10)
    
    context_prompt = self._build_reply_context(reply_context, history)
    
    # Use task tool for delegation
    state = {
        "messages": [
            SystemMessage(content=context_prompt),
            HumanMessage(content=msg.content),
        ]
    }
    
    result = await self.agent.ainvoke(state, {
        "configurable": {"thread_id": msg.session_key},
    })
    
    return translate_result_to_outbound(result, msg)

def _build_reply_context(self, reply_to: dict, history: list) -> str:
    """Build context prompt for reply handler."""
    return f"""## Reply Context

You are replying to a message.

**Original Message:**
{reply_to.get('text', 'N/A')}

From: @{reply_to.get('from', {}).get('username', 'Unknown')}

**Recent Conversation:**
{_format_history(history)}

**Reply Request:**
Respond appropriately to the user's reply.
"""
```

### Step 3: Configuration Schema

**File:** `nanobot/config/deepagents_schema.py`

```python
class DeepAgentsTaskRoutingConfig(BaseConfig):
    """Task routing configuration for delegation."""
    
    auto_delegate_reply_to: bool = True
    control_commands: list[str] = Field(
        default_factory=lambda: ["/help", "/new", "/stop", "/tasks", "/cancel"]
    )
    delegate_threshold_chars: int = 500
    always_delegate_patterns: list[str] = Field(default_factory=list)
    never_delegate_patterns: list[str] = Field(default_factory=list)

class DeepAgentsConfig(BaseConfig):
    # ... existing fields ...
    task_routing: DeepAgentsTaskRoutingConfig = Field(
        default_factory=DeepAgentsTaskRoutingConfig
    )
```

### Step 4: Update Template

**File:** `nanobot/templates/deepagents.json`

```json
{
  "task_routing": {
    "auto_delegate_reply_to": true,
    "control_commands": ["/help", "/new", "/stop", "/tasks", "/cancel"],
    "delegate_threshold_chars": 500,
    "always_delegate_patterns": [],
    "never_delegate_patterns": []
  },
  "subagents": [
    {
      "name": "reply-handler",
      "description": "Handles replies to messages with context",
      "system_prompt": "You are a reply assistant. Consider the original message context...",
      "model": null
    }
  ]
}
```

### Step 5: Tests

**File:** `tests/test_task_routing.py`

```python
def test_is_control_command():
    """Test control command detection."""
    agent = DeepAgent(...)
    
    assert agent._is_control_command("/help") is True
    assert agent._is_control_command("/new") is True
    assert agent._is_control_command("Hello") is False
    assert agent._is_control_command("Please help me") is False

@pytest.mark.asyncio
async def test_delegate_reply():
    """Test reply-to delegation."""
    msg = InboundMessage(
        channel="telegram",
        content="What do you think about this?",
        metadata={
            "reply_to_message": {
                "text": "I just deployed a new feature!",
                "from": {"username": "developer"},
            }
        }
    )
    
    response = await agent.process(msg)
    # Verify delegation occurred
```

## Testing Commands

```bash
# Run unit tests
pytest tests/test_task_routing.py -v

# Run with real Telegram (requires API key)
NANOBOT_TEST_API_KEY=... pytest tests/e2e/test_reply_delegation.py -v

# Test locally
nanobot agent --use-langgraph
# Then send a reply-to message via Telegram
```

## Success Metrics

- Reply-to messages correctly detected in all channels
- Control commands never delegated
- Reply context included in subagent prompt
- Response quality matches or exceeds non-delegated replies

## Rollback Plan

If issues arise:
1. Set `auto_delegate_reply_to: false` in `deepagents.json`
2. All messages processed by main agent
3. No breaking changes to existing behavior

## Notes

- Consider rate limiting for subagent spawning
- Monitor token usage for delegated tasks
- May need to adjust `delegate_threshold_chars` based on usage patterns
