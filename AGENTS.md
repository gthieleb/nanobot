# Agent Instructions

## Testing

Run tests using the nanobot virtualenv:

```bash
# Run all unit tests
~/.virtualenvs/nanobot/bin/pytest tests/ -v --ignore=tests/test_matrix_channel.py --ignore=tests/test_task_delegation.py --ignore=tests/test_task_delegation_config.py --ignore=tests/test_deepagent_delegation.py --ignore=tests/e2e

# Run specific test file
~/.virtualenvs/nanobot/bin/pytest tests/test_commands.py -v

# Run with coverage
~/.virtualenvs/nanobot/bin/pytest tests/ --cov=nanobot --cov-report=term-missing
```

Install dev dependencies if needed:
```bash
~/.virtualenvs/nanobot/bin/pip install -e ".[dev]"
```

## Linting

```bash
~/.virtualenvs/nanobot/bin/ruff check nanobot/
~/.virtualenvs/nanobot/bin/ruff format nanobot/
```
