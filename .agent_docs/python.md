# Python Coding Standards

This project follows the Python skill standards defined in `~/.bob/skills/python/SKILL.md`.

## Key Points

- **Python Version:** 3.13+
- **Build System:** hatchling
- **Linting/Formatting:** Ruff (replaces Black, isort, Flake8)
- **Type Checking:** mypy in strict mode
- **Testing:** pytest with pytest-asyncio, >= 80% coverage required
- **CLI Framework:** Typer (NEVER argparse or click)
- **Config:** pydantic-settings (NEVER os.environ directly)
- **Async-First:** All I/O operations use async libraries

## Project Layout

This project uses **Layout B (package)** from the skill:
- Package: `src/rag_tester/`
- pyproject.toml: `packages = ["src/rag_tester"]`
- Script: `rag-tester = "rag_tester:app"`
- Imports: `from rag_tester.module import ...`

## Required Files

Every project must have:
- `config.py` - Settings with pydantic-settings
- `logging_config.py` - Logging setup (rich + file output)
- `tracing.py` - OpenTelemetry tracing
- `version.py` - Version info (injected at build time)
- `py.typed` - PEP 561 marker for type checking

## Quality Gate

`make check` runs:
1. `make lint` - Ruff linting with auto-fix disabled
2. `make format-check` - Ruff format verification
3. `make typecheck` - mypy strict type checking
4. `make security` - bandit security scan
5. `make test-cov` - pytest with coverage >= 80%

All must pass before committing.

## Common Patterns

### Logging
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Message with %s formatting", value)  # Use % not f-strings
```

### Tracing
```python
from rag_tester.tracing import trace_span

with trace_span("category.operation", attributes={"key": "value"}):
    # traced code
```

### Config
```python
from rag_tester.config import Settings
settings = Settings()  # Loads from env vars with RAG_TESTER_ prefix
```

## Forbidden Practices

- ❌ `print()` - Use `logger.debug()` or `typer.echo()`
- ❌ Bare `except:` - Always specify error type
- ❌ `type: ignore` without comment
- ❌ Mutable default arguments
- ❌ `.format()` - Use f-strings
- ❌ `assert` in production code
- ❌ Sync I/O in async code (requests, open, subprocess.run)
