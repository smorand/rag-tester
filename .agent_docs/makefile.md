# Makefile Documentation

The Makefile provides a unified interface for all project operations. **NEVER run `uv run pytest`, `uv run ruff`, etc. directly** - always use `make` commands.

## Essential Commands

### Development
```bash
make sync          # Install/update dependencies (uv sync)
make run           # Run the application
make run-dev       # Run in development mode
```

### Quality Checks
```bash
make check         # Full quality gate (MUST pass before commit)
make lint          # Ruff linting
make lint-fix      # Ruff linting with auto-fix
make format        # Format code with Ruff
make format-check  # Check formatting without changes
make typecheck     # mypy type checking
make security      # bandit security scan
```

### Testing
```bash
make test          # Run tests
make test-cov      # Run tests with coverage report (>= 80% required)
```

### Build & Deploy
```bash
make build         # Build distribution packages
make install       # Install package locally
make uninstall     # Uninstall package
make docker-build  # Build Docker image
make docker-push   # Push Docker image
make docker        # Build and push Docker image
```

### Docker Compose
```bash
make run-up        # Start services with docker-compose
make run-down      # Stop services
```

### Cleanup
```bash
make clean         # Remove build artifacts
make clean-all     # Remove all generated files including venv
```

### Information
```bash
make info          # Show project information
make help          # Show all available commands
```

## Quality Gate (`make check`)

The `check` target is the **full quality gate** that must pass before every commit:

1. **Lint** - Ruff linting (no auto-fix)
2. **Format Check** - Ruff format verification
3. **Type Check** - mypy strict type checking
4. **Security** - bandit security scan
5. **Test Coverage** - pytest with >= 80% coverage

If any step fails, the entire check fails.

## Version Injection

The Makefile automatically injects the version from git tags:
- `VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")`
- Used in `make build` and `make docker-build`
- Written to `src/rag_tester/version.py` before building

## Customization

The template Makefile **MUST NOT** be modified for Python targets. Custom non-Python targets may be added at the end of the file if needed for project-specific operations.
