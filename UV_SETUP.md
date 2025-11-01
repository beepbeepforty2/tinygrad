# UV Project Setup for tinygrad

This project is now configured as a UV project with `pyproject.toml`.

## Quick Commands

### Activate the virtual environment
```bash
source .venv/bin/activate
```

### Run Python commands with uv
```bash
uv run python script.py
uv run pytest
```

### Install/sync dependencies
```bash
# Sync all dependencies from pyproject.toml
uv sync

# Install additional package
uv pip install <package>

# Install with extras (e.g., testing, linting, docs)
uv sync --extra testing
uv sync --extra linting
uv sync --extra docs
```

### List installed packages
```bash
uv pip list
```

### Update dependencies
```bash
uv sync --upgrade
```

## Project Structure

- `pyproject.toml` - Project configuration with dependencies
- `.venv/` - Virtual environment (managed by uv)
- `uv.lock` - Locked dependency versions (auto-generated)

## Available Extras

- `testing_minimal` - Basic testing dependencies
- `testing_unit` - Unit testing with common tools
- `testing` - Full testing suite
- `linting` - Code quality tools
- `docs` - Documentation generation tools
- `dev` - Common development dependencies (default)
- `arm` - ARM-specific dependencies
