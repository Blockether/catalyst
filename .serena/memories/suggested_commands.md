# Essential Development Commands

## Package Management (ALWAYS use uv)
```bash
# Installation
uv add package

# Running tools  
uv run tool

# Running examples
uv run python3 examples/*

# Running verification scripts
uv run python3 verification/*

# Upgrading packages
uv add --dev package --upgrade-package package

# FORBIDDEN: uv pip install, @latest syntax
```

## Code Quality & Testing
```bash
# Linting and type checking
poe verify          # Runs both lint and typecheck
poe lint           # uv run ruff check src/ tests/
poe typecheck      # uv run mypy src/ tests/

# Formatting
poe format         # Runs ruff, black, and isort formatters
poe format-ruff    # uv run ruff format src/ tests/
poe format-black   # uv run black src/ tests/
poe format-isort   # uv run isort src/ tests/

# Testing
poe test           # uv run python3 -m pytest
poe test-cov       # uv run python3 -m pytest --cov=src --cov-report=html
poe test-cov-check # uv run python3 -m pytest --cov=src --cov-report=term-missing --cov-fail-under=85

# Complete workflow
poe check          # format + verify + test-cov-check + check-docs
```

## Documentation
```bash
poe docs-serve     # uv run mkdocs serve
poe docs-build     # uv run mkdocs build  
poe check-docs     # Validates documentation structure
```

## Cleaning
```bash
poe clean          # Cleans pyc files and cache
poe clean-cache    # uv cache clean
poe clean-pyc      # Removes __pycache__ and .pyc files
```

## System Commands (macOS Darwin)
```bash
# Standard Unix commands available:
ls, cd, grep, find, git, etc.
```

## Important Notes
- ALWAYS run `poe format` before `poe verify`
- If pytest fails with asyncio marks, try: `PYTEST_DISABLE_PLUGIN_AUTOLOAD="" uv run --frozen pytest`
- NO shell scripts (.sh) allowed in verification/ directory - use Python scripts only