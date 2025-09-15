# Development Workflow

## Package Management - CRITICAL RULES

### ✅ ALWAYS Use uv
```bash
uv add package                    # Add new dependency
uv add --dev package             # Add dev dependency  
uv run tool                      # Run any command
uv run python3 script.py         # Run Python scripts
```

### ❌ FORBIDDEN Commands
```bash
uv pip install                   # NEVER use this
pip install                      # NEVER use this
uv add package@latest            # NEVER use @latest syntax
```

## Development Process

### 1. Initial Setup
```bash
# Clone and install
git clone <repo>
uv sync                          # Install dependencies
```

### 2. Daily Development Cycle
```bash
# Before starting work
poe format                       # Format code
poe verify                       # Check types and linting

# After making changes
poe test                         # Run tests
poe check                        # Full verification
```

### 3. Adding New Features
1. **Check existing libraries first** - look at neighboring files, pyproject.toml
2. **Follow existing patterns** - examine similar components
3. **Use proper naming conventions** - `{Module}Core.py`, `{Module}Test.py`
4. **Add comprehensive tests** - one test file per implementation file

### 4. Error Resolution Order
1. **Format first**: `poe format`
2. **Fix type errors**: Add type hints, None checks
3. **Fix linting**: Remove unused imports, etc.

## Testing Guidelines

### Test Structure
- Use classes for all tests
- Test file names end with `Test`
- One test file per implementation file
- Use anyio for async tests (not asyncio)

### Test Quality (NO WEAK TESTS!)
- No `if` statements in tests
- Test actual values, not just shapes
- No try/catch for testing
- Use hardcoded expected values
- No magic numbers - use class constants

### Running Tests
```bash
poe test                         # All tests
poe test-cov                     # With coverage report
poe test-cov-check              # Enforce 85% coverage
```

## Git Workflow
- Check `git status` before commits
- Never commit unless explicitly asked
- Run `poe check` before any commits
- Keep changes focused and minimal

## Performance Expectations
- Initialization: < 0.5 seconds
- Search operations: < 0.5 seconds  
- Pickle load/save: < 0.5 seconds