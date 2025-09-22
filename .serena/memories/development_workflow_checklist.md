# Development Workflow Checklist

## 🔧 **Code Quality Pipeline (REQUIRED ORDER):**

### 1. Format First
```bash
poe format      # Always run formatting first
```

### 2. Verify Second  
```bash
poe verify      # Then lint and typecheck
```

### 3. Test Last
```bash
poe test        # Run all tests
# OR for coverage check:
poe test-cov-check  # Ensures >=85% coverage
```

### 4. Full Pipeline (RECOMMENDED)
```bash
poe check       # Runs format + verify + test-cov-check + check-docs
```

## 🚨 **Error Resolution Priority:**
1. **Formatting issues first** (line length, imports, etc.)
2. **Type errors second** (add type hints, None checks, function signatures)  
3. **Linting issues last** (unused imports, etc.)

## 🛠 **Common Fixes:**
- **Line length**: Break strings with parentheses, multi-line function calls, split imports
- **Type errors**: Add None checks, narrow string types, match existing patterns
- **Missing dependencies**: Check pyproject.toml for available libraries
- **Pytest issues**: Try adding `PYTEST_DISABLE_PLUGIN_AUTOLOAD=""` to pytest run command

## 📦 **Package Management (CRITICAL):**
- **ONLY use uv, NEVER pip**
- Installation: `uv add package`
- Running tools: `uv run tool`
- Running Examples: `uv run python3 examples/*`
- Running Verification Scripts: `uv run python3 verification/*`
- Upgrading: `uv add --dev package --upgrade-package package`
- **FORBIDDEN**: `uv pip install`, `@latest` syntax

## ✅ **Pre-commit Requirements:**
- Check git status before commits
- Ensure all linting/typing passes
- Keep changes minimal and focused
- Document public APIs
- Test thoroughly, especially edge cases and error conditions

## 🚫 **NEVER commit changes unless explicitly asked**
- Only commit when user explicitly requests it
- Run all quality checks before committing