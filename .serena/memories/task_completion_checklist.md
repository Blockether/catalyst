# Task Completion Checklist

When completing any development task, follow this exact order:

## 1. Code Quality Checks (REQUIRED)
```bash
# Always run in this order:
poe format      # Format code first
poe verify      # Then lint and typecheck  
```

## 2. Testing (REQUIRED)  
```bash
poe test        # Run all tests
# OR for coverage check:
poe test-cov-check  # Ensures >=85% coverage
```

## 3. Full Verification (RECOMMENDED)
```bash
poe check       # Runs format + verify + test-cov-check + check-docs
```

## Error Resolution Priority
1. **Formatting issues first** (line length, imports, etc.)
2. **Type errors second** (add type hints, None checks, function signatures)
3. **Linting issues last** (unused imports, etc.)

## Common Fixes
- **Line length**: Break strings with parentheses, multi-line function calls, split imports
- **Type errors**: Add None checks, narrow string types, match existing patterns
- **Missing dependencies**: Check package.json/pyproject.toml for available libraries

## Pre-commit Requirements
- Check git status before commits
- Ensure all linting/typing passes
- Keep changes minimal and focused
- Document public APIs
- Test thoroughly, especially edge cases and error conditions

## NEVER commit changes unless explicitly asked
- Only commit when user explicitly requests it
- Run all quality checks before committing