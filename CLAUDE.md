# Development Guidelines

This document contains critical information about working with this codebase. Follow these guidelines precisely.

## Core Development Rules

1. Package Management
   - ONLY use uv, NEVER pip
   - Installation: `uv add package`
   - Running tools: `uv run tool`
   - Running Examples: `uv run python3 examples/*`
   - Running Verification Scripts: `uv run python3 verification/*`
   - Upgrading: `uv add --dev package --upgrade-package package`
   - FORBIDDEN: `uv pip install`, `@latest` syntax

2. Code Quality
   - Type hints required for all code
   - Public APIs must have docstrings
   - Functions must be focused and small
   - Follow existing patterns exactly
   - Line length: 120 chars maximum
   - Every module should have the files ending with `Core` and `internal` folder,
   - MAKE ALL properties in class private by prepending `_` to variable name,
   - AVOID MAGIC NUMBERS! Instead create STATIC fields in class,
   - Avoid `Any` type. Prefer typed classed which inherits from Pydantic `BaseModel` over `Any` type..
   - IF THE PROPERTY should be public then hide it using `_` and create a property function!,
   - Every test file should end in `Test` postfix. In the test file we should use classes always!!
   - using `hasattr` is forbidden by default and same as `getattr`
   - use static type instead of dictionary whenever it's possible and it makes sense
   -
3. Testing Requirements
   - Framework: `poe test`
   - Async testing: use `anyio` over `asyncio`
   - Coverage: test edge cases and errors
   - New features require tests
   - Bug fixes require regression tests
   - Every implementation file MUST HAVE ONLY ONE TEST FILE,
   - NO WEAK TESTS! TESTS SHOULD:
     - Not have any `if` statements,
     - Should test real values and not only shape,
     - Should not have `try`/`catch` like testing to prevent false positives!
     - MUST have hardcoded values mostly and not ranges like `len(expression) > than_magic_number`
     - NO MAGIC NUMBERS! Instead put these numbers in class!

4. Verification Scripts
   - ONLY use Python scripts in the verification folder
   - NO shell scripts (.sh files) in verification
   - All deployment and verification tasks must use Python
   - Shell scripts forbidden in verification directory

## Python Tools

## Code Formatting

1. Ruff
   - Tool: `poe format`

2. Type Checking & linting
   - Tool: `poe verify`

## Error Resolution

1. CI Failures
   - Fix order:
     1. Formatting
     2. Type errors
     3. Linting
   - Type errors:
     - Get full line context
     - Check Optional types
     - Add type narrowing
     - Verify function signatures

2. Common Issues
   - Line length:
     - Break strings with parentheses
     - Multi-line function calls
     - Split imports
   - Types:
     - Add None checks
     - Narrow string types
     - Match existing patterns
   - Pytest:
     - If the tests aren't finding the asyncio pytest mark, try adding PYTEST_DISABLE_PLUGIN_AUTOLOAD=""
       to the start of the pytest run command eg:
       `PYTEST_DISABLE_PLUGIN_AUTOLOAD="" uv run --frozen pytest`

3. Best Practices
   - Check git status before commits
   - Run formatters before type checks
   - Keep changes minimal
   - Follow existing patterns
   - Document public APIs
   - Test thoroughly

## Exception Handling

- **Always use `logger.exception()` instead of `logger.error()` when catching exceptions**
  - Don't include the exception in the message: `logger.exception("Failed")` not `logger.exception(f"Failed: {e}")`
- **Catch specific exceptions** where possible:
  - File ops: `except (OSError, PermissionError):`
  - JSON: `except json.JSONDecodeError:`
  - Network: `except (ConnectionError, TimeoutError):`
- **Only catch `Exception` for**:
  - Top-level handlers that must not crash
  - Cleanup blocks (log at debug level)
