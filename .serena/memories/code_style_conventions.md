# Code Style and Conventions

## Core Development Rules

### Code Quality Requirements
- **Type hints required** for all code
- **Public APIs must have docstrings**
- **Functions must be focused and small**
- **Follow existing patterns exactly**
- **Line length: 120 chars maximum**
- **Avoid `Any` type** - prefer typed classes inheriting from Pydantic `BaseModel`
- **WHEN you are spoting the obvious error YOU SHOULD ALWAYS correct it without proposing the FALLBACK (NO FALLBACK POLICY)**

### File Organization
- Every module should have files ending with `Core` and `internal` folder
- **MAKE ALL properties in class private** by prepending `_` to variable name
- **IF property should be public** then hide it using `_` and create a property function
- **AVOID MAGIC NUMBERS** - instead create STATIC fields in class

### Comment Style
- **PROHIBITED: Meaningless inline comments (e.g., `x = 1  # Set x to 1`). Comments must add value, not state the obvious. NO COMMENTS on implementation side unless they explain complex logic.**
- **NO inline comments after code** - Never write comments on the same line as code
- Comments should be on their own line above the code they describe
- Docstrings are preferred over comments for function/class documentation

### Class Design
- **Classes with only static methods should not have instances created**
  - Mark such classes appropriately (e.g., with abstract base class or clear documentation)
  - Consider using module-level functions instead if appropriate

### Forbidden Practices
- Using `hasattr` and `getattr` by default
- Use static types instead of dictionaries whenever possible and sensible

### Frontend and Templating Standards
- **Templating Engine**: Always use **Jinja2 (.j2)** templates as the goto solution
- **CSS Framework**: Always use **TailwindCSS** for styling
- **Interactive Components**: Always use **HTMX** for dynamic frontend behavior
- **Template Organization**:
  - Base templates should provide common structure
  - Use template inheritance for consistency
  - Partial templates for reusable components
- **No custom CSS** unless absolutely necessary - leverage TailwindCSS utility classes
- **Prefer HTMX attributes** over custom JavaScript for interactivity

### Testing Requirements
- **Framework**: pytest with anyio (preferred over asyncio)
- **Test file naming**: Must end in `Test` postfix (e.g., `KnowledgeSearchCoreTest.py`)
- **Test classes**: Always use classes in test files
- **One test file per implementation file**

### Test Quality Standards - NO WEAK TESTS!
Tests MUST:
- **Not have any `if` statements**
- **Test real values and not only shape**
- **Not use `try`/`catch` for testing** (prevents false positives)
- **Have hardcoded values mostly** and not ranges like `len(expression) > magic_number`
- **NO MAGIC NUMBERS** - put numbers in class constants

### Exception Handling
- **Always use `logger.exception()` instead of `logger.error()` when catching exceptions**
- Don't include the exception in the message: `logger.exception("Failed")` not `logger.exception(f"Failed: {e}")`
- **Catch specific exceptions** where possible:
  - File ops: `except (OSError, PermissionError):`
  - JSON: `except json.JSONDecodeError:`
  - Network: `except (ConnectionError, TimeoutError):`
- **Only catch `Exception` for**:
  - Top-level handlers that must not crash
  - Cleanup blocks (log at debug level)

### Formatting Tools
- **Ruff**: Primary linter and formatter
- **Black**: Code formatting (line-length=120, target py312)
- **isort**: Import sorting (black profile, line_length=120)
- **MyPy**: Type checking with strict settings
