# No Any Type or Casts to Any Policy

## Strict Rule: ANY Type is PROHIBITED

### Policy Statement
**ABSOLUTELY NO CASTS TO ANY TYPE ARE ALLOWED IN THIS CODEBASE**

### Key Requirements

1. **Type Safety First**
   - NEVER use `Any` type in function signatures, return types, or variable annotations
   - NEVER cast to `Any` type using `cast(Any, ...)`
   - NEVER use `# type: ignore` to bypass type checking
   - NEVER use `typing.Any` as a fallback when types are unclear

2. **Required Practices**
   - ALWAYS use specific, typed Pydantic models instead of `Any`
   - ALWAYS define proper type hints for all functions and methods
   - ALWAYS use Union types or Optional when multiple types are possible
   - ALWAYS create typed dataclasses or Pydantic models for complex data structures

3. **Common Violations to Avoid**
   ```python
   # PROHIBITED - Never do this:
   def process_data(data: Any) -> Any:  # NO!
       return cast(Any, result)  # NO!
   
   # PROHIBITED - Never do this:
   result: Any = some_function()  # NO!
   
   # PROHIBITED - Never do this:
   from typing import Any  # Should not be imported!
   ```

4. **Correct Alternatives**
   ```python
   # CORRECT - Always do this:
   def process_data(data: SpecificModel) -> ProcessedResult:
       return ProcessedResult(...)
   
   # CORRECT - Use Union for multiple types:
   def handle_input(data: Union[str, int, float]) -> str:
       return str(data)
   
   # CORRECT - Create proper models:
   class DataModel(BaseModel):
       field1: str
       field2: int
   ```

5. **Exception Handling**
   - If encountering external library code that returns Any, immediately wrap it in a proper type
   - If absolutely stuck, ask for help rather than using Any
   - Document why a specific type was chosen over Any

6. **Code Review Checklist**
   - [ ] No imports of `Any` from typing
   - [ ] No `cast(Any, ...)` statements
   - [ ] No `: Any` type annotations
   - [ ] All functions have proper return type hints
   - [ ] All parameters have specific type hints
   - [ ] Complex data uses Pydantic models or dataclasses

## Enforcement
This rule is NON-NEGOTIABLE. Any code containing `Any` type or casts to `Any` must be immediately refactored to use proper typing.

## Rationale
- Type safety prevents runtime errors
- Proper types serve as documentation
- IDE autocomplete works correctly with proper types
- Easier to refactor and maintain typed code
- Reduces bugs and improves code quality