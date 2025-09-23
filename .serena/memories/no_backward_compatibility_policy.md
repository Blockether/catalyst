# No Backward Compatibility Policy

## Key Principle
This codebase follows a STRICT policy of NO BACKWARD COMPATIBILITY tricks or workarounds.

## Guidelines
1. **NEVER** add backward compatibility fields or methods
2. **NEVER** keep deprecated fields "for compatibility"
3. **NEVER** add fallback logic for old API patterns
4. When changing APIs, make clean breaks - update all references immediately
5. If something needs to change, change it completely and update all affected code

## Why This Matters
- Clean, maintainable code without legacy cruft
- Clear API contracts without ambiguity
- Forces proper updates instead of technical debt accumulation
- Prevents confusion about which fields/methods should be used

## Example of What NOT to Do
```python
# BAD - Never do this!
class Config:
    new_field: str
    old_field: str = None  # Keep for backward compatibility
```

## Example of What TO Do
```python
# GOOD - Clean break
class Config:
    new_field: str
    # Update all references to use new_field
```

Remember: If changing an API, update ALL references immediately. No compatibility layers!