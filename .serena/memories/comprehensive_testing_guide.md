# Comprehensive Testing Guide - MANDATORY RULES

## 🚫 **ABSOLUTELY FORBIDDEN IN TESTS:**

### 1. **NO CONDITIONAL LOGIC**
- **NO `if` statements in test methods**
- **NO `try/except` blocks in tests**
- Tests must be deterministic and test specific conditions
- Catching exceptions hides failures and creates false positives

BAD:
```python
if result.success:
    assert result.value == 42
    
try:
    result = function()
    assert result == expected
except:
    pass  # This hides failures!
```

GOOD:
```python
assert result.success is True
assert result.value == 42

result = function()
assert result == expected
```

### 2. **NO WEAK ASSERTIONS**
- No assertions like `len(x) > 0` or `value >= something`
- Use exact values and strong equality checks
- Test real values, not just shapes/types

BAD:
```python
assert len(results) > 0
assert score >= 0.5
assert isinstance(result, dict)
assert "key" in result
```

GOOD:
```python
assert len(results) == 3
assert score == 0.75
assert result == {"key": "expected_value", "count": 42}
```

### 3. **NO MAGIC NUMBERS**
- Define constants with meaningful names
- Use class-level constants for repeated values

BAD:
```python
assert timeout == 30
```

GOOD:
```python
DEFAULT_TIMEOUT_SECONDS = 30
assert timeout == DEFAULT_TIMEOUT_SECONDS
```

### 4. **NO MOCKS IN INTEGRATION TESTS**
- **FORBIDDEN**: `from unittest.mock import Mock, MagicMock, patch, AsyncMock`
- Integration tests must validate real system behavior
- Use real LLM calls, real API endpoints, real components

### 5. **NO BUSINESS LOGIC IN TEST IMPLEMENTATIONS**
- No complex scoring logic, decision trees in test classes
- Keep test helper classes simple and stupid
- Return hardcoded responses only

BAD:
```python
# Complex logic in test implementation
if type == "acronym" and term_upper in KNOWN_ACRONYMS:
    return complex_response()
elif score > 0.3 and has_numbering:
    return different_response()
```

GOOD:
```python
# Simple, hardcoded response
return {"result": "expected_acronym_value"}
```

## ✅ **REQUIRED IN ALL TESTS:**

### 1. **Use Hardcoded Expected Values**
```python
# BAD: Computing expected value
expected = input_value * 2
assert result == expected

# GOOD: Hardcoded expectation
assert result == 84  # input_value(42) * 2
```

### 2. **Clear Test Names**
```python
# BAD
def test_function():

# GOOD
def test_function_raises_value_error_for_negative_input():
```

### 3. **Exception Testing**
```python
with pytest.raises(ValueError, match="Invalid input"):
    function_that_should_fail()
```

### 4. **One Assertion Per Logical Concept**
- Group related assertions together
- Separate unrelated assertions into different tests

## 📁 **Test Organization:**

### File Naming Convention
- All test files MUST end with `Test.py`
- Test classes should be used for organization
- Example: `ConsensusVotingTest.py`
- Every implementation file MUST HAVE ONLY ONE TEST FILE

### Test Types

#### Integration Tests
- Follow `tests/blockether_catalyst/integrations/conftest.py` pattern
- Use real LLM endpoints with `@pytest.mark.real_llm`
- Test actual end-to-end functionality
- Fail fast if external dependencies unavailable

#### Unit Tests
- Test individual components in isolation
- Use simple, deterministic test doubles if needed
- Focus on testing real implementation code
- No business logic in test helpers

## 🎯 **TESTING FRAMEWORK:**
- Framework: `poe test`
- Async testing: use `anyio` over `asyncio`
- Coverage: test edge cases and errors
- New features require tests
- Bug fixes require regression tests

## 🔥 **REMEMBER:**
**Tests are documentation of expected behavior. They should be:**
- **Deterministic**: Same input → same output every time
- **Specific**: Test one thing at a time
- **Clear**: Anyone can understand what's being tested
- **Strong**: No ambiguity about pass/fail
- **Real**: Tests should validate REAL system behavior, not test artifacts!