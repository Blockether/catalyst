# OpenAI JSON Schema Restrictions for Structured Output

## Critical Rule: No Descriptions on $ref Fields

### The Problem
When using OpenAI's structured output (response_format), the API enforces strict JSON Schema validation. Specifically:

**Fields that reference other Pydantic models CANNOT have descriptions at the reference level.**

### Error Example
```
Invalid schema for response_format 'AnswerOutput': 
context=('properties', 'contradiction_presence'), 
$ref cannot have keywords {'description'}
```

### What This Means

#### ❌ PROHIBITED - Will cause 400 Bad Request:
```python
class EvaluationFactor(BaseModel):
    score: float
    reasoning: str

class AnswerOutput(BaseModel):
    # This will FAIL with OpenAI API
    contradiction_presence: EvaluationFactor = VotingField(
        description="This description causes the error!"  # ❌ NOT ALLOWED
    )
```

#### ✅ CORRECT - No description on model references:
```python
class AnswerOutput(BaseModel):
    # No description when field type is another model
    contradiction_presence: EvaluationFactor = VotingField(
        # NO description parameter here
        comparison=ComparisonStrategy.DERIVED,
        threshold=0.8,
    )
```

#### ✅ OK - Descriptions allowed on primitive types:
```python
class AnswerOutput(BaseModel):
    # Primitive types CAN have descriptions
    score: float = VotingField(
        description="Score between 0-1",  # ✅ This is fine
        comparison=ComparisonStrategy.RANGE,
    )
    
    name: str = VotingField(
        description="The name",  # ✅ This is fine
        comparison=ComparisonStrategy.EXACT,
    )
```

### Rules Summary

1. **Primitive Types** (str, int, float, bool, etc.): 
   - ✅ CAN have descriptions

2. **Model References** (BaseModel, RootModel, custom classes):
   - ❌ CANNOT have descriptions at the field level
   - Put descriptions in the model's class docstring instead

3. **Lists of Models** (List[SomeModel]):
   - ❌ CANNOT have descriptions
   - Already warned by VotingField for List[ComplexType]

### Best Practices

1. **Document in the Model**: 
   ```python
   class EvaluationFactor(BaseModel):
       """Document what this factor represents here."""
       score: float
       reasoning: str
   ```

2. **Use Field Names**: Make field names self-documenting:
   ```python
   # Instead of: factor: EvaluationFactor = VotingField(description="...")
   # Use: contradiction_presence: EvaluationFactor = VotingField()
   ```

3. **Add Comments**: Use Python comments for clarification:
   ```python
   # Evaluates whether contradictions exist in the answer
   contradiction_presence: EvaluationFactor = VotingField()
   ```

### Detection Pattern
Look for this error pattern:
- Error code: 400
- Message contains: `$ref cannot have keywords {'description'}`
- Context shows a field name that references another model

### Fix Checklist
- [ ] Identify the field causing the error (shown in context)
- [ ] Confirm it references another Pydantic model
- [ ] Remove the `description` parameter from VotingField
- [ ] Add documentation to the model class if needed
- [ ] Test with OpenAI API to confirm fix