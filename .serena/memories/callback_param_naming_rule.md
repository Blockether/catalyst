# Callback Parameter Naming Rule

- When defining function parameters intended to accept callbacks (callable objects), ensure their names end with `_fn`.
- Applies to public and internal APIs (functions, methods, and Pydantic model fields representing callbacks).
- Examples: `validator_fn`, `on_complete_fn`, `formatter_fn`.
- Do not rename parameters provided by third-party interfaces or protocols; only enforce this rule for our own APIs.