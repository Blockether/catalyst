# Coding Standards: Settings and Configuration Naming

## Updated Rule: UPPERCASE for classic class constants; lowercase for Pydantic model fields

- **Non-Pydantic classes** (plain Python classes, dataclasses, etc.): configuration settings, thresholds, and constants should use `UPPER_CASE` names to distinguish immutable settings from runtime state.
- **Pydantic models** (`BaseModel` subclasses, including settings/config objects): fields must remain in `lower_snake_case` to align with Pydantic conventions and avoid breaking validation, serialization, and template lookups.
- `model_config`, `Config`, and other Pydantic-required identifiers remain untouched and stay in lowercase/camelcase as dictated by the framework.
- Instance variables continue to use `lower_snake_case`, with private attributes prefixed by `_` as usual.

This keeps standard Python constants obvious while respecting Pydantic's API expectations.