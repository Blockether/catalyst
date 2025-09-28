# Backward Compatibility Policy

- Do **not** implement backward compatibility shims or fallbacks.
- When APIs change (names, signatures, behavior), update internal usage rather than preserving old contracts.
- Remove or overhaul legacy code instead of keeping compatibility layers.
- Document breaking changes in code comments or changelog as needed, but do not reintroduce deprecated behavior.