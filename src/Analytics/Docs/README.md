# Quote Intelligence Service (AdvancedPatterns)

Back to platform overview: [Root README](../../../README.md)

## Business Requirement

Northstar’s quoting surface must remain responsive under variable load while preserving:

- resilience (fallback when primary source degrades)
- observability (traceable path decisions)
- auditability (replayable snapshots and integrity markers)

## Implemented Business Rules

Source: `src/AdvancedPatterns/Pipelines/AdvancedPatternsPipeline.cs`

1. Request metadata is stamped before quote retrieval.
2. Primary quote path returns premium value (`125`) when healthy.
3. Fallback quote path returns conservative value (`100`) on primary failure.
4. Negative items are rejected during transform stage.
5. Output includes digest/snapshot traces for operational replay.

Runtime controls:

- middleware chain: Trace -> RetryOnce -> TimeoutGuard
- `ForEach` failure modes selectable (`ContinueOnError`, `FailFast`, `CollectAndThrow`)
- resource boundaries for Network, CPU, and DiskIO

## Features Demonstrated

**EvalApp pattern:** Middleware chain, fallback branching, resource gates, ForEach failure modes

**SOLID principle:** LSP (middleware and fallback steps are interchangeable)

## Implementation


| Concern | Path |
|---|---|
| Pipeline declaration | `src/AdvancedPatterns/Pipelines/AdvancedPatternsPipeline.cs` |
| Middleware behavior | `src/AdvancedPatterns/Middleware/` |
| Helper operations | `src/AdvancedPatterns/Helpers/AdvancedPatternHelpers.cs` |
| Executable specs | `Tests/Features/AdvancedPatterns/` |


Verify: `dotnet test --filter "AdvancedPatterns"`
