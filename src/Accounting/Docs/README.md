# Settlement Sync Service (BatchSync)

Back to platform overview: [Root README](../../../README.md)

## Business Requirement

Northstar finance requires nightly reconciliation with external partner systems:

- every transaction ID must be accounted for
- partial completion is acceptable, silent loss is not
- outcomes must be reproducible for audit and incident replay

## Implemented Business Rules

Source: `src/BatchSync/Steps/ProcessBatchStep.cs`

1. Every `ItemId` is processed and classified as success or failure.
2. `SuccessCount + ErrorCount` must equal total input item count.
3. Failures are retained in `FailedIds` for follow-up replay.
4. Timeout and server-error lanes are simulated deterministically for repeatable CI behavior.

## Features Demonstrated

**EvalApp pattern:** Async step with partial-success modeling in data contract

**SOLID principle:** DIP (external API strategy is swapped behind service abstractions)

## Implementation


| Concern | Path |
|---|---|
| Pipeline topology | `src/BatchSync/Pipelines/BatchSyncPipeline.cs` |
| Batch processing behavior | `src/BatchSync/Steps/ProcessBatchStep.cs` |
| Data model | `src/BatchSync/BatchSyncData.cs` |
| Executable specs | `Tests/Features/BatchSync/` |


Verify: `dotnet test --filter "BatchSync"`
