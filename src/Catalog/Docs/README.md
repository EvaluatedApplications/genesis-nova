# Catalog Intake Service (Ingestion)

Back to platform overview: [Root README](../../../README.md)

## Business Requirement

Northstar ingests third-party catalog records continuously:

- valid rows should be promoted immediately
- invalid rows should be quarantined with explicit reason
- bad data must not block good data from flowing

## Implemented Business Rules

Source: `src/Ingestion/Steps/ValidateItemStep.cs`, `src/Ingestion/Steps/ProcessAllItemsStep.cs`

1. `Name` is required (blank names are rejected).
2. `Amount` must be greater than zero.
3. Valid rows are transformed into `ValidatedRecord`.
4. Invalid rows become `ValidationError` entries.
5. Pipeline summarizes success/error counts from collected outputs.

## Features Demonstrated

**EvalApp pattern:** ForEach with error-as-data collection

**SOLID principle:** OCP (validation rules expand via new steps, not rewrites)

## Implementation


| Concern | Path |
|---|---|
| Pipeline topology | `src/Ingestion/Pipelines/IngestionPipeline.cs` |
| Validation rules | `src/Ingestion/Steps/ValidateItemStep.cs` |
| Item processing | `src/Ingestion/Steps/ProcessAllItemsStep.cs` |
| Executable specs | `Tests/Features/Ingestion/` |


Verify: `dotnet test --filter "Ingestion"`
