# API Surface Coverage Feature

## What this teaches

This feature demonstrates advanced EvalApp.Consumer APIs that are easy to miss in basic tutorials:

- `WithStepFactory(...)` using `ServiceProviderStepFactory`
- `WithEvents(...)` with full pipeline/step lifecycle callbacks
- `WithPressure(...)` + scoped `.Pressure(...)`
- `AddParallelGroup(...)` with a custom `IMergeStrategy<T>`
- `AddReadOnlyBridge(...)` for merge-only bridge projections
- true saga flow using `.BeginSaga()` / `.EndSaga()`
- saga-only operations: `AddMaterialize`, `AddForEach`, `AddStepWithCompensation`, and `AddGate(..., compensate)`
- optional tuning variants via `BuildWithBayesianTuning(...)`

## Pipeline shape

1. Resolve a step through a custom step factory.
2. Run two branches in parallel and merge with a custom merge strategy.
3. Run a read-only bridge projection and merge it into the main flow.
4. Enter saga scope:
   - materialize async items
   - process items with saga `ForEach`
   - register compensations
   - run a gated external call with compensation
5. Execute custom pressure scope and window-budget scope.
6. Finalize and return result.

## Why this matters

This is the parity feature for teams who want to understand and exercise nearly all consumer builder surfaces in one concrete example, not just the common `AddStep + Gate + ForEach` path.
