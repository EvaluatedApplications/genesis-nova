# Advanced Patterns Feature

## What this teaches

This feature demonstrates advanced EvalApp capabilities in one runnable pipeline:

- `WithResource(...)` + `WithTuning()` for adaptive concurrency signals
- `WithWindowBudget(...)` and task-level `.WindowBudget(...)`
- multiple middleware layers (`Trace`, retry, timeout guard)
- `Materialize(...)` from async streams into pipeline state
- `AddSubTaskFor(...)` for scoped sub-task transformations
- `AddStepWithFallback(...)` for resilient primary/fallback behavior
- CPU and Disk I/O gates in addition to Network gates
- `ForEach` permutations using `ContinueOnError`, `FailFast`, and `CollectAndThrow`

## Pipeline shape

1. Seed metadata and materialize async input.
2. Run a sub-task that stamps metadata.
3. Fetch a quote with fallback inside a Network gate.
4. Transform items in `ForEach` with a selectable failure mode.
5. Apply a window-budget marker.
6. Compute a digest in a CPU gate.
7. Persist a snapshot in a Disk I/O gate.

## Why this matters

This module is the "full surface" tutorial sample: it combines reliability, throttling, composability, and observability patterns that appear in production pipelines.

## Try it

Run:

```bash
dotnet run
```

Then inspect the Advanced Patterns section and `Trace` output to see middleware markers, fallback behavior, and gate-executed stages.
