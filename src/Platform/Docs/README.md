# Platform Capability Validation Service (ApiSurface)

Back to platform overview: [Root README](../../../README.md)

## Business Requirement

Northstar platform engineering needs a controlled service that validates advanced pipeline capabilities before they are adopted by product teams:

- lifecycle events
- step factory activation behavior
- parallel merge semantics
- bridge projections
- saga compensation behavior
- pressure/window budget controls

This service is intentionally synthetic and internal-facing.

## Implemented Business Rules

Source: `src/ApiSurface/Pipelines/ApiSurfacePipeline.cs`

1. Policy bonus can be injected through service-provider step factory.
2. Parallel heuristics (`+10`, `*3`) are merged by strategy contract.
3. Bridge projection contributes baseline benchmark value.
4. Saga reserve and gate stages increment counters, with compensation on failure.
5. Pressure and window scopes emit markers for behavior verification.

## Features Demonstrated

**EvalApp pattern:** Lifecycle events, service factory resolution, saga compensation, tuning variants, pressure/window budgets

**SOLID principle:** ISP (service factory and merge strategy are abstraction-driven)

## Implementation


| Concern | Path |
|---|---|
| Coverage pipeline | `src/ApiSurface/Pipelines/ApiSurfacePipeline.cs` |
| Events/merge/support/steps | `src/ApiSurface/Events/`, `Merge/`, `Support/`, `Steps/` |
| Executable specs | `Tests/Features/ApiSurface/` |


Verify: `dotnet test --filter "ApiSurface"`
