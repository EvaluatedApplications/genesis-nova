# Northstar Commerce Processing Platform (Reference Repo)

This repository represents a fictional production codebase for **Northstar Commerce Group**.

It demonstrates how Northstar implements business workflows (pricing, Catalog, settlement, fulfillment) using EvalApp pipelines with SOLID-driven design.

## Business Context

Northstar operates a multi-channel commerce platform with three recurring operational pressures:

1. **Price personalization must be fast and explainable**
2. **Third-party feeds are noisy but cannot block operations**
3. **Order fulfillment spans multiple external systems and partial-failure scenarios**

This repo models those pressures as runnable services.

## Business Requirements -> Implemented Services

| Requirement | Service module | Primary source |
|---|---|---|
| Personalized, policy-driven pricing | Pricing | `src/Pricing/` |
| Nightly partner settlement reconciliation | Accounting | `src/Accounting/` |
| Catalog intake with quarantine of bad records | Catalog | `src/Catalog/` |
| Inventory/payment/shipment transaction flow | Orders | `src/Orders/` |
| End-to-end quote-to-fulfillment Commerce | Commerce | `src/Commerce/` |
| Resilient low-latency quote path under load | Analytics | `src/Analytics/` |
| Platform API-surface and parity validation | Platform | `src/Platform/` |

## Architecture Snapshot

Northstar’s application entrypoint is one unified pipeline declaration in `src/Program.cs`:

- single `Eval.App("SolidStarter")`
- multiple business domains (`DefineDomain(...)`)
- composed sub-pipelines where Commerce chains outputs into downstream inputs
- centralized resource/tuning declarations at app level

## Why These Technical Choices

### Why EvalApp

Northstar selected EvalApp because business flows require:

- explicit Commerce topology (`ForEach`, `If`, `Gate`, saga)
- predictable handling of partial failures
- controllable throughput and resource boundaries
- source-visible architecture (builder chain as executable map)

### Why SOLID

Each service demonstrates a SOLID principle in action:

- Pricing: **SRP** (each step = one pricing concern)
- Orders: **DIP** (depends on service interfaces, not implementations)
- Catalog: **OCP** (validation rules expand via steps, not rewrites)
- Commerce: **ISP** (narrow, composable pipeline contracts)
- Analytics: **LSP** (middleware and fallback steps are interchangeable)

## Run the Platform

```bash
dotnet build
dotnet run --project src/EvalApp.Solid.Starter.csproj
dotnet test
```


## Service Documentation

Each service README links EvalApp capabilities to SOLID principles:

| Service | Purpose | Documentation |
|---------|---------|---|
| **Pricing** | Personalized, policy-driven pricing | [Read](src/Pricing/Docs/README.md) |
| **Orders** | Multi-step fulfillment with saga pattern | [Read](src/Orders/Docs/README.md) |
| **Catalog** | Feed intake with validation and quarantine | [Read](src/Catalog/Docs/README.md) |
| **Accounting** | Nightly settlement reconciliation | [Read](src/Accounting/Docs/README.md) |
| **Commerce** | End-to-end quote-to-fulfillment orchestration | [Read](src/Commerce/Docs/README.md) |
| **Analytics** | Resilient low-latency processing under load | [Read](src/Analytics/Docs/README.md) |
| **Platform** | API surface and parity validation | [Read](src/Platform/Docs/README.md) |


