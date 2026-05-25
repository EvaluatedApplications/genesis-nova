# Commerce Orchestration Feature

## What this teaches

This capstone shows how EvalApp handles:
- multiple domains in one app
- pipeline composition (`Pipeline<T>` as a step)
- cross-domain data flow by passing one pipeline output into the next
- `ContextPureStep` and `ContextSideEffectStep`
- `ForEach` with tunable parallelism
- `Gate(ResourceKind.X)` with different resource kinds
- `If` branching at the topology level

## Domain map

| Domain | Responsibility |
|---|---|
| Pricing | Calculate the quote from the order |
| Fulfillment | Pack items, choose shipping, generate label, archive order |
| Orchestration | Compose pricing and fulfillment into one end-to-end flow |

## Pipeline shape

1. `PriceOrder` runs the pricing pipeline.
2. `FulfillOrder` runs the fulfillment pipeline.
3. The orchestration pipeline composes both, so the output of one becomes the input of the next.

## Why this matters

This is the closest thing to a real application story in the tutorial:
- one data record flows across multiple domains
- policy is injected through context
- line items are processed in parallel
- external work is gated
- the final order is archived after shipping is prepared

## Try it

Run:

```bash
dotnet run
```

Then inspect the console trace to see the pricing and fulfillment stages run in order.
