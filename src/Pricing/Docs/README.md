# Pricing Service (RulesEngine)

Back to platform overview: [Root README](../../../README.md)

## Business Requirement

Northstar needs deterministic and explainable pricing for checkout:

- loyalty customers should receive policy-governed discounts
- campaigns must be stack-safe and margin-protected
- tax compliance must apply after discounts

## Implemented Business Rules

Source: `src/RulesEngine/Steps/` and `src/RulesEngine/Context/PricingContext.cs`

1. Eligibility is granted when shopper is VIP, or has purchase history > 5, or total spend > 1000.
2. Clearance in basket enforces at least 20% promotional discount floor.
3. Promo code `SUMMER20` enforces at least 15% promotional discount floor.
4. VIP receives loyalty bonus (`VipLoyaltyBonus`) with total discount capped by `VipDiscount`.
5. Tax is computed on discounted subtotal using context tax rate (default 8%).

Step order is intentional:

`CalculateNetPrice -> EvaluateEligibility -> ApplyPromoRules -> ApplyTax -> CalculateFinalPrice`

## Features Demonstrated

**EvalApp pattern:** Sequential pipeline with context-driven policies

**SOLID principle:** SRP (each step = one pricing concern)

## Implementation


| Concern | Path |
|---|---|
| Pipeline topology | `src/RulesEngine/Pipelines/RulesEnginePipeline.cs` |
| Pricing context policy | `src/RulesEngine/Context/PricingContext.cs` |
| Pricing step implementations | `src/RulesEngine/Steps/` |
| Executable specs | `Tests/Features/RulesEngine/` |


Verify: `dotnet test --filter "RulesEngine"`
