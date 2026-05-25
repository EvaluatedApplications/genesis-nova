using EvalApp.Consumer;
using EvalApp.Solid.Starter.Pricing;
using EvalApp.Solid.Starter.Pricing.Context;

namespace EvalApp.Solid.Starter.Pricing.Pipelines;

/// <summary>
/// RulesEngine Pipeline — demonstrates SOLID principles via conditional pricing logic.
/// 
/// Flow:
///   1. CalculateNetPrice — Sum items (SRP: one responsibility)
///   2. EvaluateEligibility — Determine discount eligibility (OCP: extensible rules)
///   3. ApplyPromoRules — Apply business rules via decision tree (no if/else explosion)
///   4. ApplyTax — Calculate tax on discounted price
///   5. CalculateFinalPrice — Apply discount and finalize (SRP: one responsibility)
///
/// Phase 4 Pattern:
/// - Introduces PricingContext for domain-specific configuration
/// - Uses ContextPureStep for tax calculation so policy is injected, not hardcoded
/// - Each step operates on immutable PricingData records
/// 
/// If/Else Pattern Demonstration:
/// - The tutorial still uses data-driven rules inside steps for pricing decisions
/// - Branching is shown in the cross-domain orchestration capstone instead
/// 
/// SOLID Benefits:
/// - SRP: Each step is a single, focused responsibility.
/// - OCP: New rules added in ApplyPromotionRulesStep without changing pipeline topology.
/// - DIP: Steps depend on abstraction (PureStep<T>), not concrete implementations.
/// </summary>
public static class RulesEnginePipeline
{
    public static ICompiledPipeline<PricingData> Build()
    {
        ICompiledPipeline<PricingData> pipeline = null!;

        Eval.App("RulesEngine")
            .WithContext(NullGlobalContext.Instance)
            .DefineDomain("Pricing", PricingContext.Default)
                .DefineTask<PricingData>("CalculatePrice")
                    .AddStep("CalculateNetPrice", new CalculateNetPriceStep())
                    .AddStep("EvaluateEligibility", new EvaluateDiscountEligibilityStep())
                    .AddStep<ApplyPromotionRulesStep>("ApplyPromoRules")
                    .AddStep<ApplyTaxStep>("ApplyTax")
                    .AddStep("CalculateFinalPrice", new CalculateFinalPriceStep())
                    .Run(out pipeline)
                .Build();

        return pipeline;
    }
}

