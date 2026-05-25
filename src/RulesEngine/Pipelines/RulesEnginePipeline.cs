namespace EvalApp.Solid.Starter.Features.RulesEngine;

/// <summary>
/// RulesEngine Pipeline — demonstrates SOLID principles via pricing logic.
/// 
/// Flow:
///   1. CalculateNetPrice — Sum items (SRP: one responsibility)
///   2. EvaluateEligibility — Determine discount eligibility (OCP: extensible rules)
///   3. ApplyPromoRules — Apply business rules via decision tree (no if/else explosion)
///   4. CalculateFinalPrice — Apply discount (SRP: one responsibility)
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
            .DefineDomain("Pricing")
                .DefineTask<PricingData>("CalculatePrice")
                    .AddStep("CalculateNetPrice", new CalculateNetPriceStep())
                    .AddStep("EvaluateEligibility", new EvaluateDiscountEligibilityStep())
                    .AddStep("ApplyPromoRules", new ApplyPromotionRulesStep())
                    .AddStep("CalculateFinalPrice", new CalculateFinalPriceStep())
                    .Run(out pipeline)
                .Build();

        return pipeline;
    }
}
