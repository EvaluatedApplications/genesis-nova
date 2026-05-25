using EvalApp.Consumer;
using EvalApp.Solid.Starter.Commerce;
using EvalApp.Solid.Starter.Commerce.Contexts;
using EvalApp.Solid.Starter.Commerce.Steps;

namespace EvalApp.Solid.Starter.Commerce.Pipelines;

public static class CommercePricingPipeline
{
    public static ICompiledPipeline<CommerceWorkflowData> Build(PricingDomainContext? context = null)
    {
        ICompiledPipeline<CommerceWorkflowData> pipeline = null!;

        Eval.App("CommerceTutorial")
            .WithContext(NullGlobalContext.Instance)
            .DefineDomain("Pricing", context ?? PricingDomainContext.Default)
                .DefineTask<CommerceWorkflowData>("PriceOrder")
                    .AddStep<CalculateQuoteStep>("CalculateQuote")
                    .Run(out pipeline)
                .Build();

        return pipeline;
    }
}

