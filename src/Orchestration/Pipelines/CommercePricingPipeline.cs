using EvalApp.Consumer;
using EvalApp.Solid.Starter.Features.Orchestration;
using EvalApp.Solid.Starter.Features.Orchestration.Contexts;
using EvalApp.Solid.Starter.Features.Orchestration.Steps;

namespace EvalApp.Solid.Starter.Features.Orchestration.Pipelines;

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
