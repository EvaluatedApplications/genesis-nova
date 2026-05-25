using EvalApp.Consumer;
using EvalApp.Solid.Starter.Features.Orchestration;
using EvalApp.Solid.Starter.Features.Orchestration.Contexts;
using EvalApp.Solid.Starter.Features.Orchestration.Steps;

namespace EvalApp.Solid.Starter.Features.Orchestration.Pipelines;

public static class CommerceOrchestrationPipeline
{
    public static ICompiledPipeline<CommerceWorkflowData> Build(
        PricingDomainContext? pricingContext = null,
        FulfillmentDomainContext? fulfillmentContext = null)
    {
        ICompiledPipeline<CommerceWorkflowData> orchestrationPipeline = null!;

        var pricingPipeline = CommercePricingPipeline.Build(pricingContext);
        var fulfillmentPipeline = CommerceFulfillmentPipeline.Build(fulfillmentContext);

        var app = Eval.App("CommerceTutorial")
            .WithContext(NullGlobalContext.Instance);

        app.DefineDomain("Orchestration", NullGlobalContext.Instance)
            .DefineTask<CommerceWorkflowData>("RunCommerce")
                .AddStep("PriceOrder", new PriceOrderStep(pricingPipeline))
                .AddStep("FulfillOrder", new FulfillOrderStep(fulfillmentPipeline))
                .Run(out orchestrationPipeline)
            .Build();

        return orchestrationPipeline;
    }
}
