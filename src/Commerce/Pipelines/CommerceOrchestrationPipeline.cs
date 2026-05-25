using EvalApp.Consumer;
using EvalApp.Solid.Starter.Commerce;
using EvalApp.Solid.Starter.Commerce.Contexts;
using EvalApp.Solid.Starter.Commerce.Steps;

namespace EvalApp.Solid.Starter.Commerce.Pipelines;

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

