using System.Collections.Immutable;
using EvalApp.Consumer;
using EvalApp.Solid.Starter.Commerce;
using EvalApp.Solid.Starter.Commerce.Contexts;
using EvalApp.Solid.Starter.Commerce.Steps;

namespace EvalApp.Solid.Starter.Commerce.Pipelines;

public static class CommerceFulfillmentPipeline
{
    public static ICompiledPipeline<CommerceWorkflowData> Build(FulfillmentDomainContext? context = null)
    {
        ICompiledPipeline<CommerceWorkflowData> pipeline = null!;

        Eval.App("CommerceTutorial")
            .WithContext(NullGlobalContext.Instance)
            .WithResource(ResourceKind.Network, new TunableConfig(Min: 1, Max: 8, Default: 4))
            .WithResource(ResourceKind.Database, new TunableConfig(Min: 1, Max: 4, Default: 2))
            .DefineDomain("Fulfillment", context ?? FulfillmentDomainContext.Default)
                .DefineTask<CommerceWorkflowData>("FulfillOrder")
                    .AddStep<PrepareFulfillmentLinesStep>("PrepareLines")
                    .ForEach<CommerceLineItem>(
                        data => data.Lines,
                        (data, lines) => data with
                        {
                            Lines = lines.ToImmutableList(),
                            Trace = data.AppendTrace($"Fulfillment:Packed:{lines.Count}").Trace
                        },
                        "FulfillmentLines",
                        Tunable.ForItems(),
                        ForEachFailureMode.ContinueOnError,
                        item => item.AddStep<PackLineStep>("PackLine"))
                    .If(
                        data => data.FinalTotal >= (context ?? FulfillmentDomainContext.Default).FreeShippingThreshold,
                        then: branch => branch.AddStep<SelectShippingStep>("SelectExpressShipping"),
                        @else: branch => branch.AddStep<SelectShippingStep>("SelectStandardShipping"))
                    .Gate(
                        ResourceKind.Network,
                        data => { },
                        gate => gate.AddStep<GenerateShippingLabelStep>("GenerateLabel"))
                    .Gate(
                        ResourceKind.Database,
                        data => { },
                        gate => gate.AddStep<ArchiveOrderStep>("ArchiveOrder"))
                    .Run(out pipeline)
                .Build();

        return pipeline;
    }
}

