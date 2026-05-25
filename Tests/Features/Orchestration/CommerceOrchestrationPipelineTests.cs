using System.Collections.Immutable;
using System.Reflection;
using EvalApp.Extensions;
using EvalApp.Solid.Starter.Features.Orchestration;
using EvalApp.Solid.Starter.Features.Orchestration.Pipelines;
using EvalApp.Solid.Starter.Shared;

namespace EvalApp.Solid.Starter.Tests.Features.Orchestration;

public class CommerceOrchestrationPipelineTests
{
    [Fact]
    public async Task WhenVipOrder_Then_ComposesPricingAndFulfillment()
    {
        var order = new OrderContext(
            new ShopperProfile("VIP-1", 18, 4200m, IsVip: true, DateTime.UtcNow),
            ImmutableList.Create(
                new Item("SKU-PHONE", "Phone", 699m, ItemCategory.Premium),
                new Item("SKU-CASE", "Case", 29m, ItemCategory.Standard),
                new Item("SKU-CLEAR", "Cable", 12m, ItemCategory.Clearance)),
            "VIPSHIP");

        var pipeline = CommerceOrchestrationPipeline.Build();
        var result = await pipeline.RunAsync(new CommerceWorkflowData(order, ImmutableList<CommerceLineItem>.Empty));

        var finalData = result.GetData();

        Assert.Equal("Express", finalData.ShippingMethod);
        Assert.NotNull(finalData.LabelId);
        Assert.NotNull(finalData.ArchiveId);
        Assert.True(finalData.FinalTotal > 0m);
        Assert.Contains(finalData.Trace!, t => t.StartsWith("Pricing:"));
        Assert.Contains(finalData.Trace!, t => t.StartsWith("Fulfillment:"));
        Assert.True(finalData.Lines.Count < order.Items.Count);
    }

    [Fact]
    public async Task WhenStandardOrder_Then_UsesStandardShipping()
    {
        var order = new OrderContext(
            new ShopperProfile("STD-1", 1, 50m, IsVip: false, DateTime.UtcNow),
            ImmutableList.Create(
                new Item("SKU-MUG", "Mug", 25m, ItemCategory.Standard),
                new Item("SKU-BOOK", "Book", 20m, ItemCategory.Standard)),
            "");

        var pipeline = CommerceOrchestrationPipeline.Build();
        var result = await pipeline.RunAsync(new CommerceWorkflowData(order, ImmutableList<CommerceLineItem>.Empty));

        var finalData = result.GetData();

        Assert.Equal("Standard", finalData.ShippingMethod);
        Assert.Equal(order.Items.Count, finalData.Lines.Count);
        Assert.NotNull(finalData.LabelId);
        Assert.NotNull(finalData.ArchiveId);
    }

    [Fact]
    public void WhenDryRunValidates_Then_PipelineIsValid()
    {
        var pipeline = CommerceOrchestrationPipeline.Build();
        var dryRun = PipelineDryRun<CommerceWorkflowData>.Validate(GetPipelineRoot(pipeline));

        Assert.True(dryRun.IsValid);
        Assert.True(dryRun.StepCount > 0);
        Assert.True(dryRun.NodeCount > 0);
    }

    [Fact]
    public void WhenVisualized_Then_ShowsPricingAndFulfillment()
    {
        var pipeline = CommerceOrchestrationPipeline.Build();
        var text = PipelineVisualizer<CommerceWorkflowData>.ToTextTree(GetPipelineRoot(pipeline));

        Assert.Contains("PriceOrder", text);
        Assert.Contains("FulfillOrder", text);
    }

    [Fact]
    public void WhenFulfillmentVisualized_Then_ShowsNestedSteps()
    {
        var pipeline = CommerceFulfillmentPipeline.Build();
        var text = PipelineVisualizer<CommerceWorkflowData>.ToTextTree(GetPipelineRoot(pipeline));

        Assert.Contains("PrepareLines", text);
    }

    private static dynamic GetPipelineRoot(ICompiledPipeline<CommerceWorkflowData> pipeline)
    {
        var internalPipelineField = pipeline.GetType().GetField("_pipeline", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("Compiled pipeline does not expose its internal pipeline");

        var internalPipeline = internalPipelineField.GetValue(pipeline)
            ?? throw new InvalidOperationException("Compiled pipeline field was null");

        var rootProperty = internalPipeline.GetType().GetProperty("Root", BindingFlags.Instance | BindingFlags.Public)
            ?? throw new InvalidOperationException("Internal pipeline root was not found");

        return rootProperty.GetValue(internalPipeline)
            ?? throw new InvalidOperationException("Internal pipeline root was null");
    }
}
