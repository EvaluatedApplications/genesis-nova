using EvalApp.Consumer;
using EvalApp.Solid.Starter.Features.AdvancedPatterns;
using EvalApp.Solid.Starter.Features.AdvancedPatterns.Pipelines;

namespace EvalApp.Solid.Starter.Tests.Features.AdvancedPatterns;

public class AdvancedPatternsPipelineTests
{
    [Fact]
    public async Task WhenPrimaryQuoteFails_Then_UsesFallbackAndCompletes()
    {
        var pipeline = AdvancedPatternsPipeline.Build();
        var input = new AdvancedDemoData(
            InputItems: [1, 2, 3],
            ForcePrimaryQuoteFailure: true);

        var result = await pipeline.RunAsync(input);
        Assert.True(result.IsSuccess);
        var finalData = result.GetData();

        Assert.Equal("fallback", finalData.QuoteSource);
        Assert.Equal(100m, finalData.Quote);
        Assert.NotNull(finalData.CpuDigest);
        Assert.NotNull(finalData.SnapshotPath);
        Assert.True(File.Exists(finalData.SnapshotPath));

        File.Delete(finalData.SnapshotPath);
    }

    [Fact]
    public async Task WhenContinueOnError_Then_InvalidItemsAreSkipped()
    {
        var pipeline = AdvancedPatternsPipeline.Build(ForEachFailureMode.ContinueOnError);
        var input = new AdvancedDemoData(
            InputItems: [2, 4, -1, 8],
            ForcePrimaryQuoteFailure: true);

        var result = await pipeline.RunAsync(input);
        Assert.True(result.IsSuccess);
        var finalData = result.GetData();

        Assert.Equal(3, finalData.SuccessCount);
        Assert.Equal(1, finalData.ErrorCount);
        Assert.Equal([4, 8, 16], finalData.MaterializedItems);

        if (!string.IsNullOrWhiteSpace(finalData.SnapshotPath) && File.Exists(finalData.SnapshotPath))
            File.Delete(finalData.SnapshotPath);
    }

    [Fact]
    public async Task WhenFailFast_Then_PipelineReturnsFailure()
    {
        var pipeline = AdvancedPatternsPipeline.Build(ForEachFailureMode.FailFast);
        var input = new AdvancedDemoData(InputItems: [1, -2, 3]);

        var result = await pipeline.RunAsync(input);
        Assert.True(result.IsFailure);
    }

    [Fact]
    public async Task WhenCollectAndThrow_Then_PipelineReturnsFailure()
    {
        var pipeline = AdvancedPatternsPipeline.Build(ForEachFailureMode.CollectAndThrow);
        var input = new AdvancedDemoData(InputItems: [1, -2, -3]);

        var result = await pipeline.RunAsync(input);
        Assert.True(result.IsFailure);
    }

    [Fact]
    public async Task WhenAdvancedPipelineRuns_Then_MiddlewareAndWindowBudgetAreApplied()
    {
        var pipeline = AdvancedPatternsPipeline.Build();
        var input = new AdvancedDemoData(InputItems: [2, 4, 6]);

        var result = await pipeline.RunAsync(input);
        Assert.True(result.IsSuccess);
        var finalData = result.GetData();

        Assert.NotNull(finalData.Trace);
        Assert.Contains(finalData.Trace, t => t.Contains("Middleware:Trace:Before", StringComparison.Ordinal));
        Assert.Contains(finalData.Trace, t => t.Contains("WindowBudget:Applied", StringComparison.Ordinal));
        Assert.Contains(finalData.Trace, t => t.Contains("Cpu:DigestComputed", StringComparison.Ordinal));
        Assert.Contains(finalData.Trace, t => t.Contains("Disk:SnapshotWritten", StringComparison.Ordinal));

        if (!string.IsNullOrWhiteSpace(finalData.SnapshotPath) && File.Exists(finalData.SnapshotPath))
            File.Delete(finalData.SnapshotPath);
    }
}
