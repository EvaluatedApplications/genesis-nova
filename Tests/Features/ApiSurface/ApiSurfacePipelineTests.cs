using System.Reflection;
using EvalApp.Consumer;
using EvalApp.Extensions;
using EvalApp.Solid.Starter.Features.ApiSurface;
using EvalApp.Solid.Starter.Features.ApiSurface.Pipelines;

namespace EvalApp.Solid.Starter.Tests.Features.ApiSurface;

public class ApiSurfacePipelineTests
{
    [Fact]
    public async Task WhenApiSurfaceRuns_Then_ParallelBridgePressureAndEventsAreApplied()
    {
        var events = new List<string>();
        var pipeline = ApiSurfacePipeline.Build(events);

        var result = await pipeline.RunAsync(new ApiSurfaceData(Input: 4));
        Assert.True(result.IsSuccess);

        var finalData = result.GetData();

        Assert.Equal(14, finalData.ParallelLeft);
        Assert.Equal(12, finalData.ParallelRight);
        Assert.Equal(1007, finalData.BridgedValue);
        Assert.Equal(18, finalData.SagaCounter);
        Assert.True(finalData.PressureScoped);

        Assert.NotNull(finalData.ProcessedSagaItems);
        Assert.Equal(3, finalData.ProcessedSagaItems!.Count);
        Assert.Contains(40, finalData.ProcessedSagaItems);
        Assert.Contains(50, finalData.ProcessedSagaItems);
        Assert.Contains(60, finalData.ProcessedSagaItems);

        Assert.Contains(finalData.Trace ?? [], t => t.Contains("ParallelGroup:CustomMerge", StringComparison.Ordinal));
        Assert.Contains(finalData.Trace ?? [], t => t.Contains("Pressure:CustomScope", StringComparison.Ordinal));
        Assert.Contains(events, e => e.StartsWith("Pipeline:Starting", StringComparison.Ordinal));
        Assert.Contains(events, e => e.StartsWith("Pipeline:Completed", StringComparison.Ordinal));
    }

    [Fact]
    public async Task WhenSagaGateFails_Then_PipelineReturnsFailure()
    {
        var pipeline = ApiSurfacePipeline.Build();
        var result = await pipeline.RunAsync(new ApiSurfaceData(Input: 3, TriggerSagaFailure: true));

        Assert.True(result.IsFailure);
        var finalData = result.GetData();
        Assert.DoesNotContain(finalData.Trace ?? [], t => t.Contains("ApiSurface:Completed", StringComparison.Ordinal));
    }

    [Fact]
    public void WhenPipelineIsSerialized_Then_SagaAndParallelNodesArePresent()
    {
        var pipeline = ApiSurfacePipeline.Build();
        var root = GetPipelineRoot(pipeline);

        var dryRun = PipelineDryRun<ApiSurfaceData>.Validate(root);
        Assert.True(dryRun.IsValid);

        var json = PipelineSerializer<ApiSurfaceData>.ToJson(root);
        var visual = PipelineVisualizer<ApiSurfaceData>.ToTextTree(root);

        Assert.Contains("\"type\": \"Saga\"", json);
        Assert.Contains("\"type\": \"Parallel\"", json);
        Assert.Contains("[Saga", visual);
        Assert.Contains("[Parallel]", visual);
    }

    [Fact]
    public async Task WhenBayesianAndSettingsEnabled_Then_PipelineStillRuns()
    {
        var tempSettings = Path.Combine(Path.GetTempPath(), $"evalapp-solid-tuner-{Guid.NewGuid():N}.json");
        await File.WriteAllTextAsync(tempSettings, "{}");

        try
        {
            var pipeline = ApiSurfacePipeline.BuildWithBayesianTuning(tempSettings);
            var result = await pipeline.RunAsync(new ApiSurfaceData(Input: 2));
            Assert.True(result.IsSuccess);
        }
        finally
        {
            if (File.Exists(tempSettings))
                File.Delete(tempSettings);
        }
    }

    [Fact]
    public void WhenNoLicenseKey_Then_LicenseValidatorReturnsUnlicensed()
    {
        var validator = LicenseValidator.ForNavPathfinder();

        var mode = validator.Check(null);
        var periodic = validator.CheckPeriodic(null);

        Assert.Equal(LicenseMode.Unlicensed, mode);
        Assert.Equal(LicenseMode.Unlicensed, periodic);
    }

    private static dynamic GetPipelineRoot(ICompiledPipeline<ApiSurfaceData> pipeline)
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
