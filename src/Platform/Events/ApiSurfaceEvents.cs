using CoreAbstractions = EvalApp.Abstractions;

namespace EvalApp.Solid.Starter.Platform.Events;

public sealed class ApiSurfaceEvents(List<string> eventLog) : CoreAbstractions.IPipelineEvents<ApiSurfaceData>
{
    private readonly List<string> _eventLog = eventLog;

    public ValueTask OnPipelineStarting(ApiSurfaceData data, CoreAbstractions.StepContext context)
    {
        _eventLog.Add("Pipeline:Starting");
        return default;
    }

    public ValueTask OnPipelineCompleted(CoreAbstractions.StepResult<ApiSurfaceData> result, CoreAbstractions.StepContext context)
    {
        _eventLog.Add($"Pipeline:Completed:{result.GetType().Name}");
        return default;
    }

    public ValueTask OnStepStarting(ApiSurfaceData data, CoreAbstractions.StepInfo stepInfo, CoreAbstractions.StepContext context)
    {
        _eventLog.Add($"Step:Starting:{stepInfo.StepName}");
        return default;
    }

    public ValueTask OnStepCompleted(CoreAbstractions.StepResult<ApiSurfaceData> result, CoreAbstractions.StepInfo stepInfo, CoreAbstractions.StepContext context)
    {
        _eventLog.Add($"Step:Completed:{stepInfo.StepName}:{(result.IsSuccess ? "Success" : "Failure")}");
        return default;
    }
}

