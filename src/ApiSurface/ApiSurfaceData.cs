namespace EvalApp.Solid.Starter.Features.ApiSurface;

public sealed record ApiSurfaceData(
    int Input,
    bool TriggerSagaFailure = false,
    List<int>? SagaItems = null,
    List<int>? ProcessedSagaItems = null,
    int ParallelLeft = 0,
    int ParallelRight = 0,
    int BridgedValue = 0,
    int SagaCounter = 0,
    bool SagaCompensated = false,
    bool PressureScoped = false,
    List<string>? Trace = null)
{
    public ApiSurfaceData AppendTrace(string entry)
    {
        var trace = Trace is null ? new List<string>() : new List<string>(Trace);
        trace.Add(entry);
        return this with { Trace = trace };
    }
}
