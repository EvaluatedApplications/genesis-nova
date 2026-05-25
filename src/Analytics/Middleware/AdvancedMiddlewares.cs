using EvalApp.Consumer;

namespace EvalApp.Solid.Starter.Analytics.Middleware;

public sealed class TraceMiddleware(string name) : IStepMiddleware<AdvancedDemoData>
{
    private readonly string _name = name;

    public async ValueTask<PipelineResult<AdvancedDemoData>> ExecuteAsync(
        AdvancedDemoData data,
        Func<AdvancedDemoData, CancellationToken, ValueTask<PipelineResult<AdvancedDemoData>>> next,
        CancellationToken ct = default)
    {
        var enrichedInput = data.AppendTrace($"Middleware:{_name}:Before");
        var result = await next(enrichedInput, ct);

        return result switch
        {
            PipelineResult<AdvancedDemoData>.Success success =>
                new PipelineResult<AdvancedDemoData>.Success(
                    success.Data.AppendTrace($"Middleware:{_name}:After:Success")),
            PipelineResult<AdvancedDemoData>.Failure failure =>
                new PipelineResult<AdvancedDemoData>.Failure(
                    failure.Data.AppendTrace($"Middleware:{_name}:After:Failure"),
                    failure.Exception,
                    failure.Message),
            PipelineResult<AdvancedDemoData>.Skipped skipped =>
                new PipelineResult<AdvancedDemoData>.Skipped(
                    skipped.Data.AppendTrace($"Middleware:{_name}:After:Skipped"),
                    skipped.Reason),
            _ => result
        };
    }
}

public sealed class RetryOnceMiddleware : IStepMiddleware<AdvancedDemoData>
{
    public async ValueTask<PipelineResult<AdvancedDemoData>> ExecuteAsync(
        AdvancedDemoData data,
        Func<AdvancedDemoData, CancellationToken, ValueTask<PipelineResult<AdvancedDemoData>>> next,
        CancellationToken ct = default)
    {
        PipelineResult<AdvancedDemoData> firstAttempt;
        try
        {
            firstAttempt = await next(data, ct);
        }
        catch (Exception ex)
        {
            return new PipelineResult<AdvancedDemoData>.Failure(
                data.AppendTrace("Middleware:RetryOnce:FirstAttemptThrew"),
                ex,
                ex.Message);
        }

        if (firstAttempt is not PipelineResult<AdvancedDemoData>.Failure firstFailure)
            return firstAttempt;

        var retryInput = firstFailure.Data.AppendTrace("Middleware:RetryOnce:Retrying");
        try
        {
            var secondAttempt = await next(retryInput, ct);
            return secondAttempt;
        }
        catch (Exception ex)
        {
            return new PipelineResult<AdvancedDemoData>.Failure(
                retryInput.AppendTrace("Middleware:RetryOnce:RetryThrew"),
                ex,
                ex.Message);
        }
    }
}

public sealed class TimeoutGuardMiddleware(TimeSpan timeout) : IStepMiddleware<AdvancedDemoData>
{
    private readonly TimeSpan _timeout = timeout;

    public async ValueTask<PipelineResult<AdvancedDemoData>> ExecuteAsync(
        AdvancedDemoData data,
        Func<AdvancedDemoData, CancellationToken, ValueTask<PipelineResult<AdvancedDemoData>>> next,
        CancellationToken ct = default)
    {
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        cts.CancelAfter(_timeout);

        try
        {
            return await next(data, cts.Token);
        }
        catch (OperationCanceledException ex) when (!ct.IsCancellationRequested)
        {
            return new PipelineResult<AdvancedDemoData>.Failure(
                data.AppendTrace("Middleware:TimeoutGuard:TimedOut"),
                ex,
                $"Pipeline timed out after {_timeout.TotalMilliseconds:F0}ms.");
        }
    }
}

