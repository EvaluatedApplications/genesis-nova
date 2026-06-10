namespace EvalApp.Consumer;

/// <summary>
/// Defines event callbacks for pipeline execution.
/// Implementations should provide fire-and-forget event handling without blocking the pipeline.
/// </summary>
public interface IPipelineEvents<T>
{
    /// <summary>
    /// Called when a pipeline step begins execution.
    /// </summary>
    ValueTask OnStepStartedAsync(string stepName, T data);

    /// <summary>
    /// Called when a pipeline step completes successfully.
    /// </summary>
    ValueTask OnStepCompletedAsync(string stepName, T data, long elapsedMs);

    /// <summary>
    /// Called when a pipeline step encounters an error.
    /// </summary>
    ValueTask OnErrorAsync(string stepName, Exception exception, T partialState);

    /// <summary>
    /// Called to report overall progress percentage (0-100).
    /// </summary>
    ValueTask OnProgressAsync(double percentage, string message);
}
