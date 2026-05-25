namespace EvalApp.Solid.Starter.Features.BatchSync;

/// <summary>
/// BatchSync Pipeline — demonstrates throttled batch processing with partial success.
///
/// Flow:
///   1. FetchItems — Generate or load ItemIds to process
///   2. ProcessBatch — Call API for each item, handle failures, populate Results
///   3. CalculateSummary — Count successes/errors, produce summary metrics
///
/// SOLID Benefits:
/// - SRP: Each step has single responsibility (fetch, process, summarize)
/// - OCP: Easy to add new processing strategies without changing topology
/// - DIP: Steps depend on abstraction (PureStep, AsyncStep), not concrete implementations
///
/// Failure Modes:
/// - Partial success: Some items succeed, others timeout/fail
/// - Both successful Results and FailedIds are populated
/// - ErrorCount + SuccessCount = TotalItems
///
/// Customization:
/// - Adjust successRate parameter to simulate different API reliability
/// - Adjust minDelayMs/maxDelayMs to simulate different latencies
/// </summary>
public static class BatchSyncPipeline
{
    /// <summary>
    /// Build pipeline with adaptive concurrency tuning (licensed mode).
    /// </summary>
    public static ICompiledPipeline<BatchSyncData> Build(
        double successRate = 0.8,
        int minDelayMs = 10,
        int maxDelayMs = 100)
    {
        ICompiledPipeline<BatchSyncData> pipeline = null!;

        Eval.App("BatchSync")
            .DefineDomain("Processing")
                .DefineTask<BatchSyncData>("SyncBatch")
                    .AddStep("FetchItems", new FetchItemsStep())
                    .AddStep("ProcessBatch", new ProcessBatchStep(successRate, minDelayMs, maxDelayMs))
                    .AddStep("CalculateSummary", new CalculateSummaryStep())
                    .Run(out pipeline)
                .Build();

        return pipeline;
    }

    /// <summary>
    /// Build simple sequential pipeline (unlicensed mode - no tuning).
    /// </summary>
    public static ICompiledPipeline<BatchSyncData> BuildSimple(
        double successRate = 0.8,
        int minDelayMs = 10,
        int maxDelayMs = 100)
    {
        return Build(successRate, minDelayMs, maxDelayMs);
    }
}
