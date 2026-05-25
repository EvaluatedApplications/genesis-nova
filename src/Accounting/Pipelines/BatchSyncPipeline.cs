namespace EvalApp.Solid.Starter.Accounting.Pipelines;

/// <summary>
/// BatchSync Pipeline — demonstrates batched async I/O with partial success.
///
/// Flow:
///   1. FetchItems — Generate or load ItemIds to process (CPU-only, no gate)
///   2. ProcessBatch — Call API for each item (GATED by Network resource)
///   3. CalculateSummary — Count successes/errors (CPU-only, no gate)
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
    /// Build the foundational batch pipeline.
    /// The capstone orchestration feature demonstrates gates and adaptive tuning.
    /// </summary>
    public static ICompiledPipeline<BatchSyncData> Build(
        double successRate = 0.8,
        int minDelayMs = 10,
        int maxDelayMs = 100)
    {
        // Use the simple build for now; tuning can be added in Phase 2
        return BuildSimple(successRate, minDelayMs, maxDelayMs);
    }

    /// <summary>
    /// Build simple sequential pipeline without gates (for comparison/testing).
    /// </summary>
    public static ICompiledPipeline<BatchSyncData> BuildSimple(
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
}

