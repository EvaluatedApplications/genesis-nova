namespace EvalApp.Solid.Starter.Catalog.Pipelines;

/// <summary>
/// IngestionPipeline — demonstrates stream validation and partial success.
/// 
/// Flow:
///   1. Materialize — Initialize output collections (ValidItems, InvalidItems)
///   2. ProcessAllItems — Iterate stream, validate each, collect successes and failures
///   3. SummarizeResults — Count successes/errors, build summary string
///
/// SOLID Benefits:
/// - SRP: Each step is a single, focused responsibility
/// - OCP: Easy to add new validation rules without changing pipeline topology
/// - DIP: Steps depend on abstraction (PureStep<T>), not concrete implementations
///
/// Partial Success Semantics:
/// - Iterates all items despite validation failures
/// - Both ValidItems and InvalidItems are populated (not mutually exclusive)
/// - Final data record includes both success and error counts for reporting
/// </summary>
public static class IngestionPipeline
{
    /// <summary>
    /// Build the foundational stream-processing pipeline.
    /// The capstone orchestration feature demonstrates ForEach and adaptive concurrency.
    /// </summary>
    public static ICompiledPipeline<IngestionData> Build()
    {
        ICompiledPipeline<IngestionData> pipeline = null!;

        Eval.App("Ingestion")
            .DefineDomain("BatchProcessing")
                .DefineTask<IngestionData>("ProcessStream")
                    .AddStep("Materialize", new MaterializeStep())
                    .AddStep("ProcessAllItems", new ProcessAllItemsStep())
                    .AddStep("Summarize", new SummarizeResultsStep())
                    .Run(out pipeline)
                .Build();

        return pipeline;
    }
}

