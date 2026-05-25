namespace EvalApp.Solid.Starter.Catalog;

/// <summary>
/// SummarizeResults — build summary of processing outcome.
/// Counts successes, errors, and constructs human-readable summary string.
/// Pure step: one responsibility (summary aggregation).
/// </summary>
public class SummarizeResultsStep : PureStep<IngestionData>
{
    public override IngestionData Execute(IngestionData data)
    {
        if (data.ValidItems == null || data.InvalidItems == null)
        {
            return data with { Summary = "No items processed" };
        }

        var successCount = data.ValidItems.Count;
        var errorCount = data.InvalidItems.Count;
        var totalProcessed = data.TotalProcessed;

        var summary = successCount switch
        {
            _ when successCount == totalProcessed && totalProcessed > 0
                => $"All {totalProcessed} items processed successfully",
            _ when successCount == 0 && totalProcessed > 0
                => $"All {totalProcessed} items failed validation",
            _ when successCount > 0 && errorCount > 0
                => $"Partial success: {successCount} valid, {errorCount} invalid (total {totalProcessed})",
            _ => "No items processed"
        };

        return data with
        {
            SuccessCount = successCount,
            ErrorCount = errorCount,
            Summary = summary
        };
    }
}

