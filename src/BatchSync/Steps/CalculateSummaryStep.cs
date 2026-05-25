namespace EvalApp.Solid.Starter.Features.BatchSync;

/// <summary>
/// Calculate success/error counts based on Results and FailedIds.
/// Pure step: analyzes data, no I/O.
/// </summary>
public class CalculateSummaryStep : PureStep<BatchSyncData>
{
    public override BatchSyncData Execute(BatchSyncData data)
    {
        var successCount = data.Results?.Count ?? 0;
        var errorCount = data.FailedIds?.Count ?? 0;

        return data with
        {
            SuccessCount = successCount,
            ErrorCount = errorCount
        };
    }
}
