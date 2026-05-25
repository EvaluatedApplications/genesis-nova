namespace EvalApp.Solid.Starter.Accounting;

/// <summary>
/// Fetch ItemIds from source (simulated as 1-10 for demo).
/// Pure step: generates test data, no I/O.
/// </summary>
public class FetchItemsStep : PureStep<BatchSyncData>
{
    public override BatchSyncData Execute(BatchSyncData data)
    {
        // If ItemIds already provided, return as-is
        if (data.ItemIds != null && data.ItemIds.Count > 0)
        {
            return data;
        }

        // Generate test ItemIds (1 through 10)
        var itemIds = Enumerable.Range(1, 10).ToList();

        return data with
        {
            ItemIds = itemIds,
            Results = new Dictionary<int, ApiResponse>(),
            FailedIds = new List<int>()
        };
    }
}

