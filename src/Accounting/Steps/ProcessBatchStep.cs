namespace EvalApp.Solid.Starter.Accounting;

/// <summary>
/// Process all items by calling the API for each.
/// AsyncStep: network I/O, demonstrates async iteration with error handling.
///
/// This step:
/// - Iterates over all ItemIds
/// - Calls CallApiAsync for each (with configurable success rate)
/// - Populates Results dict with successful responses
/// - Tracks failed ItemIds
/// - Checks CancellationToken after each API call
/// </summary>
public class ProcessBatchStep : AsyncStep<BatchSyncData>
{
    private readonly double _successRate;
    private readonly int _minDelayMs;
    private readonly int _maxDelayMs;

    public ProcessBatchStep(double successRate = 0.8, int minDelayMs = 10, int maxDelayMs = 100)
    {
        _successRate = successRate;
        _minDelayMs = minDelayMs;
        _maxDelayMs = maxDelayMs;
    }

    public override async ValueTask<BatchSyncData> ExecuteAsync(BatchSyncData data, CancellationToken ct)
    {
        if (data.ItemIds == null || data.ItemIds.Count == 0)
        {
            return data;
        }

        var results = data.Results ?? new Dictionary<int, ApiResponse>();
        var failedIds = data.FailedIds ?? new List<int>();

        // Process each ItemId
        foreach (var itemId in data.ItemIds)
        {
            ct.ThrowIfCancellationRequested();
            
            try
            {
                var response = await CallApiAsync(itemId, ct);
                results[itemId] = response;
            }
            catch
            {
                // Silently track failure - caller can inspect FailedIds
                failedIds.Add(itemId);
            }
        }

        return data with
        {
            Results = results,
            FailedIds = failedIds
        };
    }

    private async Task<ApiResponse> CallApiAsync(int itemId, CancellationToken ct)
    {
        // Deterministic latency keeps tests stable while still simulating variance.
        var delayRange = Math.Max(1, _maxDelayMs - _minDelayMs + 1);
        var delay = _minDelayMs + (itemId % delayRange);
        await Task.Delay(delay, ct);

        // Deterministic routing keeps success/failure scenarios repeatable in CI.
        var randomValue = ((itemId - 1) % 10) / 10.0;

        if (randomValue < _successRate)
        {
            return new ApiResponse(itemId, "Success", $"Processed item {itemId}");
        }
        else if (randomValue < _successRate + 0.15)
        {
            // Simulate timeout
            throw new TimeoutException($"API timeout for item {itemId}");
        }
        else
        {
            // Simulate server error
            throw new InvalidOperationException($"API error for item {itemId}: Internal server error");
        }
    }
}

