using EvalApp.Solid.Starter.Features.BatchSync;

namespace EvalApp.Solid.Starter.Tests.Features.BatchSync;

public static class BatchSyncTestData
{
    public static ApiResponse CreateApiResponse(int itemId = 1, string status = "Success", string? message = null)
        => new ApiResponse(itemId, status, message);

    public static BatchSyncData CreateBatchSyncData(List<int>? itemIds = null, Dictionary<int, ApiResponse>? results = null, List<int>? failedIds = null)
    {
        itemIds ??= new List<int>();
        results ??= new Dictionary<int, ApiResponse>();
        failedIds ??= new List<int>();

        return new BatchSyncData(itemIds, results, failedIds);
    }

    public static BatchSyncData CreateWithItemIds(int count = 10)
    {
        var itemIds = Enumerable.Range(1, count).ToList();
        return new BatchSyncData(itemIds);
    }

    public static BatchSyncData CreateWithResults(Dictionary<int, ApiResponse> results)
    {
        return new BatchSyncData(results.Keys.ToList(), results);
    }

    public static BatchSyncData CreateWithFailures(List<int> failedIds)
    {
        var itemIds = failedIds.ToList();
        return new BatchSyncData(itemIds, FailedIds: failedIds);
    }

    public static BatchSyncData CreateWithMixedResults(int successCount = 7, int failureCount = 3)
    {
        var results = new Dictionary<int, ApiResponse>();
        for (int i = 1; i <= successCount; i++)
        {
            results[i] = CreateApiResponse(i, "Success", $"Processed item {i}");
        }

        var failedIds = Enumerable.Range(successCount + 1, failureCount).ToList();
        var allItemIds = Enumerable.Range(1, successCount + failureCount).ToList();

        return new BatchSyncData(allItemIds, results, failedIds);
    }
}
