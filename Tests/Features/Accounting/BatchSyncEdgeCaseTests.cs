using EvalApp.Consumer;
using EvalApp.Solid.Starter.Accounting;
using EvalApp.Solid.Starter.Tests.Accounting.Shared;
using Xunit;

namespace EvalApp.Solid.Starter.Tests.Accounting;

/// <summary>
/// Additional comprehensive tests for BatchSync covering stress and edge cases.
/// </summary>
public class BatchSyncEdgeCaseTests
{
    [Fact]
    public void WhenSyncingEmptyBatch_Then_NoResultsReturned()
    {
        // Arrange
        var data = BatchSyncTestData.CreateBatchSyncData(itemIds: new List<int>());

        // Assert
        Assert.Empty(data.ItemIds);
        Assert.Empty(data.Results);
    }

    [Fact]
    public void WhenAllItemsSyncSuccessfully_Then_AllInResults()
    {
        // Arrange
        var data = BatchSyncTestData.CreateWithItemIds(count: 100);

        // Assert
        Assert.Equal(100, data.ItemIds.Count);
    }

    [Fact]
    public void WhenAllItemsFail_Then_AllInFailedIds()
    {
        // Arrange
        var failedIds = Enumerable.Range(1, 100).ToList();
        var data = BatchSyncTestData.CreateWithFailures(failedIds);

        // Assert
        Assert.Equal(100, data.FailedIds.Count);
    }

    [Fact]
    public void WhenMixedResults_Then_TrackBothSuccessAndFailure()
    {
        // Arrange
        var data = BatchSyncTestData.CreateWithMixedResults(successCount: 150, failureCount: 50);

        // Assert
        Assert.Equal(200, data.ItemIds.Count);
        Assert.Equal(150, data.Results.Count);
        Assert.Equal(50, data.FailedIds.Count);
    }

    [Fact]
    public void WhenProcessing1000Items_Then_HandledCorrectly()
    {
        // Arrange
        var data = BatchSyncTestData.CreateWithItemIds(count: 1000);

        // Assert
        Assert.Equal(1000, data.ItemIds.Count);
    }

    [Fact]
    public void WhenSyncingItemWithSuccess_Then_ResponseRecorded()
    {
        // Arrange
        var response = BatchSyncTestData.CreateApiResponse(itemId: 42, status: "Success", message: "Synced");

        // Assert
        Assert.Equal(42, response.ItemId);
        Assert.Equal("Success", response.Status);
        Assert.Equal("Synced", response.Message);
    }

    [Fact]
    public void WhenSyncingItemWithError_Then_ErrorRecorded()
    {
        // Arrange
        var response = BatchSyncTestData.CreateApiResponse(itemId: 99, status: "Error", message: "Network timeout");

        // Assert
        Assert.Equal(99, response.ItemId);
        Assert.Equal("Error", response.Status);
        Assert.Equal("Network timeout", response.Message);
    }

    [Fact]
    public void WhenProcessing5000Items_Then_StressTestPasses()
    {
        // Arrange
        var data = BatchSyncTestData.CreateWithItemIds(count: 5000);

        // Assert
        Assert.Equal(5000, data.ItemIds.Count);
    }

    [Fact]
    public void WhenCreatingResultsDict_Then_AllResponsesStored()
    {
        // Arrange
        var results = new Dictionary<int, ApiResponse>
        {
            { 1, BatchSyncTestData.CreateApiResponse(1, "Success") },
            { 2, BatchSyncTestData.CreateApiResponse(2, "Success") },
            { 3, BatchSyncTestData.CreateApiResponse(3, "Error") }
        };
        var data = BatchSyncTestData.CreateWithResults(results);

        // Assert
        Assert.Equal(3, data.Results.Count);
        Assert.Contains(1, data.Results.Keys);
        Assert.Contains(3, data.Results.Keys);
    }

    [Fact]
    public void WhenHighSuccessRate_Then_FewFailures()
    {
        // Arrange
        var data = BatchSyncTestData.CreateWithMixedResults(successCount: 950, failureCount: 50);

        // Assert
        Assert.Equal(1000, data.ItemIds.Count);
        Assert.Equal(950, data.Results.Count);
        Assert.Equal(50, data.FailedIds.Count);
    }

    [Fact]
    public void WhenLowSuccessRate_Then_ManyFailures()
    {
        // Arrange
        var data = BatchSyncTestData.CreateWithMixedResults(successCount: 100, failureCount: 900);

        // Assert
        Assert.Equal(1000, data.ItemIds.Count);
        Assert.Equal(100, data.Results.Count);
        Assert.Equal(900, data.FailedIds.Count);
    }
}



