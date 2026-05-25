using Xunit;
using EvalApp.Consumer;
using EvalApp.Solid.Starter.Accounting;

namespace EvalApp.Solid.Starter.Tests.Accounting;

public class BatchSyncPipelineTests
{
    [Fact]
    public async Task WhenAllItemsSucceed_Then_ResultsPopulated()
    {
        // Arrange
        var data = BatchSyncTestData.CreateWithItemIds(count: 5);
        var pipeline = BatchSyncPipeline.BuildSimple(successRate: 1.0); // 100% success

        // Act
        var result = await pipeline.RunAsync(data);

        // Assert
        Assert.IsType<PipelineResult<BatchSyncData>.Success>(result);
        var successResult = (PipelineResult<BatchSyncData>.Success)result;
        var finalData = successResult.Data;

        Assert.NotNull(finalData.Results);
        Assert.Equal(5, finalData.Results.Count);
        Assert.Equal(5, finalData.SuccessCount);
        Assert.Empty(finalData.FailedIds ?? new List<int>());
        Assert.Equal(0, finalData.ErrorCount);
    }

    [Fact]
    public async Task WhenSomeItemsFail_Then_PartialResults()
    {
        // Arrange
        var data = BatchSyncTestData.CreateWithItemIds(count: 10);
        var pipeline = BatchSyncPipeline.BuildSimple(successRate: 0.7); // 70% success, 30% failure

        // Act
        var result = await pipeline.RunAsync(data);

        // Assert
        Assert.IsType<PipelineResult<BatchSyncData>.Success>(result);
        var successResult = (PipelineResult<BatchSyncData>.Success)result;
        var finalData = successResult.Data;

        // With 70% success rate, we should have some successes and some failures
        Assert.NotNull(finalData.Results);
        Assert.NotNull(finalData.FailedIds);
        Assert.True(finalData.SuccessCount > 0);
        Assert.True(finalData.ErrorCount > 0);
        Assert.Equal(10, finalData.SuccessCount + finalData.ErrorCount);
    }

    [Fact]
    public async Task WhenAllItemsFail_Then_AllInFailedIds()
    {
        // Arrange
        var data = BatchSyncTestData.CreateWithItemIds(count: 5);
        var pipeline = BatchSyncPipeline.BuildSimple(successRate: 0.0); // 0% success

        // Act
        var result = await pipeline.RunAsync(data);

        // Assert
        Assert.IsType<PipelineResult<BatchSyncData>.Success>(result);
        var successResult = (PipelineResult<BatchSyncData>.Success)result;
        var finalData = successResult.Data;

        Assert.Empty(finalData.Results ?? new Dictionary<int, ApiResponse>());
        Assert.NotNull(finalData.FailedIds);
        Assert.Equal(5, finalData.FailedIds.Count);
        Assert.Equal(0, finalData.SuccessCount);
        Assert.Equal(5, finalData.ErrorCount);
    }

    [Fact]
    public async Task WhenEmptyInput_Then_HandledGracefully()
    {
        // Arrange
        var data = BatchSyncTestData.CreateBatchSyncData(new List<int>());
        var pipeline = BatchSyncPipeline.BuildSimple(successRate: 1.0);

        // Act
        var result = await pipeline.RunAsync(data);

        // Assert
        Assert.IsType<PipelineResult<BatchSyncData>.Success>(result);
        var successResult = (PipelineResult<BatchSyncData>.Success)result;
        var finalData = successResult.Data;

        Assert.Equal(10, finalData.ItemIds.Count);
        Assert.Equal(10, finalData.SuccessCount);
        Assert.Equal(0, finalData.ErrorCount);
    }

    [Fact]
    public async Task WhenItemsProcessed_Then_ResultsContainCorrectData()
    {
        // Arrange
        var data = BatchSyncTestData.CreateWithItemIds(count: 3);
        var pipeline = BatchSyncPipeline.BuildSimple(successRate: 1.0);

        // Act
        var result = await pipeline.RunAsync(data);

        // Assert
        var successResult = (PipelineResult<BatchSyncData>.Success)result;
        var finalData = successResult.Data;

        // Verify each successful result has correct ItemId
        foreach (var (itemId, response) in (Dictionary<int, ApiResponse>)finalData.Results!)
        {
            Assert.Equal(itemId, response.ItemId);
            Assert.Equal("Success", response.Status);
            Assert.NotNull(response.Message);
        }
    }

    [Fact]
    public async Task WhenCancellationRequested_Then_OperationCancelled()
    {
        // Arrange
        var data = BatchSyncTestData.CreateWithItemIds(count: 5);
        var pipeline = BatchSyncPipeline.BuildSimple();
        var cts = new CancellationTokenSource();
        cts.CancelAfter(TimeSpan.FromMilliseconds(1)); // Cancel almost immediately

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(
            async () => await pipeline.RunAsync(data, cts.Token));
    }

    [Fact]
    public async Task WhenLargeItemCount_Then_ProcessesAll()
    {
        // Arrange
        var data = BatchSyncTestData.CreateWithItemIds(count: 100);
        var pipeline = BatchSyncPipeline.BuildSimple(successRate: 0.9, minDelayMs: 1, maxDelayMs: 5);

        // Act
        var result = await pipeline.RunAsync(data);

        // Assert
        var successResult = (PipelineResult<BatchSyncData>.Success)result;
        var finalData = successResult.Data;

        // All 100 items should be processed (either success or failure)
        Assert.Equal(100, finalData.SuccessCount + finalData.ErrorCount);
    }
}

public class BatchSyncStepTests
{
    [Fact]
    public void FetchItemsStep_WhenCalled_GeneratesItemIds()
    {
        // Arrange
        var data = BatchSyncTestData.CreateBatchSyncData(new List<int>());
        var step = new FetchItemsStep();

        // Act
        var result = step.Execute(data);

        // Assert
        Assert.NotNull(result.ItemIds);
        Assert.Equal(10, result.ItemIds.Count);
        Assert.Equal(Enumerable.Range(1, 10), result.ItemIds);
    }

    [Fact]
    public void FetchItemsStep_WhenItemIdsExist_ReturnAsIs()
    {
        // Arrange
        var itemIds = new List<int> { 5, 10, 15 };
        var data = BatchSyncTestData.CreateBatchSyncData(itemIds);
        var step = new FetchItemsStep();

        // Act
        var result = step.Execute(data);

        // Assert
        Assert.Equal(itemIds, result.ItemIds);
    }

    [Fact]
    public void FetchItemsStep_InitializesResultsAndFailedIds()
    {
        // Arrange
        var data = BatchSyncTestData.CreateBatchSyncData(new List<int>());
        var step = new FetchItemsStep();

        // Act
        var result = step.Execute(data);

        // Assert
        Assert.NotNull(result.Results);
        Assert.NotNull(result.FailedIds);
        Assert.Empty(result.Results);
        Assert.Empty(result.FailedIds);
    }

    [Fact]
    public void CalculateSummaryStep_WhenItemsSucceed_CountsCorrectly()
    {
        // Arrange
        var results = new Dictionary<int, ApiResponse>
        {
            { 1, BatchSyncTestData.CreateApiResponse(1) },
            { 2, BatchSyncTestData.CreateApiResponse(2) },
            { 3, BatchSyncTestData.CreateApiResponse(3) }
        };
        var data = new BatchSyncData(new List<int> { 1, 2, 3, 4, 5 }, results, new List<int> { 4, 5 });
        var step = new CalculateSummaryStep();

        // Act
        var result = step.Execute(data);

        // Assert
        Assert.Equal(3, result.SuccessCount);
        Assert.Equal(2, result.ErrorCount);
    }

    [Fact]
    public void CalculateSummaryStep_WhenNoResults_CountsAsZero()
    {
        // Arrange
        var data = new BatchSyncData(new List<int> { 1, 2, 3 }, null, null);
        var step = new CalculateSummaryStep();

        // Act
        var result = step.Execute(data);

        // Assert
        Assert.Equal(0, result.SuccessCount);
        Assert.Equal(0, result.ErrorCount);
    }

    [Fact]
    public void CalculateSummaryStep_WhenAllFail_CountsAllAsErrors()
    {
        // Arrange
        var data = new BatchSyncData(
            new List<int> { 1, 2, 3 },
            new Dictionary<int, ApiResponse>(),
            new List<int> { 1, 2, 3 });
        var step = new CalculateSummaryStep();

        // Act
        var result = step.Execute(data);

        // Assert
        Assert.Equal(0, result.SuccessCount);
        Assert.Equal(3, result.ErrorCount);
    }
}

public class BatchSyncDataTests
{
    [Fact]
    public void BatchSyncData_CanBeMutatedWithExpression()
    {
        // Arrange
        var original = new BatchSyncData(new List<int> { 1, 2, 3 });

        // Act
        var updated = original with
        {
            Results = new Dictionary<int, ApiResponse> { { 1, new ApiResponse(1, "Success") } },
            SuccessCount = 1
        };

        // Assert
        Assert.Empty(original.Results ?? new Dictionary<int, ApiResponse>());
        Assert.NotEmpty(updated.Results);
        Assert.Equal(0, original.SuccessCount);
        Assert.Equal(1, updated.SuccessCount);
    }

    [Fact]
    public void BatchSyncData_PreservesUnchangedFields()
    {
        // Arrange
        var itemIds = new List<int> { 1, 2, 3 };
        var original = new BatchSyncData(itemIds, SuccessCount: 5);

        // Act
        var updated = original with { ErrorCount = 2 };

        // Assert
        Assert.Equal(itemIds, updated.ItemIds);
        Assert.Equal(5, updated.SuccessCount);
        Assert.Equal(2, updated.ErrorCount);
    }

    [Fact]
    public void BatchSyncData_DefaultValues()
    {
        // Arrange & Act
        var data = new BatchSyncData(new List<int>());

        // Assert
        Assert.Null(data.Results);
        Assert.Null(data.FailedIds);
        Assert.Equal(0, data.SuccessCount);
        Assert.Equal(0, data.ErrorCount);
    }

    [Fact]
    public void ApiResponse_CanBeCreated()
    {
        // Arrange & Act
        var response = new ApiResponse(42, "Success", "Processed successfully");

        // Assert
        Assert.Equal(42, response.ItemId);
        Assert.Equal("Success", response.Status);
        Assert.Equal("Processed successfully", response.Message);
    }
}



