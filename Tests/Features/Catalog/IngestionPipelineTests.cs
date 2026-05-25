using EvalApp.Consumer;
using EvalApp.Solid.Starter.Catalog;
using EvalApp.Solid.Starter.Tests.Catalog.Shared;
using Xunit;

namespace EvalApp.Solid.Starter.Tests.Catalog;

public class IngestionPipelineTests
{
    [Fact]
    public async Task WhenAllValid_Then_AllProcessedSuccessfully()
    {
        // Arrange
        var data = IngestionTestData.CreateAllValidData(count: 5);
        var pipeline = IngestionPipeline.Build();

        // Act
        var result = await pipeline.RunAsync(data);
        var finalData = ((PipelineResult<IngestionData>.Success)result).Data;

        // Assert
        Assert.Equal(5, finalData.ValidItems?.Count);
        Assert.Empty(finalData.InvalidItems ?? new List<ValidationError>());
        Assert.Equal(5, finalData.SuccessCount);
        Assert.Equal(0, finalData.ErrorCount);
        Assert.Contains("All 5", finalData.Summary);
    }

    [Fact]
    public async Task WhenAllInvalid_Then_AllFailedValidation()
    {
        // Arrange
        var data = IngestionTestData.CreateAllInvalidData(count: 4);
        var pipeline = IngestionPipeline.Build();

        // Act
        var result = await pipeline.RunAsync(data);
        var finalData = ((PipelineResult<IngestionData>.Success)result).Data;

        // Assert
        Assert.Empty(finalData.ValidItems ?? new List<ValidatedRecord>());
        Assert.Equal(4, finalData.InvalidItems?.Count);
        Assert.Equal(0, finalData.SuccessCount);
        Assert.Equal(4, finalData.ErrorCount);
        Assert.Contains("All 4", finalData.Summary);
    }

    [Fact]
    public async Task WhenSomeInvalid_Then_ValidAndInvalidLists()
    {
        // Arrange
        var data = IngestionTestData.CreateMixedData(validCount: 3, invalidCount: 2);
        var pipeline = IngestionPipeline.Build();

        // Act
        var result = await pipeline.RunAsync(data);
        var finalData = ((PipelineResult<IngestionData>.Success)result).Data;

        // Assert
        Assert.Equal(3, finalData.ValidItems?.Count);
        Assert.Equal(2, finalData.InvalidItems?.Count);
        Assert.Equal(3, finalData.SuccessCount);
        Assert.Equal(2, finalData.ErrorCount);
        Assert.Contains("Partial success: 3 valid, 2 invalid", finalData.Summary);
    }

    [Fact]
    public async Task WhenValidItems_Then_ContainsProcessedAtTimestamp()
    {
        // Arrange
        var beforeTime = DateTime.UtcNow;
        var data = IngestionTestData.CreateAllValidData(count: 2);
        var pipeline = IngestionPipeline.Build();

        // Act
        var result = await pipeline.RunAsync(data);
        var finalData = ((PipelineResult<IngestionData>.Success)result).Data;
        var afterTime = DateTime.UtcNow;

        // Assert
        Assert.All(finalData.ValidItems!, item =>
        {
            Assert.True(item.ProcessedAt >= beforeTime);
            Assert.True(item.ProcessedAt <= afterTime);
        });
    }

    [Fact]
    public async Task WhenInvalidItems_Then_ContainsErrorReasons()
    {
        // Arrange
        var data = new IngestionData(new List<RawRecord>
        {
            new RawRecord(1, "", 100m),  // Empty name
            new RawRecord(2, "Valid", -50m)  // Negative amount
        });
        var pipeline = IngestionPipeline.Build();

        // Act
        var result = await pipeline.RunAsync(data);
        var finalData = ((PipelineResult<IngestionData>.Success)result).Data;

        // Assert
        Assert.Equal(2, finalData.InvalidItems?.Count);
        Assert.Contains(finalData.InvalidItems, e => e.Reason.Contains("Name"));
        Assert.Contains(finalData.InvalidItems, e => e.Reason.Contains("greater than zero"));
    }

    [Fact]
    public async Task WhenEmptyStream_Then_ZeroProcessed()
    {
        // Arrange
        var data = IngestionTestData.CreateIngestionData(new List<RawRecord>());
        var pipeline = IngestionPipeline.Build();

        // Act
        var result = await pipeline.RunAsync(data);
        var finalData = ((PipelineResult<IngestionData>.Success)result).Data;

        // Assert
        Assert.Equal(0, finalData.SuccessCount);
        Assert.Equal(0, finalData.ErrorCount);
        Assert.Contains("No items processed", finalData.Summary);
    }

    [Fact]
    public async Task WhenNameEmpty_Then_ValidationFails()
    {
        // Arrange
        var data = new IngestionData(new List<RawRecord>
        {
            new RawRecord(1, "", 100m)
        });
        var pipeline = IngestionPipeline.Build();

        // Act
        var result = await pipeline.RunAsync(data);
        var finalData = ((PipelineResult<IngestionData>.Success)result).Data;

        // Assert
        Assert.Empty(finalData.ValidItems ?? new List<ValidatedRecord>());
        Assert.Single(finalData.InvalidItems!);
        Assert.Contains("Name cannot be empty", finalData.InvalidItems[0].Reason);
    }

    [Fact]
    public async Task WhenAmountZero_Then_ValidationFails()
    {
        // Arrange
        var data = new IngestionData(new List<RawRecord>
        {
            new RawRecord(1, "Item", 0m)
        });
        var pipeline = IngestionPipeline.Build();

        // Act
        var result = await pipeline.RunAsync(data);
        var finalData = ((PipelineResult<IngestionData>.Success)result).Data;

        // Assert
        Assert.Empty(finalData.ValidItems ?? new List<ValidatedRecord>());
        Assert.Single(finalData.InvalidItems!);
        Assert.Contains("greater than zero", finalData.InvalidItems[0].Reason);
    }

    [Fact]
    public async Task WhenAmountNegative_Then_ValidationFails()
    {
        // Arrange
        var data = new IngestionData(new List<RawRecord>
        {
            new RawRecord(1, "Item", -100m)
        });
        var pipeline = IngestionPipeline.Build();

        // Act
        var result = await pipeline.RunAsync(data);
        var finalData = ((PipelineResult<IngestionData>.Success)result).Data;

        // Assert
        Assert.Empty(finalData.ValidItems ?? new List<ValidatedRecord>());
        Assert.Single(finalData.InvalidItems!);
        Assert.Contains("greater than zero", finalData.InvalidItems[0].Reason);
    }

    [Fact]
    public async Task WhenValidNames_Then_PreservesInOutput()
    {
        // Arrange
        var data = new IngestionData(new List<RawRecord>
        {
            new RawRecord(1, "Apple", 50m),
            new RawRecord(2, "Banana", 75m)
        });
        var pipeline = IngestionPipeline.Build();

        // Act
        var result = await pipeline.RunAsync(data);
        var finalData = ((PipelineResult<IngestionData>.Success)result).Data;

        // Assert
        Assert.Equal(2, finalData.ValidItems?.Count);
        Assert.Equal("Apple", finalData.ValidItems![0].Name);
        Assert.Equal("Banana", finalData.ValidItems![1].Name);
    }
}

public class IngestionStepTests
{
    [Fact]
    public void MaterializeStep_WhenCalled_InitializesCollections()
    {
        // Arrange
        var data = IngestionTestData.CreateIngestionData(new List<RawRecord> { IngestionTestData.CreateRawRecord() });
        var step = new MaterializeStep();

        // Act
        var result = step.Execute(data);

        // Assert
        Assert.NotNull(result.ValidItems);
        Assert.NotNull(result.InvalidItems);
        Assert.Empty(result.ValidItems);
        Assert.Empty(result.InvalidItems);
        Assert.Equal(1, result.TotalProcessed);
    }

    [Fact]
    public void ValidateItemStep_WhenValidRecord_ReturnsNull()
    {
        // Arrange
        var step = new ValidateItemStep();
        var record = IngestionTestData.CreateRawRecord(1, "Valid", 100m);

        // Act
        var errorReason = step.ValidateRecord(record);

        // Assert
        Assert.Null(errorReason);
    }

    [Fact]
    public void ValidateItemStep_WhenEmptyName_ReturnError()
    {
        // Arrange
        var step = new ValidateItemStep();
        var record = new RawRecord(1, "", 100m);

        // Act
        var errorReason = step.ValidateRecord(record);

        // Assert
        Assert.NotNull(errorReason);
        Assert.Contains("Name", errorReason);
    }

    [Fact]
    public void ValidateItemStep_WhenNegativeAmount_ReturnError()
    {
        // Arrange
        var step = new ValidateItemStep();
        var record = new RawRecord(1, "Item", -50m);

        // Act
        var errorReason = step.ValidateRecord(record);

        // Assert
        Assert.NotNull(errorReason);
        Assert.Contains("greater than zero", errorReason);
    }

    [Fact]
    public void ProcessItemStep_WhenCalled_TransformsRecord()
    {
        // Arrange
        var step = new ProcessItemStep();
        var record = IngestionTestData.CreateRawRecord(1, "Item", 100m);
        var beforeTime = DateTime.UtcNow;

        // Act
        var result = step.ProcessRecord(record);
        var afterTime = DateTime.UtcNow;

        // Assert
        Assert.Equal(1, result.Id);
        Assert.Equal("Item", result.Name);
        Assert.Equal(100m, result.Amount);
        Assert.True(result.ProcessedAt >= beforeTime);
        Assert.True(result.ProcessedAt <= afterTime);
    }

    [Fact]
    public void SummarizeResultsStep_WhenAllValid_ReturnsSummary()
    {
        // Arrange
        var data = new IngestionData(
            new List<RawRecord> { IngestionTestData.CreateRawRecord() },
            ValidItems: new List<ValidatedRecord> { IngestionTestData.CreateValidatedRecord() },
            InvalidItems: new List<ValidationError>(),
            TotalProcessed: 1,
            SuccessCount: 0,
            ErrorCount: 0);
        var step = new SummarizeResultsStep();

        // Act
        var result = step.Execute(data);

        // Assert
        Assert.NotNull(result.Summary);
        Assert.Contains("All 1", result.Summary);
        Assert.Equal(1, result.SuccessCount);
        Assert.Equal(0, result.ErrorCount);
    }

    [Fact]
    public void ProcessAllItemsStep_WhenMixed_PopulatesBothCollections()
    {
        // Arrange
        var data = new IngestionData(new List<RawRecord>
        {
            new RawRecord(1, "Valid", 100m),
            new RawRecord(2, "", 100m),
            new RawRecord(3, "Also Valid", 50m)
        });
        var step = new ProcessAllItemsStep();

        // Act
        var result = step.Execute(data);

        // Assert
        Assert.Equal(2, result.ValidItems?.Count);
        Assert.Single(result.InvalidItems!);
        Assert.Equal("Valid", result.ValidItems![0].Name);
        Assert.Equal("Also Valid", result.ValidItems![1].Name);
    }
}



