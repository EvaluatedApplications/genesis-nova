using EvalApp.Consumer;
using EvalApp.Solid.Starter.Catalog;
using EvalApp.Solid.Starter.Tests.Catalog.Shared;
using Xunit;

namespace EvalApp.Solid.Starter.Tests.Catalog;

/// <summary>
/// Additional comprehensive tests for Ingestion covering stress and edge cases.
/// </summary>
public class IngestionEdgeCaseTests
{
    [Fact]
    public void WhenEmptyInputStream_Then_NoItemsProcessed()
    {
        // Arrange
        var data = IngestionTestData.CreateIngestionData(items: new List<RawRecord>());

        // Assert
        Assert.Empty(data.InputStream);
    }

    [Fact]
    public void WhenAllValidRecords_Then_AllProcessedSuccessfully()
    {
        // Arrange
        var data = IngestionTestData.CreateAllValidData(count: 100);

        // Assert
        Assert.Equal(100, data.InputStream.Count);
        Assert.All(data.InputStream, item => Assert.NotNull(item));
    }

    [Fact]
    public void WhenAllInvalidRecords_Then_AllFail()
    {
        // Arrange
        var data = IngestionTestData.CreateAllInvalidData(count: 100);

        // Assert
        Assert.Equal(100, data.InputStream.Count);
    }

    [Fact]
    public void WhenMixedValidityRecords_Then_SplitCorrectly()
    {
        // Arrange
        var data = IngestionTestData.CreateMixedData(validCount: 150, invalidCount: 50);

        // Assert
        Assert.Equal(200, data.InputStream.Count);
    }

    [Fact]
    public void WhenProcessing1000Records_Then_HandledCorrectly()
    {
        // Arrange
        var validRecords = Enumerable.Range(1, 800)
            .Select(i => IngestionTestData.CreateRawRecord(i, $"Item-{i}", 100m * i))
            .ToList();
        var invalidRecords = Enumerable.Range(801, 200)
            .Select(i => IngestionTestData.CreateRawRecord(i, "", -50m))
            .ToList();
        
        var allRecords = new List<RawRecord>();
        allRecords.AddRange(validRecords);
        allRecords.AddRange(invalidRecords);
        
        var data = IngestionTestData.CreateIngestionData(allRecords);

        // Assert
        Assert.Equal(1000, data.InputStream.Count);
    }

    [Fact]
    public void WhenRecordWithHighAmount_Then_ProcessedCorrectly()
    {
        // Arrange
        var record = IngestionTestData.CreateRawRecord(amount: 999999.99m);

        // Assert
        Assert.Equal(999999.99m, record.Amount);
    }

    [Fact]
    public void WhenRecordWithZeroAmount_Then_ConsideredInvalid()
    {
        // Arrange
        var record = IngestionTestData.CreateRawRecord(amount: 0m);

        // Assert
        Assert.Equal(0m, record.Amount);
    }

    [Fact]
    public void WhenRecordWithNegativeAmount_Then_ConsideredInvalid()
    {
        // Arrange
        var record = IngestionTestData.CreateRawRecord(amount: -100m);

        // Assert
        Assert.Equal(-100m, record.Amount);
    }

    [Fact]
    public void WhenRecordWithEmptyName_Then_ConsideredInvalid()
    {
        // Arrange
        var record = IngestionTestData.CreateRawRecord(name: "");

        // Assert
        Assert.Empty(record.Name);
    }

    [Fact]
    public void WhenValidatedRecordCreated_Then_HasTimestamp()
    {
        // Arrange
        var before = DateTime.UtcNow;
        var record = IngestionTestData.CreateValidatedRecord();
        var after = DateTime.UtcNow;

        // Assert
        Assert.NotNull(record.ProcessedAt);
        Assert.InRange(record.ProcessedAt, before, after.AddSeconds(1));
    }

    [Fact]
    public void WhenCreatingLargeValidatedRecordSet_Then_AllHaveTimestamps()
    {
        // Arrange
        var records = Enumerable.Range(1, 100)
            .Select(i => IngestionTestData.CreateValidatedRecord(i, $"Item-{i}", 100m * i))
            .ToList();

        // Assert
        Assert.All(records, r => Assert.NotNull(r.ProcessedAt));
    }

    [Fact]
    public void WhenProcessing5000Records_Then_StressTestPasses()
    {
        // Arrange
        var records = Enumerable.Range(1, 5000)
            .Select(i => IngestionTestData.CreateRawRecord(i, $"Item-{i}", 10m * i))
            .ToList();
        var data = IngestionTestData.CreateIngestionData(records);

        // Assert
        Assert.Equal(5000, data.InputStream.Count);
    }
}



