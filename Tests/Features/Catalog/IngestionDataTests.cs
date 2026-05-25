using EvalApp.Solid.Starter.Catalog;
using EvalApp.Solid.Starter.Tests.Catalog.Shared;
using Xunit;

namespace EvalApp.Solid.Starter.Tests.Catalog;

public class IngestionDataTests
{
    [Fact]
    public void WhenCreated_Then_HasInitialDefaults()
    {
        // Arrange & Act
        var data = new IngestionData(new List<RawRecord>());

        // Assert
        Assert.NotNull(data.InputStream);
        Assert.Equal(0, data.TotalProcessed);
        Assert.Equal(0, data.SuccessCount);
        Assert.Equal(0, data.ErrorCount);
        Assert.Null(data.Summary);
    }

    [Fact]
    public void WhenMutated_Then_ReturnsNewInstance()
    {
        // Arrange
        var original = IngestionTestData.CreateIngestionData();
        var validItems = new List<ValidatedRecord>
        {
            IngestionTestData.CreateValidatedRecord()
        };

        // Act
        var mutated = original with { ValidItems = validItems };

        // Assert
        Assert.NotSame(original, mutated);
        Assert.Null(original.ValidItems);
        Assert.NotNull(mutated.ValidItems);
        Assert.Single(mutated.ValidItems);
    }

    [Fact]
    public void WhenInputStreamPopulated_Then_PreservesData()
    {
        // Arrange
        var items = new List<RawRecord>
        {
            IngestionTestData.CreateRawRecord(1, "Item1", 100m),
            IngestionTestData.CreateRawRecord(2, "Item2", 200m)
        };
        var data = IngestionTestData.CreateIngestionData(items);

        // Act & Assert
        Assert.Equal(2, data.InputStream.Count);
        Assert.Equal("Item1", data.InputStream[0].Name);
        Assert.Equal(100m, data.InputStream[0].Amount);
    }
}



