using EvalApp.Solid.Starter.Features.Ingestion;

namespace EvalApp.Solid.Starter.Tests.Features.Ingestion;

public static class IngestionTestData
{
    public static RawRecord CreateRawRecord(int id = 1, string name = "Test Item", decimal amount = 100m)
        => new RawRecord(id, name, amount);

    public static ValidatedRecord CreateValidatedRecord(int id = 1, string name = "Test Item", decimal amount = 100m)
        => new ValidatedRecord(id, name, amount, DateTime.UtcNow);

    public static ValidationError CreateValidationError(int id = 1, string reason = "Invalid")
        => new ValidationError(id, reason);

    public static IngestionData CreateIngestionData(List<RawRecord>? items = null)
    {
        items ??= new List<RawRecord>();
        return new IngestionData(items);
    }

    public static IngestionData CreateAllValidData(int count = 5)
    {
        var items = Enumerable.Range(1, count)
            .Select(i => CreateRawRecord(i, $"Item-{i}", 100m * i))
            .ToList();
        return CreateIngestionData(items);
    }

    public static IngestionData CreateAllInvalidData(int count = 5)
    {
        var items = new List<RawRecord>();
        for (int i = 0; i < count; i++)
        {
            if (i % 2 == 0)
            {
                // Empty name
                items.Add(new RawRecord(i + 1, "", 100m));
            }
            else
            {
                // Negative amount
                items.Add(new RawRecord(i + 1, $"Item-{i + 1}", -50m));
            }
        }
        return CreateIngestionData(items);
    }

    public static IngestionData CreateMixedData(int validCount = 3, int invalidCount = 2)
    {
        var items = new List<RawRecord>();

        // Add valid items
        for (int i = 1; i <= validCount; i++)
        {
            items.Add(CreateRawRecord(i, $"Valid-{i}", 100m * i));
        }

        // Add invalid items
        for (int i = 1; i <= invalidCount; i++)
        {
            int id = validCount + i;
            if (i % 2 == 1)
            {
                // Empty name
                items.Add(new RawRecord(id, "", 100m));
            }
            else
            {
                // Negative amount
                items.Add(new RawRecord(id, $"Invalid-{i}", -50m));
            }
        }

        return CreateIngestionData(items);
    }
}
