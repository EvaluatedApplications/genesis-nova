namespace EvalApp.Solid.Starter.Catalog;

/// <summary>
/// ProcessItem — transform validated record into enriched output.
/// Adds ProcessedAt timestamp.
/// Pure step: one responsibility (transformation logic).
/// </summary>
public class ProcessItemStep : PureStep<IngestionData>
{
    public override IngestionData Execute(IngestionData data)
    {
        // This is designed to be called per-item in a ForEach loop
        // Return data as-is; actual transformation happens when pipeline executes
        return data;
    }

    /// <summary>
    /// Transform raw record into validated record with timestamp.
    /// </summary>
    public ValidatedRecord ProcessRecord(RawRecord record)
    {
        return new ValidatedRecord(
            record.Id,
            record.Name,
            record.Amount,
            DateTime.UtcNow);
    }
}

