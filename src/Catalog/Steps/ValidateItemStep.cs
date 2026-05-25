namespace EvalApp.Solid.Starter.Catalog;

/// <summary>
/// ValidateItem — verify constraints on individual record.
/// Rules:
///   - Name must not be empty
///   - Amount must be greater than 0
/// 
/// Pure step: one responsibility (validation logic).
/// Returns data unchanged if valid, or collects error if invalid.
/// </summary>
public class ValidateItemStep : PureStep<IngestionData>
{
    public override IngestionData Execute(IngestionData data)
    {
        // This is designed to be called per-item in a ForEach loop
        // For now, return data as-is; the pipeline will iterate and call this per-item
        // Actual validation happens when the pipeline executes with items
        return data;
    }

    /// <summary>
    /// Validate a single raw record. Returns error reason if invalid, null if valid.
    /// </summary>
    public string? ValidateRecord(RawRecord record)
    {
        if (string.IsNullOrWhiteSpace(record.Name))
            return "Name cannot be empty";

        if (record.Amount <= 0)
            return "Amount must be greater than zero";

        return null;
    }
}

