namespace EvalApp.Solid.Starter.Catalog;

/// <summary>
/// Input record — raw, unvalidated data item from stream.
/// </summary>
public record RawRecord(int Id, string Name, decimal Amount);

/// <summary>
/// Success record — validated and enriched with processing timestamp.
/// </summary>
public record ValidatedRecord(int Id, string Name, decimal Amount, DateTime ProcessedAt);

/// <summary>
/// Error record — validation failure with reason.
/// </summary>
public record ValidationError(int Id, string Reason);

/// <summary>
/// Pipeline data — flowing through ingestion stages.
/// Demonstrates partial success semantics: both ValidItems and InvalidItems are populated.
/// </summary>
public record IngestionData(
    List<RawRecord> InputStream,
    List<ValidatedRecord>? ValidItems = null,
    List<ValidationError>? InvalidItems = null,
    int TotalProcessed = 0,
    int SuccessCount = 0,
    int ErrorCount = 0,
    string? Summary = null);

