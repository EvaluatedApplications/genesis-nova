namespace EvalApp.Solid.Starter.Features.AdvancedPatterns;

public sealed record AdvancedMeta(
    string Stage = "Init",
    DateTime? LastUpdatedUtc = null);

public sealed record AdvancedDemoData(
    List<int> InputItems,
    bool ForcePrimaryQuoteFailure = false,
    List<int>? MaterializedItems = null,
    decimal Quote = 0m,
    string QuoteSource = "none",
    string? CpuDigest = null,
    string? SnapshotPath = null,
    AdvancedMeta? Meta = null,
    List<string>? Trace = null,
    int SuccessCount = 0,
    int ErrorCount = 0)
{
    public AdvancedDemoData AppendTrace(string message)
    {
        var trace = Trace is null ? new List<string>() : new List<string>(Trace);
        trace.Add(message);
        return this with { Trace = trace };
    }
}
