namespace GenesisNova.Train;

public sealed class SpaceDecisionJournal
{
    private readonly int _capacity;
    private readonly Queue<SpaceDecisionJournalEntry> _entries = new();

    public SpaceDecisionJournal(int capacity)
        => _capacity = Math.Max(1, capacity);

    public IReadOnlyList<SpaceDecisionJournalEntry> Entries => _entries.ToArray();

    public void Record(SpaceDecisionJournalEntry entry)
    {
        _entries.Enqueue(entry with
        {
            AffectedConcepts = Normalize(entry.AffectedConcepts),
            AccumulatedRetrospectiveReward = Math.Clamp(entry.AccumulatedRetrospectiveReward, -8.0, 8.0)
        });
        while (_entries.Count > _capacity)
            _entries.Dequeue();
    }

    public void ReplaceWith(IEnumerable<SpaceDecisionJournalEntry> entries)
    {
        _entries.Clear();
        foreach (var entry in entries.TakeLast(_capacity))
            Record(entry);
    }

    public IReadOnlyList<SpaceDecisionJournalEntry> FindOverlapping(IReadOnlySet<string> concepts)
        => _entries
            .Where(e => e.AffectedConcepts.Any(concepts.Contains))
            .OrderByDescending(e => e.Step)
            .ToArray();

    public void ApplyRetrospectiveReward(int step, int actionId, double reward)
    {
        if (_entries.Count == 0)
            return;

        var updated = _entries.Select(entry => entry.Step == step && entry.ActionId == actionId
            ? entry with { AccumulatedRetrospectiveReward = Math.Clamp(entry.AccumulatedRetrospectiveReward + reward, -8.0, 8.0) }
            : entry).ToArray();
        _entries.Clear();
        foreach (var entry in updated)
            _entries.Enqueue(entry);
    }

    private static IReadOnlyList<string> Normalize(IReadOnlyList<string> concepts)
        => (concepts ?? Array.Empty<string>())
            .Where(c => !string.IsNullOrWhiteSpace(c))
            .Select(c => c.Trim().ToLowerInvariant())
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
}

public sealed record SpaceDecisionJournalEntry(
    int Step,
    int ActionId,
    string Tool,
    IReadOnlyList<string> AffectedConcepts,
    string StateEncoding,
    double PreMutationNoiseRatio,
    double PreMutationAverageBridgeConfidence,
    double PreMutationRelationPressure,
    int PreMutationNodes,
    int PreMutationRelations,
    double AccumulatedRetrospectiveReward);
