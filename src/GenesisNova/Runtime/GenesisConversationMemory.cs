using GenesisNova.Persistence;

namespace GenesisNova.Runtime;

public sealed class GenesisConversationMemory
{
    private const int RecentTurnLimit = 12;
    private const int SummaryCharLimit = 4096;

    private readonly List<ConversationTurn> _recentTurns = [];
    private string _summary = string.Empty;
    private int _resetCount;
    private int _compactionCount;
    private double _branchTrust = 1.0;
    private DateTimeOffset? _lastResetAtUtc;

    public string Summary => _summary;
    public int ResetCount => _resetCount;
    public int CompactionCount => _compactionCount;
    public double BranchTrust => _branchTrust;
    public DateTimeOffset? LastResetAtUtc => _lastResetAtUtc;
    public IReadOnlyList<(DateTimeOffset TimestampUtc, string Role, string Content, bool IsResetSignal, string? Note)> RecentTurns
        => _recentTurns.Select(t => (t.TimestampUtc, t.Role, t.Content, t.IsResetSignal, t.Note)).ToArray();

    public void ObserveTurn(string role, string content, bool resetSignal = false, string? note = null)
    {
        if (string.IsNullOrWhiteSpace(role) && string.IsNullOrWhiteSpace(content))
            return;

        _recentTurns.Add(new ConversationTurn(
            TimestampUtc: DateTimeOffset.UtcNow,
            Role: NormalizeRole(role),
            Content: NormalizeContent(content),
            IsResetSignal: resetSignal,
            Note: note));

        if (resetSignal)
            RecordReset(note);

        CompactIfNeeded();
    }

    public bool Compact()
    {
        if (_recentTurns.Count == 0)
            return false;

        var compactCount = Math.Max(0, _recentTurns.Count - RecentTurnLimit);
        if (compactCount == 0 && _summary.Length <= SummaryCharLimit)
            return false;

        var compacted = _recentTurns.Take(compactCount).ToArray();
        if (compacted.Length > 0)
            _summary = MergeSummary(_summary, Summarize(compacted));

        if (compactCount > 0)
            _recentTurns.RemoveRange(0, compactCount);

        TrimSummary();
        _compactionCount++;
        return true;
    }

    public string BuildContextBrief()
    {
        var lines = new List<string>();
        if (!string.IsNullOrWhiteSpace(_summary))
        {
            lines.Add("summary:");
            lines.Add(_summary);
        }

        if (_recentTurns.Count > 0)
        {
            lines.Add("recent:");
            lines.AddRange(_recentTurns.Select(turn =>
            {
                var text = Truncate(turn.Content, 160);
                var marker = turn.IsResetSignal ? " reset" : string.Empty;
                return $"- {turn.Role}{marker}: {text}";
            }));
        }

        lines.Add($"state: resets={_resetCount} trust={_branchTrust:F2} compactions={_compactionCount}");
        if (_lastResetAtUtc.HasValue)
            lines.Add($"last-reset: {_lastResetAtUtc:O}");
        return string.Join(Environment.NewLine, lines);
    }

    public GenesisConversationSnapshot ExportSnapshot()
        => new(
            Summary: _summary,
            RecentTurns: _recentTurns
                .Select(turn => new ConversationTurnSnapshot(turn.TimestampUtc, turn.Role, turn.Content, turn.IsResetSignal, turn.Note))
                .ToArray(),
            ResetCount: _resetCount,
            CompactionCount: _compactionCount,
            BranchTrust: _branchTrust,
            LastResetAtUtc: _lastResetAtUtc);

    public void ImportSnapshot(GenesisConversationSnapshot snapshot)
    {
        _summary = snapshot.Summary ?? string.Empty;
        _recentTurns.Clear();
        foreach (var turn in snapshot.RecentTurns)
        {
            _recentTurns.Add(new ConversationTurn(
                turn.TimestampUtc,
                NormalizeRole(turn.Role),
                NormalizeContent(turn.Content),
                turn.IsResetSignal,
                turn.Note));
        }

        _resetCount = Math.Max(0, snapshot.ResetCount);
        _compactionCount = Math.Max(0, snapshot.CompactionCount);
        _branchTrust = Math.Clamp(snapshot.BranchTrust, 0.0, 1.0);
        _lastResetAtUtc = snapshot.LastResetAtUtc;
        TrimSummary();
    }

    private void RecordReset(string? note)
    {
        _resetCount++;
        _lastResetAtUtc = DateTimeOffset.UtcNow;
        _branchTrust = Math.Max(0.0, (_branchTrust * 0.75) - 0.1);
        _summary = MergeSummary(_summary, $"reset signal: {Truncate(note ?? "user reset", 120)}");
        TrimSummary();
        Compact();
    }

    private void CompactIfNeeded()
    {
        var recentChars = _recentTurns.Sum(turn => turn.Content.Length);
        if (_recentTurns.Count <= RecentTurnLimit && recentChars <= SummaryCharLimit)
            return;

        Compact();
    }

    private void TrimSummary()
    {
        if (_summary.Length <= SummaryCharLimit)
            return;

        var lines = _summary
            .Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .ToList();

        while (lines.Count > 0 && string.Join(Environment.NewLine, lines).Length > SummaryCharLimit)
            lines.RemoveAt(0);

        _summary = string.Join(Environment.NewLine, lines);
    }

    private static string MergeSummary(string current, string addition)
    {
        if (string.IsNullOrWhiteSpace(current))
            return addition.Trim();

        return string.Join(Environment.NewLine, [current.TrimEnd(), addition.Trim()]);
    }

    private static string Summarize(IEnumerable<ConversationTurn> turns)
        => string.Join(Environment.NewLine, turns.Select(turn =>
        {
            var resetMarker = turn.IsResetSignal ? " reset" : string.Empty;
            var note = string.IsNullOrWhiteSpace(turn.Note) ? string.Empty : $" ({turn.Note})";
            return $"- {turn.Role}{resetMarker}{note}: {Truncate(turn.Content, 120)}";
        }));

    private static string NormalizeRole(string role)
        => string.IsNullOrWhiteSpace(role) ? "unknown" : role.Trim().ToLowerInvariant();

    private static string NormalizeContent(string content)
        => string.Join(" ", content.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));

    private static string Truncate(string value, int maxLength)
        => value.Length <= maxLength ? value : value[..Math.Max(0, maxLength - 1)] + "…";

    private sealed record ConversationTurn(
        DateTimeOffset TimestampUtc,
        string Role,
        string Content,
        bool IsResetSignal,
        string? Note);

    public async Task DeleteRecentTurnsAsync(int count)
    {
       var removeCount = Math.Min(count, _recentTurns.Count);
       if (removeCount > 0)
           _recentTurns.RemoveRange(Math.Max(0, _recentTurns.Count - removeCount), removeCount);
       await Task.CompletedTask;
    }

    public async Task ClearAsync()
    {
       _recentTurns.Clear();
       _summary = string.Empty;
       _resetCount = 0;
       _compactionCount = 0;
       _branchTrust = 1.0;
       _lastResetAtUtc = null;
       await Task.CompletedTask;
    }
}
