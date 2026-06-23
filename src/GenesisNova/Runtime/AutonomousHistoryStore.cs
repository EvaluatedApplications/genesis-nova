using GenesisNova.Persistence;
using GenesisNova.Train;

namespace GenesisNova.Runtime;

internal sealed class AutonomousHistoryStore
{
    private const int MaxHistory = 512;
    private readonly List<GenesisAutonomousTrainingRound> _history = [];

    public IReadOnlyList<GenesisAutonomousTrainingRound> History => _history;

    public void Append(GenesisAutonomousTrainingRound round)
    {
        _history.Add(round);
        if (_history.Count <= MaxHistory)
            return;

        var removeCount = _history.Count - MaxHistory;
        _history.RemoveRange(0, removeCount);
    }

    public void Restore(GenesisAutonomousTrainingSnapshot? snapshot)
    {
        _history.Clear();
        if (snapshot?.History is null || snapshot.History.Length == 0)
            return;

        foreach (var round in snapshot.History.TakeLast(MaxHistory))
            _history.Add(round);
    }

    public GenesisAutonomousTrainingSnapshot Export()
        => new(_history.TakeLast(MaxHistory).ToArray());
}
