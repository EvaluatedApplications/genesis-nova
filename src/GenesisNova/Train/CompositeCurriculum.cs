using System.Collections.Generic;
using System.Linq;

namespace GenesisNova.Train;

/// <summary>
/// Runs several curricula at once — the orchestrator drives ONE curriculum, and this composite wraps whichever
/// children are enabled (e.g. checkbox-selected Gym + Memory+Code). Each cycle it interleaves their training
/// batches and unions their probes; the cycle grade is forwarded to every child so each advances its own
/// difficulty. Op-tokens are the union (so an enabled Memory+Code registers find/contains/calls).
/// </summary>
public sealed class CompositeCurriculum : ITrainingCurriculum
{
    private readonly IReadOnlyList<ITrainingCurriculum> _children;

    public CompositeCurriculum(IEnumerable<ITrainingCurriculum> children) => _children = children.ToList();

    public string Name => _children.Count == 0 ? "(none)" : string.Join("+", _children.Select(c => c.Name));
    public int Difficulty => _children.Count == 0 ? 0 : _children.Max(c => c.Difficulty);
    public IReadOnlyList<string> OperationTokens => _children.SelectMany(c => c.OperationTokens).Distinct().ToList();
    public IReadOnlyList<ITrainingCurriculum> Units => _children;   // each child is graded + gated independently
    public IReadOnlyList<(string Input, string Output)> NextTrainBatch() => _children.SelectMany(c => c.NextTrainBatch()).ToList();
    public IReadOnlyList<TrainingProbe> NextProbes() => _children.SelectMany(c => c.NextProbes()).ToList();
    public void RecordCycle(CycleGrade grade) { foreach (var c in _children) c.RecordCycle(grade); }
}
