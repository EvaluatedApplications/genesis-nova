using System.Collections.Generic;
using System.Linq;

namespace GenesisNova.Train;

/// <summary>
/// FOCUSED multi-trainer curriculum — the autonomous trainer's proven scheduling, ported: train ONE unit at a
/// time to mastery while MASTERED (and focus-EXHAUSTED) units ride along as light REPLAY; later units wait their
/// turn. "Focused converges where mixed oscillates." Each unit is graded independently (orchestrator
/// <see cref="ITrainingCurriculum.Units"/>) so it tracks its OWN mastery; the focus is the first unit that is
/// neither mastered nor exhausted, and a regressed unit AUTO-REOPENS (mastery flips → first-unmastered again).
/// </summary>
public sealed class FocusedCurriculum : ITrainingCurriculum
{
    private readonly List<FocusUnit> _all;
    private readonly int _replayCap;
    private FocusUnit? _focus;

    public FocusedCurriculum(IEnumerable<ITrainingCurriculum> children, double masteryBar = 0.80,
        int stabilityWindow = 3, int focusBudget = 30, int replayCap = 8)
    {
        _all = children.Select(c => new FocusUnit(c, masteryBar, stabilityWindow, focusBudget)).ToList();
        _replayCap = replayCap;
    }

    public string Name => "focused(" + string.Join(",", _all.Select(u => u.Name)) + ")";
    public int Difficulty => _focus?.Difficulty ?? (_all.Count > 0 ? _all.Max(u => u.Difficulty) : 0);
    public IReadOnlyList<string> OperationTokens => _all.SelectMany(u => u.OperationTokens).Distinct().ToList();

    public IReadOnlyList<(string Input, string Output)> NextTrainBatch()
    {
        // FOCUS = first unit neither mastered nor exhausted (auto-reopen makes a regressed unit first again).
        _focus = _all.FirstOrDefault(u => !u.Mastered && !u.Exhausted);
        foreach (var u in _all) u.IsFocus = u == _focus;
        var batch = new List<(string Input, string Output)>();
        if (_focus is null) { foreach (var u in _all) batch.AddRange(u.NextTrainBatch()); return batch; } // all done → maintenance
        batch.AddRange(_focus.NextTrainBatch());                                                           // the focus trains fully
        foreach (var u in _all)
            if (u != _focus && (u.Mastered || u.Exhausted)) batch.AddRange(u.NextTrainBatch().Take(_replayCap)); // riders: light replay
        return batch;
    }

    // Grade only the ACTIVE set (focus + mastered/exhausted riders) — held-back units don't waste probes; they
    // are unmastered by definition until it's their turn. Read AFTER NextTrainBatch (which sets the focus).
    public IReadOnlyList<ITrainingCurriculum> Units
    {
        get
        {
            var active = new List<ITrainingCurriculum>();
            if (_focus is not null) active.Add(_focus);
            foreach (var u in _all) if (u != _focus && (u.Mastered || u.Exhausted)) active.Add(u);
            return active.Count > 0 ? active : _all.Cast<ITrainingCurriculum>().ToList();
        }
    }

    public IReadOnlyList<TrainingProbe> NextProbes() => System.Array.Empty<TrainingProbe>(); // grading is per-unit
    public void RecordCycle(CycleGrade grade) { }                                            // units record their own
}

/// <summary>Wraps a child curriculum with mastery tracking for <see cref="FocusedCurriculum"/>: MASTERED = held
/// the bar for a stability window AND reached the drive-to-depth difficulty; EXHAUSTED = spent its FocusBudget of
/// focus cycles without mastering (then rides as replay); a regression un-masters it (auto-reopen).</summary>
public sealed class FocusUnit : ITrainingCurriculum
{
    private readonly ITrainingCurriculum _inner;
    private readonly double _bar;
    private readonly int _window, _focusBudget;
    private int _streak;

    public FocusUnit(ITrainingCurriculum inner, double bar, int window, int focusBudget)
    {
        _inner = inner; _bar = bar; _window = window; _focusBudget = focusBudget;
    }

    public bool IsFocus { get; set; }
    public int FocusAttempts { get; private set; }
    public bool Mastered { get; private set; }
    public bool Exhausted => !Mastered && FocusAttempts >= _focusBudget;

    public string Name => _inner.Name;
    public int Difficulty => _inner.Difficulty;
    public IReadOnlyList<string> OperationTokens => _inner.OperationTokens;
    public IReadOnlyList<(string Input, string Output)> NextTrainBatch() => _inner.NextTrainBatch();
    public IReadOnlyList<TrainingProbe> NextProbes() => _inner.NextProbes();

    public void RecordCycle(CycleGrade grade)
    {
        _inner.RecordCycle(grade);                       // inner advances its OWN difficulty (drive-to-depth)
        if (IsFocus && !Mastered) FocusAttempts++;
        if (grade.Accuracy >= _bar)
        {
            _streak++;
            if (!Mastered && _streak >= _window && _inner.Difficulty >= _inner.MasteryDepth) Mastered = true; // drive-to-depth (per unit)
        }
        else
        {
            _streak = 0;
            if (Mastered && grade.Accuracy < _bar - 0.15) Mastered = false; // regression → auto-reopen
        }
    }
}
