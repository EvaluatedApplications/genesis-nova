using System;
using System.Collections.Generic;
using System.Linq;

namespace GenesisNova.Train;

/// <summary>
/// FOCUSED multi-trainer curriculum — the autonomous trainer's proven scheduling, ported: keep EXACTLY ONE unit
/// in focused depth at a time while the already-introduced units RIDE ALONG as light, CAPPED replay. The focus
/// ROTATES: a unit holds focus until it masters OR spends its per-turn budget, then hands off to the next
/// not-yet-mastered unit (round-robin, wrapping). "Focused converges where mixed oscillates."
///
/// The rotation is the whole point for UNBOUNDED units (the gym muscles, which never truly "master"): they hand
/// off by spending a turn, then the focus comes back around and gives them a FRESH turn — so the curriculum
/// SUSTAINS one-at-a-time depth forever instead of latching every muscle "exhausted" and collapsing into a heavy,
/// uncapped all-creators mix (the very "mixed oscillates" regime this exists to avoid). Bounded creator curricula
/// still drive to depth and drop out once genuinely MASTERED; only when ALL units have truly mastered does it go
/// to terminal light rehearsal. Each unit is graded independently (orchestrator <see cref="ITrainingCurriculum.Units"/>)
/// so it tracks its OWN mastery; a regressed (un-mastered) unit re-enters the rotation automatically.
/// </summary>
public sealed class FocusedCurriculum : ITrainingCurriculum
{
    private readonly List<FocusUnit> _all;
    private readonly int _replayCap;
    private readonly int _rehearsalRidersPerCycle;
    private int _riderCursor;
    private int _focusCursor;
    private FocusUnit? _focus;

    public FocusedCurriculum(IEnumerable<ITrainingCurriculum> children, double masteryBar = 0.80,
        int stabilityWindow = 3, int focusBudget = 30, int replayCap = 8, int rehearsalRidersPerCycle = 3,
        bool resuming = false)
    {
        _all = children.Select(c => new FocusUnit(c, masteryBar, stabilityWindow, focusBudget)).ToList();
        _replayCap = replayCap;
        _rehearsalRidersPerCycle = Math.Max(1, rehearsalRidersPerCycle);
        // RESUMING a prior session: every unit was already trained, so mark them all INTRODUCED. Otherwise the probe
        // set (Units = focus + introduced) starts as just the first focus muscle, so the reported accuracy on the
        // first cycles reflects ONE muscle (often a hard one) and only climbs back to the true mix as the rotation
        // re-introduces them — which reads as "training always starts lower than it ended" on every reload. With this,
        // the full trained mix is graded + rehearsed from cycle 1, so the resumed accuracy matches where it left off.
        if (resuming) foreach (var u in _all) u.MarkIntroduced();
    }

    public string Name => "focused(" + string.Join(",", _all.Select(u => u.Name)) + ")";
    public int Difficulty => _focus?.Difficulty ?? (_all.Count > 0 ? _all.Max(u => u.Difficulty) : 0);
    public IReadOnlyList<string> OperationTokens => _all.SelectMany(u => u.OperationTokens).Distinct().ToList();

    public IReadOnlyList<(string Input, string Output)> NextTrainBatch()
    {
        // ROTATE the focus: the current focus keeps depth until it masters or spends its per-turn budget, then we
        // hand off (reset its turn so a later pass gives it a fresh budget) to the next not-yet-mastered unit.
        if (_focus is null || _focus.Mastered || _focus.TurnExhausted)
        {
            _focus?.ResetTurn();                  // hand off — the just-finished focus gets a fresh budget next pass
            _focus = NextWeakest();               // give the next turn to the WEAKEST un-mastered unit (by held-out accuracy)
            _focus?.BeginTurn();                  // marks it introduced + starts its turn counter
        }
        foreach (var u in _all) u.IsFocus = u == _focus;

        var batch = new List<(string Input, string Output)>();
        if (_focus is null)                                                  // every unit genuinely MASTERED → terminal
            { AppendRehearsalWindow(batch, _all); return batch; }            // light, rotating rehearsal (no focus left)
        batch.AddRange(_focus.NextTrainBatch());                             // the focus trains fully (depth) — the majority
        var riders = _all.Where(u => u != _focus && (u.Mastered || u.HasBeenFocused)).ToList(); // introduced/mastered
        AppendRehearsalWindow(batch, riders);                                // only a BOUNDED rotating window of them
        return batch;
    }

    // Rehearse only a bounded, ROTATING WINDOW of the introduced muscles each cycle (not ALL of them): N capped
    // riders summed across many muscles otherwise outweigh the single focus and the batch reads as a full "mix"
    // again. The window rotates so every rider is still rehearsed in turn (retention preserved); combined with the
    // rotating focus, coverage stays complete while the FOCUS remains the clear majority regardless of muscle count.
    private void AppendRehearsalWindow(List<(string Input, string Output)> batch, IReadOnlyList<FocusUnit> riders)
    {
        if (riders.Count == 0) return;
        var window = Math.Min(riders.Count, _rehearsalRidersPerCycle);
        for (var i = 0; i < window; i++)
            batch.AddRange(riders[(_riderCursor + i) % riders.Count].NextTrainBatch().Take(_replayCap));
        _riderCursor = (_riderCursor + window) % riders.Count;
    }

    // WEAKEST-FIRST focus selection (the general, future-proof "give more examples to whatever is struggling"): the
    // next turn goes to the un-mastered unit with the LOWEST recent accuracy. This ONLY works because every unit grades
    // HELD-OUT generalisation (gym holds out members; grammar asserts+recalls never-seen tokens), so low accuracy means
    // genuine inability, not under-exposure — a unit that can recall its trained set but cannot generalise reports LOW
    // and gets prioritised. Never-introduced units come first (no data yet); equal-accuracy ties rotate via a cursor.
    private FocusUnit? NextWeakest()
    {
        var unmastered = _all.Where(u => !u.Mastered).ToList();
        if (unmastered.Count == 0) return null;
        // NEVER-INTRODUCED units come first, in STRICT LIST ORDER — this is the deterministic FOUNDATION order (the
        // caller arranges the list so prerequisites like grammar precede what depends on them). The rotating cursor is
        // ONLY for breaking genuine ties among ALREADY-introduced units; using it here skipped fresh units (after the
        // first handoff the cursor lands past index 0), so a unit placed second could be passed over on the first pass.
        var fresh = unmastered.Where(u => !u.HasBeenFocused).ToList();
        if (fresh.Count > 0) return fresh[0];
        var minAcc = unmastered.Min(u => u.RecentAccuracy);
        var tied = unmastered.Where(u => u.RecentAccuracy <= minAcc + 1e-6).ToList();
        var pick = tied[_focusCursor % tied.Count];
        _focusCursor++;
        return pick;
    }

    // Grade only the ACTIVE set (focus + introduced/mastered riders) — held-back, not-yet-introduced units don't
    // waste probes; they are unmastered by definition until the rotation reaches them. Read AFTER NextTrainBatch.
    public IReadOnlyList<ITrainingCurriculum> Units
    {
        get
        {
            var active = new List<ITrainingCurriculum>();
            if (_focus is not null) active.Add(_focus);
            foreach (var u in _all) if (u != _focus && (u.Mastered || u.HasBeenFocused)) active.Add(u);
            return active.Count > 0 ? active : _all.Cast<ITrainingCurriculum>().ToList();
        }
    }

    public IReadOnlyList<TrainingProbe> NextProbes() => System.Array.Empty<TrainingProbe>(); // grading is per-unit
    public void RecordCycle(CycleGrade grade) { }                                            // units record their own
}

/// <summary>Wraps a child curriculum with mastery tracking for <see cref="FocusedCurriculum"/>: MASTERED = held
/// the bar for a stability window AND reached the drive-to-depth difficulty (then it drops out of the rotation);
/// a TURN ends when the focus spends its per-turn budget without mastering (hands off, then rides as capped
/// replay until the rotation comes back); a regression un-masters it so it re-enters the rotation (auto-reopen).</summary>
public sealed class FocusUnit : ITrainingCurriculum
{
    private readonly ITrainingCurriculum _inner;
    private readonly double _bar;
    private readonly int _window, _focusBudget;
    private int _streak, _turnAttempts;

    public FocusUnit(ITrainingCurriculum inner, double bar, int window, int focusBudget)
    {
        _inner = inner; _bar = bar; _window = window; _focusBudget = focusBudget;
    }

    public bool IsFocus { get; set; }
    public bool HasBeenFocused { get; private set; }            // introduced at least once → eligible to rehearse
    public bool Mastered { get; private set; }
    public double RecentAccuracy { get; private set; }          // EMA of graded (held-out) accuracy — drives weakest-first
    bool ITrainingCurriculum.IsMastered => Mastered;            // surface mastery for the unified progress view
    public bool TurnExhausted => _turnAttempts >= _focusBudget; // this TURN is spent (per-turn, resets on handoff)

    public void BeginTurn() { HasBeenFocused = true; _turnAttempts = 0; } // claim focus: introduce + fresh budget
    public void ResetTurn() => _turnAttempts = 0;                         // hand off: clear the spent turn counter
    public void MarkIntroduced() => HasBeenFocused = true;               // resume: this unit was trained before → already in the mix

    public string Name => _inner.Name;
    public int Difficulty => _inner.Difficulty;
    public IReadOnlyList<string> OperationTokens => _inner.OperationTokens;
    public IReadOnlyList<(string Input, string Output)> NextTrainBatch() => _inner.NextTrainBatch();
    public IReadOnlyList<TrainingProbe> NextProbes() => _inner.NextProbes();

    public void RecordCycle(CycleGrade grade)
    {
        _inner.RecordCycle(grade);                       // inner advances its OWN difficulty (drive-to-depth)
        RecentAccuracy = HasBeenFocused ? 0.5 * RecentAccuracy + 0.5 * grade.Accuracy : grade.Accuracy; // EMA for weakest-first
        if (IsFocus && !Mastered) _turnAttempts++;       // only the focus spends its turn budget
        if (grade.Accuracy >= _bar)
        {
            _streak++;
            if (!Mastered && _streak >= _window && _inner.Difficulty >= _inner.MasteryDepth) Mastered = true; // drive-to-depth (per unit)
        }
        else
        {
            _streak = 0;
            if (Mastered && grade.Accuracy < _bar - 0.15) Mastered = false; // regression → auto-reopen (re-enters rotation)
        }
    }
}
