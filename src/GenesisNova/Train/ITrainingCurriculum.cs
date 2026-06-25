using System.Collections.Generic;

namespace GenesisNova.Train;

/// <summary>
/// A pluggable CURRICULUM source for the <see cref="GenesisModularTrainingOrchestrator"/>: it decides WHAT to
/// train next and owns its own difficulty/mastery state. The orchestrator owns the loop (train → probe → grade →
/// throttle), the engine owns per-example mechanics + batch recovery; the curriculum owns the lesson sequence.
/// Implementations: the procedural gym (<see cref="GymTrainer"/>), creator-driven curricula, fact sets.
/// </summary>
public interface ITrainingCurriculum
{
    /// <summary>Short label for telemetry (e.g. "gym").</summary>
    string Name { get; }

    /// <summary>Current difficulty/level/depth — surfaced in metrics; climbs as the curriculum is mastered.</summary>
    int Difficulty { get; }

    /// <summary>This cycle's training examples (input → output). Includes any rehearsal mix the curriculum wants.</summary>
    IReadOnlyList<(string Input, string Output)> NextTrainBatch();

    /// <summary>This cycle's held-out / retention probes to grade mastery.</summary>
    IReadOnlyList<TrainingProbe> NextProbes();

    /// <summary>Feedback from the cycle's grade — the curriculum advances difficulty / updates rehearsal here.</summary>
    void RecordCycle(CycleGrade grade);

    /// <summary>Op-token route-trigger verbs to register on the runtime (excluded from relation-edge formation,
    /// e.g. find/contains/calls). Registered once when the orchestrator attaches. Default: none.</summary>
    IReadOnlyList<string> OperationTokens => System.Array.Empty<string>();

    /// <summary>The independently-gated training UNITS. Each unit is probed + graded SEPARATELY and advances its
    /// OWN difficulty/mastery on its OWN score — so a composite of several curricula gates each one independently.
    /// A leaf curriculum is a single unit (itself); a composite returns its children.</summary>
    IReadOnlyList<ITrainingCurriculum> Units => new[] { this };

    /// <summary>The <see cref="Difficulty"/> this unit must reach (DRIVE-TO-DEPTH) before FocusedCurriculum marks
    /// it mastered. Open-ended/flat units use 1; a creator uses its max difficulty so the focus climbs first.</summary>
    int MasteryDepth => 1;

    /// <summary>Whether this unit has reached mastery (held the bar). Surfaced for the unified progress view so every
    /// lesson reports a coherent state; default false (a leaf that doesn't track its own mastery).</summary>
    bool IsMastered => false;
}

/// <summary>A probe: a query plus the FULL set of valid answers (fuzzy full-list grading) and how many distinct
/// valid answers are required at the current difficulty (1 for a single-answer cue).</summary>
public readonly record struct TrainingProbe(
    string Query, IReadOnlyList<string> Allowed, int RequiredDepth,
    IReadOnlyList<string>? AnswerVocabulary = null, bool RequirePlatonic = true, bool SurfaceStrict = false);

/// <summary>The graded result of one cycle, fed back to the curriculum.</summary>
public readonly record struct CycleGrade(double Accuracy, double RoutePurity, double Confidence);

/// <summary>Per-cycle telemetry emitted by the orchestrator (UI curve / control endpoint consume this).</summary>
public readonly record struct CycleMetrics(
    int Cycle, int Difficulty, double Loss, double Accuracy, double RoutePurity, double Confidence, double CycleSeconds,
    IReadOnlyList<ProbeSample> Samples, int TrainedCount = 0, int GeneratedCount = 0,
    IReadOnlyList<long>? OpClassBalance = null,                  // op head window [abstain,add,sub,mul,div] — collapse visible
    IReadOnlyDictionary<string, double>? ModuleMetrics = null,  // per-learning-module activity counters
    IReadOnlyList<UnitProgress>? Units = null);                 // UNIFIED per-lesson progress (every muscle + persona)

/// <summary>One lesson's live progress, for the unified leveling view: which lesson, its current level, this cycle's
/// accuracy, and whether it has mastered. Lets the host show the WHOLE picture (all sub-lessons at once) instead of a
/// single conflated level.</summary>
public readonly record struct UnitProgress(string Name, int Level, double Accuracy, bool Mastered);

/// <summary>A periodic diagnostic sample — a probe query, the model's output, whether it was correct, and
/// whether it routed platonic (vs neural fallback).</summary>
public readonly record struct ProbeSample(string Query, string Output, bool Correct, bool Platonic, string Expected = "", bool ValueCorrect = false);
