using System.Collections.Generic;

namespace GenesisNova.Train;

public sealed record GenesisAutonomousTrainingRequest(
    int MaxRounds = 12,
    int InitialSampleCount = 1,
    int InitialDifficulty = 0,
    int InitialEpochs = 1,
    int InitialTrainCount = 1,
    double LossThreshold = 1.2,
    int MinSampleCount = 1,
    int MaxSampleCount = 128,
    int MinTrainCount = 1,
    int MaxTrainCount = 8,
    int MaxDifficulty = 8,
    int SignalWindow = 4,
    int DifficultyStepUp = 1,
    int DifficultyStepDown = 1,
    int SampleStepUp = 1,
    int SampleStepDown = 1,
    int TrainStepUp = 1,
    int TrainStepDown = 1,
    double TrainRegressionDelta = 0.05,
    double RegressLossMultiplier = 1.25,
    double RegressTokenLossThreshold = 0.15,
    double RegressSpaceNoiseThreshold = 0.45,
    double RegressContradictionThreshold = 0.35,
    double AntiStreakPriorityWindow = 0.75,
    double NewCreatorLossMultiplier = 1.2,
    double NewCreatorBasePriority = 2.5,
    double NewCreatorRecentPenalty = 0.2,
    double MasteryLossMultiplier = 0.75,
    double WeaknessMin = 0.3,
    double WeaknessMax = 2.0,
    double ExplorationBase = 1.2,
    double MasteryPenalty = 2.5,
    double RegressionBoost = 0.35,
    double RecentHitPenalty = 0.9,
    double ConsecutivePenalty = 1.4,
    int RoundTrainBudget = 16,
    int MaxGenerationConcurrency = 4,
    string? PreferredCreator = null,
    IReadOnlyList<string>? EnabledCreators = null,
    // BOOTSTRAP-FIRST (legacy gate, used only when FocusedCurriculum is off): gate broad creators
    // until the "corenova:" primitives are mastered. Safety valve after BootstrapMaxRounds.
    bool BootstrapFirst = true,
    int BootstrapMaxRounds = 8,
    // FOCUSED CURRICULUM (default strategy): train ONE creator to convergence at a time, in
    // complexity order (corenova primitives first), replaying mastered creators for retention —
    // because our experiments showed focused training converges while mixed/composite oscillates.
    // The focus is the first creator that is neither mastered nor focus-exhausted; mastered creators
    // ride along as replay. Re-open is automatic: a regressed creator becomes first-unmastered again.
    // Set false to fall back to the legacy composite (all-creators-mixed) planner.
    bool FocusedCurriculum = true,
    // Per-creator focus budget: after this many attempts without mastering, advance past a creator
    // (which then keeps RIDING ALONG AS REPLAY, never dropped) so an open-ended creator (e.g. a text
    // corpus that never "masters") can't starve the rest of the curriculum. Generous by default
    // because driving a creator to MasteryDifficulty needs room to climb (one difficulty step per
    // successful round + the signal window to confirm), and the run is long.
    int FocusBudget = 12,
    // A prompt-answer creator is not "mastered" until its recent success (which, with
    // RequirePlatonicForCorrect, counts only answers VIA THE PLATONIC PATH) reaches this — so the
    // curriculum won't advance a creator that's merely answering neurally. Windowed-text creators
    // (no platonic path / no prompt-answer success) are exempt.
    double MasterySuccessThreshold = 0.9,
    // DRIVE-TO-DEPTH: a creator is not "mastered" (so the focus does not advance) until it has reached
    // AND succeeded at this difficulty. Without it, mastery fired at difficulty 1 and the focus
    // advanced after trivial competence; the remaining difficulties then only climbed during the
    // all-creators maintenance phase — i.e. as a COMPOSITE, reintroducing the very interference
    // oscillation the focused curriculum exists to avoid. Keeping the focus on one creator while it
    // climbs its difficulties keeps that climb stable. Clamped to [max(InitialDifficulty,1), MaxDifficulty].
    int MasteryDifficulty = 3);
