using GenesisNova.Data;

namespace GenesisNova.Train;

public sealed class GenesisAutonomousTrainingPlanner
{
    private const int SampleRecommendationMultiplier = 2;
    private readonly IReadOnlyList<IExampleCreator> _creators;
    private sealed record CreatorSignal(
        string CreatorName,
        int Attempts,
        double AvgLoss,
        double SuccessRate,
        double RecentSuccessRate,
        int MaxDifficulty,
        int LastSeenCount,
        int RecentHits,
        int ConsecutiveRecentRounds,
        bool Mastered,
        bool Regressing,
        double Priority);

    public GenesisAutonomousTrainingPlanner(IReadOnlyList<IExampleCreator>? creators = null)
    {
        _creators = (creators ?? ExampleCreatorRegistry.All)
            .OrderBy(c => c.EstimatedComplexity)
            .ThenBy(c => c.Name, StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    public GenesisAutonomousTrainingPlan Suggest(
        GenesisAutonomousTrainingRequest request,
        IReadOnlyList<GenesisAutonomousTrainingRound> history)
    {
        var activeCreators = ResolveActiveCreators(request);

        var round = history.Count;
        var signals = BuildSignals(request, history, activeCreators);

        var creator = ChooseCreator(request, signals, history, activeCreators);
        var creatorHistory = history
            .Where(h => h.CreatorName.Equals(creator.Name, StringComparison.OrdinalIgnoreCase))
            .ToArray();
        var lastForCreator = creatorHistory.Length > 0 ? creatorHistory[^1] : null;
        var creatorSignal = signals.FirstOrDefault(s => s.CreatorName.Equals(creator.Name, StringComparison.OrdinalIgnoreCase));

        var difficulty = ChooseDifficulty(request, lastForCreator, creatorSignal);
        var sampleCount = ChooseSampleCount(request, lastForCreator, creatorSignal);
        var trainCount = ChooseTrainCount(request, creatorHistory, creatorSignal);
        var epochs = ChooseEpochs(request, lastForCreator, creatorSignal);
        var reason = BuildReason(request, round, signals, creator.Name, sampleCount, trainCount, difficulty, epochs);

        return new GenesisAutonomousTrainingPlan(
            CreatorName: creator.Name,
            SampleCount: sampleCount,
            TrainCount: trainCount,
            Difficulty: difficulty,
            Epochs: epochs,
            Reason: reason);
    }

    public GenesisAutonomousCompositePlan SuggestComposite(
        GenesisAutonomousTrainingRequest request,
        IReadOnlyList<GenesisAutonomousTrainingRound> history,
        int roundIndex)
    {
        var activeCreators = ResolveActiveCreators(request);

        var signals = BuildSignals(request, history, activeCreators)
            .ToDictionary(s => s.CreatorName, StringComparer.OrdinalIgnoreCase);

        // CURRICULUM STRATEGY. Default: FOCUSED — train ONE creator to convergence at a time
        // (complexity order, corenova first) + replay mastered, since focused converges where
        // composite/mixed oscillates. Legacy: bootstrap-first gate, then composite-all.
        activeCreators = request.FocusedCurriculum
            ? ApplyFocusedCurriculum(request, activeCreators, signals)
            : ApplyBootstrapFirstGate(request, roundIndex, activeCreators, signals);

        var plans = new List<GenesisAutonomousCreatorPlan>(activeCreators.Count);
        foreach (var creator in activeCreators)
        {
            var creatorHistory = history
                .Where(h => h.CreatorName.Equals(creator.Name, StringComparison.OrdinalIgnoreCase))
                .ToArray();
            var lastForCreator = creatorHistory.Length > 0 ? creatorHistory[^1] : null;
            var signal = signals.TryGetValue(creator.Name, out var current) ? current : null;
            var sampleCount = ChooseSampleCount(request, lastForCreator, signal);
            var difficulty = ChooseDifficulty(request, lastForCreator, signal);
            var suggestedTrain = ChooseTrainCount(request, creatorHistory, signal);
            var epochs = ChooseEpochs(request, lastForCreator, signal);
            var priority = signal?.Priority ?? request.NewCreatorBasePriority;
            var reason = BuildReason(
                request,
                roundIndex,
                signals.Values.ToArray(),
                creator.Name,
                sampleCount,
                suggestedTrain,
                difficulty,
                epochs);

            plans.Add(new GenesisAutonomousCreatorPlan(
                CreatorName: creator.Name,
                SampleCount: sampleCount,
                TrainCount: suggestedTrain,
                Difficulty: difficulty,
                Epochs: epochs,
                Priority: priority,
                Reason: reason));
        }

        var allocated = AllocateTrainCounts(request, plans);
        var roundEpochs = Math.Max(1, allocated.Max(p => p.Epochs));
        var summary = $"round {roundIndex + 1}: datasets={allocated.Count} budget={allocated.Sum(p => p.TrainCount)} epochs={roundEpochs}";
        return new GenesisAutonomousCompositePlan(
            Round: roundIndex + 1,
            Epochs: roundEpochs,
            CreatorPlans: allocated,
            Reason: summary);
    }

    private static IReadOnlyList<GenesisAutonomousCreatorPlan> AllocateTrainCounts(
        GenesisAutonomousTrainingRequest request,
        IReadOnlyList<GenesisAutonomousCreatorPlan> plans)
    {
        if (plans.Count == 0)
            return [];

        var targetBudget = Math.Max(
            plans.Count * Math.Max(1, request.MinTrainCount),
            Math.Max(request.RoundTrainBudget, plans.Count));

        var adjusted = plans
            .Select(p =>
            {
                var minTrain = Math.Max(1, request.MinTrainCount);
                var maxTrain = Math.Max(minTrain, request.MaxTrainCount);
                var bounded = Math.Clamp(p.TrainCount, minTrain, maxTrain);
                var effectiveTrain = Math.Min(bounded, Math.Max(1, p.SampleCount));
                return p with { TrainCount = effectiveTrain };
            })
            .ToArray();

        var current = adjusted.Sum(p => p.TrainCount);
        if (current > targetBudget)
        {
            var ordered = adjusted
                .Select((plan, index) => (plan, index))
                .OrderByDescending(x => x.plan.Priority)
                .ThenBy(x => x.plan.CreatorName, StringComparer.OrdinalIgnoreCase)
                .ToArray();
            var cursor = ordered.Length - 1;
            while (current > targetBudget && cursor >= 0)
            {
                var slot = ordered[cursor];
                var minTrain = Math.Max(1, request.MinTrainCount);
                if (adjusted[slot.index].TrainCount > minTrain)
                {
                    adjusted[slot.index] = adjusted[slot.index] with { TrainCount = adjusted[slot.index].TrainCount - 1 };
                    current--;
                }
                else
                {
                    cursor--;
                }
            }

            return adjusted;
        }

        if (current == targetBudget)
            return adjusted;

        var absoluteCap = adjusted.Sum(plan =>
        {
            var maxTrain = Math.Max(Math.Max(1, request.MinTrainCount), request.MaxTrainCount);
            return Math.Min(maxTrain, Math.Max(1, plan.SampleCount));
        });
        targetBudget = Math.Min(targetBudget, absoluteCap);
        if (current >= targetBudget)
            return adjusted;

        var remaining = targetBudget - current;
        var weighted = adjusted
            .Select((plan, index) => (plan, index))
            .OrderByDescending(x => x.plan.Priority)
            .ThenBy(x => x.plan.CreatorName, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        var weightedCount = weighted.Length;
        var position = 0;
        while (remaining > 0 && weightedCount > 0)
        {
            var maxTrain = Math.Max(Math.Max(1, request.MinTrainCount), request.MaxTrainCount);
            if (weighted.All(w =>
            {
                var cap = Math.Min(maxTrain, Math.Max(1, adjusted[w.index].SampleCount));
                return adjusted[w.index].TrainCount >= cap;
            }))
            {
                break;
            }

            var slot = weighted[position];
            position = (position + 1) % weightedCount;

            var creatorCap = Math.Min(maxTrain, Math.Max(1, adjusted[slot.index].SampleCount));
            if (adjusted[slot.index].TrainCount >= creatorCap)
                continue;

            adjusted[slot.index] = adjusted[slot.index] with { TrainCount = adjusted[slot.index].TrainCount + 1 };
            remaining--;
        }

        return adjusted;
    }

    private int ChooseSampleCount(
        GenesisAutonomousTrainingRequest request,
        GenesisAutonomousTrainingRound? last,
        CreatorSignal? signal)
    {
        if (last is null)
            return ScaleSampleRecommendation(request.InitialSampleCount, request);

        var samples = last.SampleCount;
        if (signal?.LastSeenCount > 0)
            samples = signal.LastSeenCount;
        else if (last.CreatorProgress?.SeenCount > 0)
            samples = last.CreatorProgress.SeenCount;

        if (signal is null)
            return ScaleSampleRecommendation(samples, request);

        // Combine loss and success rate signals for more robust feedback
        var feedbackScore = 0.0;

        // Loss component: higher loss means we need more work
        if (signal.AvgLoss > request.LossThreshold)
            feedbackScore += 1.0;
        else if (signal.AvgLoss <= request.LossThreshold * 0.8)
            feedbackScore -= 0.5;

        // Success rate component (now primary signal, not secondary)
        // Low success rates are a strong signal to increase samples/difficulty
        if (signal.SuccessRate < 0.5)
            feedbackScore += 1.5; // Strong need for more diverse examples
        else if (signal.SuccessRate >= 0.7)
            feedbackScore -= 1.0; // Doing well, can reduce sample pressure
        else if (signal.SuccessRate >= 0.85)
            feedbackScore -= 1.5; // Mastered, reduce significantly

        // Regressing or weak recent performance
        if (signal.Regressing || signal.RecentSuccessRate < 0.5)
            feedbackScore += 1.0;
        else if (signal.Mastered || signal.SuccessRate >= 0.85)
            feedbackScore -= 1.0;

        // Diversity penalty for consecutive same creator
        if (signal.ConsecutiveRecentRounds >= 2)
            feedbackScore -= 1.0;

        // Apply feedback to sample count with asymmetric scaling
        if (feedbackScore > 0)
            samples += Math.Max(1, request.SampleStepUp) * (int)Math.Ceiling(feedbackScore);
        else if (feedbackScore < 0)
            samples -= Math.Max(1, request.SampleStepDown) * (int)Math.Ceiling(Math.Abs(feedbackScore));

        return ScaleSampleRecommendation(samples, request);
    }

    private static int ScaleSampleRecommendation(int samples, GenesisAutonomousTrainingRequest request)
    {
        var scaled = samples * SampleRecommendationMultiplier;
        return Math.Clamp(scaled, request.MinSampleCount, request.MaxSampleCount);
    }

    private IExampleCreator ChooseCreator(
        GenesisAutonomousTrainingRequest request,
        IReadOnlyList<CreatorSignal> signals,
        IReadOnlyList<GenesisAutonomousTrainingRound> history,
        IReadOnlyList<IExampleCreator> activeCreators)
    {
        if (!string.IsNullOrWhiteSpace(request.PreferredCreator))
        {
            var preferred = activeCreators.FirstOrDefault(c =>
                c.Name.Equals(request.PreferredCreator, StringComparison.OrdinalIgnoreCase));
            if (preferred is not null)
                return preferred;
        }

        if (signals.Count == 0)
            return activeCreators[0];

        var candidateSignals = signals.Where(s => !s.Mastered).ToArray();
        if (candidateSignals.Length == 0)
            candidateSignals = signals.ToArray();

        var chosen = candidateSignals
            .OrderByDescending(s => s.Priority)
            .ThenBy(s => s.Attempts)
            .ThenBy(s => FindCreatorIndex(activeCreators, s.CreatorName))
            .First();

        if (history.Count >= 2)
        {
            var last = history[^1].CreatorName;
            var previous = history[^2].CreatorName;
            var sameRecentCreator = last.Equals(previous, StringComparison.OrdinalIgnoreCase) &&
                                    chosen.CreatorName.Equals(last, StringComparison.OrdinalIgnoreCase);

            if (sameRecentCreator)
            {
                var alternate = candidateSignals
                    .Where(s => !s.CreatorName.Equals(chosen.CreatorName, StringComparison.OrdinalIgnoreCase))
                    .OrderByDescending(s => s.Priority)
                    .ThenBy(s => s.Attempts)
                    .FirstOrDefault();

                if (alternate is not null && alternate.Priority >= chosen.Priority - request.AntiStreakPriorityWindow)
                    chosen = alternate;
            }
        }

        return activeCreators[FindCreatorIndex(activeCreators, chosen.CreatorName)];
    }

    private int ChooseDifficulty(
        GenesisAutonomousTrainingRequest request,
        GenesisAutonomousTrainingRound? last,
        CreatorSignal? signal)
    {
        if (last is null)
            return Math.Clamp(request.InitialDifficulty, 0, request.MaxDifficulty);

        var difficulty = last.Difficulty;

        // Success rate is now the primary difficulty signal
        // Only increase difficulty if success rate is high enough
        // Decrease if success rate drops
        if (signal is not null)
        {
            if (signal.SuccessRate >= 0.75 && signal.RecentSuccessRate >= 0.7)
            {
                // Strong success: can increase difficulty
                difficulty = Math.Min(request.MaxDifficulty, difficulty + Math.Max(1, request.DifficultyStepUp));
            }
            else if (signal.SuccessRate < 0.5 || signal.RecentSuccessRate < 0.4)
            {
                // Poor success: must decrease difficulty to focus learning
                difficulty = Math.Max(0, difficulty - Math.Max(1, request.DifficultyStepDown));
            }
            else if (signal.Mastered)
            {
                // Mastered can also increase
                difficulty = Math.Min(request.MaxDifficulty, difficulty + Math.Max(1, request.DifficultyStepUp));
            }
            else if (signal.Regressing)
            {
                // Regressing should step down
                difficulty = Math.Max(0, difficulty - Math.Max(1, request.DifficultyStepDown));
            }
        }

        return Math.Clamp(difficulty, 0, request.MaxDifficulty);
    }

    private int ChooseEpochs(
        GenesisAutonomousTrainingRequest request,
        GenesisAutonomousTrainingRound? last,
        CreatorSignal? signal)
    {
        if (last is null)
            return Math.Max(1, request.InitialEpochs);

        var epochs = Math.Max(1, last.Epochs);
        if (signal?.Regressing == true || signal?.RecentSuccessRate < 0.5)
            epochs = Math.Min(epochs + 1, Math.Max(1, request.InitialEpochs + 2));
        else if (signal?.Mastered == true || signal?.SuccessRate >= 0.85)
            epochs = Math.Max(1, epochs - 1);

        return Math.Max(1, epochs);
    }

    private int ChooseTrainCount(
        GenesisAutonomousTrainingRequest request,
        IReadOnlyList<GenesisAutonomousTrainingRound> creatorHistory,
        CreatorSignal? signal)
    {
        if (creatorHistory.Count == 0)
            return Math.Clamp(request.InitialTrainCount, request.MinTrainCount, request.MaxTrainCount);

        var last = creatorHistory[^1];
        var train = Math.Clamp(
            signal?.LastSeenCount > 0 ? signal.LastSeenCount : last.SampleCount,
            request.MinTrainCount,
            request.MaxTrainCount);

        var regressing = signal?.Regressing == true || IsTokenRegressing(last, request);
        var stable = signal?.Mastered == true || signal?.RecentSuccessRate >= 0.7 || IsStable(last, request);

        if (regressing)
            train = Math.Min(request.MaxTrainCount, train + Math.Max(1, request.TrainStepUp));
        else if (stable)
            train = Math.Max(request.MinTrainCount, train - Math.Max(1, request.TrainStepDown));

        return Math.Clamp(train, request.MinTrainCount, request.MaxTrainCount);
    }

    private static bool IsRegressing(GenesisAutonomousTrainingRound round, GenesisAutonomousTrainingRequest request)
    {
        return IsTokenRegressing(round, request) ||
            round.Report.SpaceNoiseRatio > request.RegressSpaceNoiseThreshold ||
            round.Report.ContradictionRate > request.RegressContradictionThreshold;
    }

    private static bool IsTokenRegressing(GenesisAutonomousTrainingRound round, GenesisAutonomousTrainingRequest request)
    {
        var creatorLoss = GetEffectiveCreatorLoss(round);
        var creatorSuccess = GetEffectiveCreatorSuccess(round);
        var tokenRegressThreshold = Math.Max(
            request.RegressTokenLossThreshold,
            request.LossThreshold * Math.Max(0.01, request.RegressLossMultiplier));

        var minSuccessForRegression = request.LossThreshold >= 1.0 ? 0.0 : 0.45;
        var lowExampleSuccess = creatorSuccess > 0 && creatorSuccess < minSuccessForRegression;
        return creatorLoss > tokenRegressThreshold || lowExampleSuccess || round.Report.AverageLoss.TokenLoss > tokenRegressThreshold;
    }

    private static bool IsStable(GenesisAutonomousTrainingRound round, GenesisAutonomousTrainingRequest request)
    {
        var creatorLoss = GetEffectiveCreatorLoss(round);
        var creatorSuccess = GetEffectiveCreatorSuccess(round);
        var tokenStable = creatorLoss <= request.LossThreshold;
        var minSuccessForStable = request.LossThreshold >= 1.0 ? 0.0 : 0.65;
        var exampleStable = creatorSuccess <= 0 || creatorSuccess >= minSuccessForStable;
        var overallStable = round.Report.AverageLoss.TokenLoss <= request.LossThreshold;
        return (tokenStable && exampleStable) || overallStable;
    }

    private static string BuildReason(
        GenesisAutonomousTrainingRequest request,
        int round,
        IReadOnlyList<CreatorSignal> signals,
        string creator,
        int samples,
        int trainCount,
        int difficulty,
        int epochs)
    {
        var signal = signals.FirstOrDefault(s => s.CreatorName.Equals(creator, StringComparison.OrdinalIgnoreCase));
        if (signal is null || signal.Attempts == 0)
            return $"round {round + 1}: start tiny, build fresh pools, and grow horizon with mastery";

        var trend = signal.Mastered
            ? "review"
            : signal.Regressing
                ? "focus-weakness"
                : signal.AvgLoss <= request.LossThreshold
                    ? "advance"
                    : "interleave";

        if (signal.ConsecutiveRecentRounds >= 2)
            trend = "diversify";

        return $"{trend}: dataset={creator}, samples={samples}, train={trainCount}, difficulty={difficulty}, epochs={epochs}, avgLoss={signal.AvgLoss:F4}, success={signal.SuccessRate:P1}";
    }

    private IReadOnlyList<CreatorSignal> BuildSignals(
        GenesisAutonomousTrainingRequest request,
        IReadOnlyList<GenesisAutonomousTrainingRound> history,
        IReadOnlyList<IExampleCreator> activeCreators)
    {
        var list = new List<CreatorSignal>(activeCreators.Count);
        var signalWindow = Math.Max(1, request.SignalWindow);
        var recentCreators = history
            .TakeLast(signalWindow)
            .Select(h => h.CreatorName)
            .ToArray();

        var latestCreator = history.Count > 0 ? history[^1].CreatorName : null;
        var latestCreatorStreak = 0;
        if (!string.IsNullOrWhiteSpace(latestCreator))
        {
            for (var i = history.Count - 1; i >= 0; i--)
            {
                var round = history[i];
                if (!round.CreatorName.Equals(latestCreator, StringComparison.OrdinalIgnoreCase))
                    break;
                latestCreatorStreak++;
            }
        }

        foreach (var creator in activeCreators)
        {
            var creatorRounds = history
                .Where(h => h.CreatorName.Equals(creator.Name, StringComparison.OrdinalIgnoreCase))
                .ToArray();
            var recentHits = recentCreators.Count(name => name.Equals(creator.Name, StringComparison.OrdinalIgnoreCase));
            var consecutiveRecentRounds = !string.IsNullOrWhiteSpace(latestCreator) &&
                                          creator.Name.Equals(latestCreator, StringComparison.OrdinalIgnoreCase)
                ? latestCreatorStreak
                : 0;

            if (creatorRounds.Length == 0)
            {
                list.Add(new CreatorSignal(
                    CreatorName: creator.Name,
                    Attempts: 0,
                    AvgLoss: request.LossThreshold * request.NewCreatorLossMultiplier,
                    SuccessRate: 0.0,
                    RecentSuccessRate: 0.0,
                    MaxDifficulty: 0,
                    LastSeenCount: request.InitialSampleCount,
                    RecentHits: recentHits,
                    ConsecutiveRecentRounds: consecutiveRecentRounds,
                    Mastered: false,
                    Regressing: false,
                    Priority: request.NewCreatorBasePriority - (recentHits * request.NewCreatorRecentPenalty)));
                continue;
            }

            var attempts = creatorRounds.Length;
            var recent = creatorRounds.TakeLast(signalWindow).ToArray();
            var avgLoss = recent.Average(GetEffectiveCreatorLoss);
            var successRate = creatorRounds.Average(GetEffectiveCreatorSuccess);
            var recentSuccessRate = recent.Average(GetEffectiveCreatorSuccess);
            var maxDifficulty = creatorRounds.Max(r => r.Difficulty);
            var lastSeenCount = creatorRounds[^1].CreatorProgress?.SeenCount > 0
                ? creatorRounds[^1].CreatorProgress!.SeenCount
                : creatorRounds[^1].SampleCount;
            var stableCount = creatorRounds.Count(r => IsStable(r, request));
            // Capability gate: a PROMPT-ANSWER creator must also reach the success threshold — and
            // with RequirePlatonicForCorrect that success counts only answers via the platonic path,
            // so a creator answering neurally never masters. Windowed-text (no platonic/prompt-answer
            // success signal) is exempt.
            var requiresPlatonicSuccess = creator.TrainingKind == GenesisTrainingExampleKind.PromptAnswer;
            var successMastered = !requiresPlatonicSuccess || recentSuccessRate >= request.MasterySuccessThreshold;
            // DRIVE-TO-DEPTH: require the creator to have reached MasteryDifficulty (clamped to its
            // legal range) before it can master — so the focus stays on it while it climbs difficulties
            // (stable, focused climb) instead of advancing at trivial difficulty and leaving the rest
            // to the oscillation-prone composite maintenance phase. The FocusBudget safety valve still
            // advances past a creator that can't reach this depth (it then rides along as replay).
            var requiredMasteryDifficulty = Math.Clamp(
                request.MasteryDifficulty,
                Math.Max(request.InitialDifficulty, 1),
                Math.Max(request.MaxDifficulty, 1));
            var mastered = attempts >= 3 &&
                           stableCount >= 2 &&
                           avgLoss <= request.LossThreshold * request.MasteryLossMultiplier &&
                           maxDifficulty >= requiredMasteryDifficulty &&
                           successMastered;
            var regressing = IsRegressing(recent[^1], request);
            var weakness = Math.Clamp(avgLoss / Math.Max(0.0001, request.LossThreshold), request.WeaknessMin, request.WeaknessMax);
            var exploration = request.ExplorationBase / (1.0 + attempts);
            var masteryPenalty = mastered ? request.MasteryPenalty : 0.0;
            var regressionBoost = regressing ? request.RegressionBoost : 0.0;
            var diversityPenalty = (recentHits * request.RecentHitPenalty) + (consecutiveRecentRounds * request.ConsecutivePenalty);
            var priority = weakness + exploration + regressionBoost - masteryPenalty - diversityPenalty;

            list.Add(new CreatorSignal(
                CreatorName: creator.Name,
                Attempts: attempts,
                AvgLoss: avgLoss,
                SuccessRate: successRate,
                RecentSuccessRate: recentSuccessRate,
                MaxDifficulty: maxDifficulty,
                LastSeenCount: lastSeenCount,
                RecentHits: recentHits,
                ConsecutiveRecentRounds: consecutiveRecentRounds,
                Mastered: mastered,
                Regressing: regressing,
                Priority: priority));
        }

        return list;
    }

    private static double GetEffectiveCreatorLoss(GenesisAutonomousTrainingRound round)
        => round.CreatorProgress?.AverageTokenLoss ?? round.Report.AverageLoss.TokenLoss;

    private static double GetEffectiveCreatorSuccess(GenesisAutonomousTrainingRound round)
        => round.CreatorProgress?.SuccessRate ?? round.Report.ExampleSuccessRate;

    private int FindCreatorIndex(IReadOnlyList<IExampleCreator> creators, string creatorName)
    {
        for (var i = 0; i < creators.Count; i++)
        {
            if (creators[i].Name.Equals(creatorName, StringComparison.OrdinalIgnoreCase))
                return i;
        }

        return -1;
    }

    // The "corenova:" name prefix marks a CORE tool-training primitive (teaches USING the platonic
    // space) — see NumberWordCreator/CategoryRetrievalCreator. Bootstrap-first gates on these.
    private const string CoreToolPrefix = "corenova:";

    // FOCUSED CURRICULUM: the active set for this round = the single FOCUS creator (the first, in
    // complexity order, that is neither mastered nor focus-exhausted) plus every MASTERED creator AND
    // every focus-EXHAUSTED creator (both ride along as low-priority replay for retention). Creators
    // AFTER the focus that are not yet mastered/exhausted are held back — only one NEW capability is
    // learned at a time. When every creator is mastered or exhausted, the full set trains (maintenance).
    // Curriculum position is derived entirely from the mastery signals, so it needs no extra persistence
    // and re-opens regressed creators automatically (a regressed creator's signal flips and it becomes
    // first-unmastered again).
    //
    // FORGETTING FIX: a focus-exhausted-but-unmastered creator used to be DROPPED from the set (neither
    // focus nor replay) — so the curriculum simply forgot it. It now rides along as replay exactly like
    // a mastered creator: the focus advances past it (it couldn't master within FocusBudget), but its
    // examples keep being rehearsed so its partial competence is retained and can keep climbing.
    private static IReadOnlyList<IExampleCreator> ApplyFocusedCurriculum(
        GenesisAutonomousTrainingRequest request,
        IReadOnlyList<IExampleCreator> active,
        IReadOnlyDictionary<string, CreatorSignal> signals)
    {
        if (active.Count <= 1)
            return active;

        // active is already complexity-ordered (constructor sorts _creators); ResolveActiveCreators
        // preserves that order.
        IExampleCreator? focus = null;
        var replay = new List<IExampleCreator>();
        foreach (var creator in active)
        {
            var hasSignal = signals.TryGetValue(creator.Name, out var s);
            var isMastered = hasSignal && s!.Mastered;
            if (isMastered)
            {
                replay.Add(creator); // mastered → replay for retention
                continue;
            }
            if (focus is not null)
            {
                // A focus is already chosen; a LATER creator only joins as replay if it is itself
                // exhausted (it had its focus turn already). Otherwise it waits its turn.
                if (hasSignal && s!.Attempts >= request.FocusBudget)
                    replay.Add(creator);
                continue;
            }
            var exhausted = hasSignal && s!.Attempts >= request.FocusBudget;
            if (!exhausted)
                focus = creator; // first unmastered, not-yet-exhausted creator → the focus
            else
                replay.Add(creator); // exhausted-and-unmastered → replay (retained, not forgotten)
        }

        if (focus is null)
            return active; // everything mastered or exhausted → train all (maintenance pass)

        var selected = new List<IExampleCreator>(replay.Count + 1) { focus };
        selected.AddRange(replay); // replay riders get little budget via priority allocation
        return selected;
    }

    private static IReadOnlyList<IExampleCreator> ApplyBootstrapFirstGate(
        GenesisAutonomousTrainingRequest request,
        int roundIndex,
        IReadOnlyList<IExampleCreator> active,
        IReadOnlyDictionary<string, CreatorSignal> signals)
    {
        if (!request.BootstrapFirst)
            return active;

        var coreCreators = active
            .Where(c => c.Name.StartsWith(CoreToolPrefix, StringComparison.OrdinalIgnoreCase))
            .ToArray();
        if (coreCreators.Length == 0 || coreCreators.Length == active.Count)
            return active; // nothing to gate (no primitives, or everything IS a primitive)

        // Safety valve: never let an un-masterable primitive starve broad training indefinitely.
        if (roundIndex >= request.BootstrapMaxRounds)
            return active;

        // Gate broad creators out until EVERY core primitive is mastered. A primitive with no signal
        // yet (never trained) counts as not-mastered, so the very first rounds are core-only.
        var allCoreMastered = coreCreators.All(c =>
            signals.TryGetValue(c.Name, out var s) && s.Mastered);

        return allCoreMastered ? active : coreCreators;
    }

    private IReadOnlyList<IExampleCreator> ResolveActiveCreators(GenesisAutonomousTrainingRequest request)
    {
        if (_creators.Count == 0)
            throw new InvalidOperationException("No example creators are registered.");

        if (request.EnabledCreators is not { Count: > 0 })
            return _creators;

        var enabled = request.EnabledCreators
            .Where(name => !string.IsNullOrWhiteSpace(name))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        var active = _creators
            .Where(c => enabled.Contains(c.Name))
            .ToArray();
        if (active.Length == 0)
            throw new InvalidOperationException("No selected autonomous datasets matched registered creators.");

        return active;
    }
}
