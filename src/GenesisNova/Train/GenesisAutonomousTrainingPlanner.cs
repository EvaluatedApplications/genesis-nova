using GenesisNova.Data;

namespace GenesisNova.Train;

public sealed class GenesisAutonomousTrainingPlanner
{
    private readonly IReadOnlyList<IExampleCreator> _creators;
    private sealed record CreatorSignal(
        string CreatorName,
        int Attempts,
        double AvgLoss,
        int MaxDifficulty,
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
        if (_creators.Count == 0)
            throw new InvalidOperationException("No example creators are registered.");

        var round = history.Count;
        var signals = BuildSignals(request, history);

        var creator = ChooseCreator(request, signals, history);
        var creatorHistory = history
            .Where(h => h.CreatorName.Equals(creator.Name, StringComparison.OrdinalIgnoreCase))
            .ToArray();
        var lastForCreator = creatorHistory.Length > 0 ? creatorHistory[^1] : null;

        var difficulty = ChooseDifficulty(request, lastForCreator);
        var sampleCount = ChooseSampleCount(request, lastForCreator);
        var trainCount = ChooseTrainCount(request, creatorHistory);
        var epochs = ChooseEpochs(request, lastForCreator);
        var reason = BuildReason(request, round, signals, creator.Name, sampleCount, trainCount, difficulty);

        return new GenesisAutonomousTrainingPlan(
            CreatorName: creator.Name,
            SampleCount: sampleCount,
            TrainCount: trainCount,
            Difficulty: difficulty,
            Epochs: epochs,
            Reason: reason);
    }

    private IExampleCreator ChooseCreator(
        GenesisAutonomousTrainingRequest request,
        IReadOnlyList<CreatorSignal> signals,
        IReadOnlyList<GenesisAutonomousTrainingRound> history)
    {
        if (!string.IsNullOrWhiteSpace(request.PreferredCreator))
        {
            var preferred = _creators.FirstOrDefault(c =>
                c.Name.Equals(request.PreferredCreator, StringComparison.OrdinalIgnoreCase));
            if (preferred is not null)
                return preferred;
        }

        if (signals.Count == 0)
            return _creators[0];

        var candidateSignals = signals.Where(s => !s.Mastered).ToArray();
        if (candidateSignals.Length == 0)
            candidateSignals = signals.ToArray();

        var chosen = candidateSignals
            .OrderByDescending(s => s.Priority)
            .ThenBy(s => s.Attempts)
            .ThenBy(s => FindCreatorIndex(s.CreatorName))
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

        return _creators[FindCreatorIndex(chosen.CreatorName)];
    }

    private int ChooseDifficulty(GenesisAutonomousTrainingRequest request, GenesisAutonomousTrainingRound? last)
    {
        if (last is null)
            return Math.Clamp(request.InitialDifficulty, 0, request.MaxDifficulty);

        var report = last.Report;
        var difficulty = last.Difficulty;
        if (IsTokenRegressing(report, request))
            difficulty = Math.Max(0, difficulty - Math.Max(1, request.DifficultyStepDown));
        else if (report.AverageLoss.TokenLoss <= request.LossThreshold)
            difficulty = Math.Min(request.MaxDifficulty, difficulty + Math.Max(1, request.DifficultyStepUp));

        return Math.Clamp(difficulty, 0, request.MaxDifficulty);
    }

    private int ChooseSampleCount(GenesisAutonomousTrainingRequest request, GenesisAutonomousTrainingRound? last)
    {
        if (last is null)
            return Math.Clamp(request.InitialSampleCount, request.MinSampleCount, request.MaxSampleCount);

        var report = last.Report;
        var samples = last.SampleCount;
        if (IsTokenRegressing(report, request))
            samples = Math.Max(request.MinSampleCount, samples - Math.Max(1, request.SampleStepDown));
        else if (report.AverageLoss.TokenLoss <= request.LossThreshold)
            samples = Math.Min(request.MaxSampleCount, samples + Math.Max(1, request.SampleStepUp));

        return Math.Clamp(samples, request.MinSampleCount, request.MaxSampleCount);
    }

    private int ChooseEpochs(GenesisAutonomousTrainingRequest request, GenesisAutonomousTrainingRound? last)
        => last is null ? Math.Max(1, request.InitialEpochs) : Math.Max(1, last.Epochs);

    private int ChooseTrainCount(
        GenesisAutonomousTrainingRequest request,
        IReadOnlyList<GenesisAutonomousTrainingRound> creatorHistory)
    {
        if (creatorHistory.Count == 0)
            return Math.Clamp(request.InitialTrainCount, request.MinTrainCount, request.MaxTrainCount);

        var signalWindow = Math.Max(1, request.SignalWindow);
        var recent = creatorHistory
            .TakeLast(signalWindow)
            .ToArray();

        var recentAvgLoss = recent.Average(r => r.Report.AverageLoss.TokenLoss);
        var previous = creatorHistory
            .Take(Math.Max(0, creatorHistory.Count - signalWindow))
            .TakeLast(signalWindow)
            .ToArray();
        var previousAvgLoss = previous.Length > 0
            ? previous.Average(r => r.Report.AverageLoss.TokenLoss)
            : recentAvgLoss;

        var baseline = Math.Clamp(request.InitialTrainCount, request.MinTrainCount, request.MaxTrainCount);
        var regressing = recentAvgLoss > previousAvgLoss + request.TrainRegressionDelta || IsRegressing(recent[0].Report, request);
        if (regressing)
            return Math.Min(request.MaxTrainCount, baseline + Math.Max(1, request.TrainStepUp));

        if (recentAvgLoss <= request.LossThreshold)
            return Math.Max(request.MinTrainCount, baseline - Math.Max(1, request.TrainStepDown));

        return baseline;
    }

    private static bool IsRegressing(GenesisTrainingReport report, GenesisAutonomousTrainingRequest request)
    {
        return IsTokenRegressing(report, request) ||
            report.SpaceNoiseRatio > request.RegressSpaceNoiseThreshold ||
            report.ContradictionRate > request.RegressContradictionThreshold;
    }

    private static bool IsTokenRegressing(GenesisTrainingReport report, GenesisAutonomousTrainingRequest request)
    {
        var tokenRegressThreshold = Math.Max(
            request.RegressTokenLossThreshold,
            request.LossThreshold * Math.Max(0.01, request.RegressLossMultiplier));

        return report.AverageLoss.TokenLoss > tokenRegressThreshold;
    }

    private static string BuildReason(
        GenesisAutonomousTrainingRequest request,
        int round,
        IReadOnlyList<CreatorSignal> signals,
        string creator,
        int samples,
        int trainCount,
        int difficulty)
    {
        var signal = signals.FirstOrDefault(s => s.CreatorName.Equals(creator, StringComparison.OrdinalIgnoreCase));
        if (signal is null || signal.Attempts == 0)
            return $"round {round + 1}: start low and learn quickly";

        var trend = signal.Mastered
            ? "review"
            : signal.Regressing
                ? "focus-weakness"
                : signal.AvgLoss <= request.LossThreshold
                    ? "advance"
                    : "interleave";

        if (signal.ConsecutiveRecentRounds >= 2)
            trend = "diversify";

        return $"{trend}: creator={creator}, samples={samples}, train={trainCount}, difficulty={difficulty}, avgLoss={signal.AvgLoss:F4}";
    }

    private IReadOnlyList<CreatorSignal> BuildSignals(
        GenesisAutonomousTrainingRequest request,
        IReadOnlyList<GenesisAutonomousTrainingRound> history)
    {
        var list = new List<CreatorSignal>(_creators.Count);
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

        foreach (var creator in _creators)
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
                    MaxDifficulty: 0,
                    RecentHits: recentHits,
                    ConsecutiveRecentRounds: consecutiveRecentRounds,
                    Mastered: false,
                    Regressing: false,
                    Priority: request.NewCreatorBasePriority - (recentHits * request.NewCreatorRecentPenalty)));
                continue;
            }

            var attempts = creatorRounds.Length;
            var recent = creatorRounds.TakeLast(signalWindow).ToArray();
            var avgLoss = recent.Average(r => r.Report.AverageLoss.TokenLoss);
            var maxDifficulty = creatorRounds.Max(r => r.Difficulty);
            var stableCount = creatorRounds.Count(r => r.Report.AverageLoss.TokenLoss <= request.LossThreshold);
            var mastered = attempts >= 3 &&
                           stableCount >= 2 &&
                           avgLoss <= request.LossThreshold * request.MasteryLossMultiplier &&
                           maxDifficulty >= Math.Max(request.InitialDifficulty, 1);
            var regressing = IsRegressing(recent[0].Report, request);
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
                MaxDifficulty: maxDifficulty,
                RecentHits: recentHits,
                ConsecutiveRecentRounds: consecutiveRecentRounds,
                Mastered: mastered,
                Regressing: regressing,
                Priority: priority));
        }

        return list;
    }

    private int FindCreatorIndex(string creatorName)
    {
        for (var i = 0; i < _creators.Count; i++)
        {
            if (_creators[i].Name.Equals(creatorName, StringComparison.OrdinalIgnoreCase))
                return i;
        }

        return -1;
    }
}
