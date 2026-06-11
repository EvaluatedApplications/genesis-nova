using EvalApp.Consumer;
using GenesisNova.Data;
using GenesisNova.Train;
using System.Collections.Generic;
using System.Linq;

namespace GenesisNova.Runtime;

internal sealed class BuildAutonomousTrainingBatchStep : IStep<GenesisAutonomousTrainTaskData>
{
    public ValueTask<GenesisAutonomousTrainTaskData> ExecuteAsync(GenesisAutonomousTrainTaskData data, CancellationToken ct)
    {
        var plan = data.Plan ?? throw new InvalidOperationException("Autonomous plan is required.");
        var pools = data.CandidatePools ?? throw new InvalidOperationException("Candidate pools are required.");
        ct.ThrowIfCancellationRequested();
        data.UiLogger?.Invoke($"[auto] round {plan.Round}: building mixed training batch...");
        var perCreator = new List<(string CreatorName, int Difficulty, int SampleCount, int Epochs, GenesisExample[] Examples)>(plan.CreatorPlans.Count);
        foreach (var creatorPlan in plan.CreatorPlans)
        {
            if (!pools.TryGetValue(creatorPlan.CreatorName, out var pool))
                throw new InvalidOperationException($"Missing candidate pool for creator: {creatorPlan.CreatorName}");

            var trainCount = Math.Clamp(creatorPlan.TrainCount, 1, pool.Length);
            var shuffled = pool.ToArray();
            ShuffleInPlace(shuffled);
            var selected = shuffled.Take(trainCount).ToArray();

            perCreator.Add((creatorPlan.CreatorName, creatorPlan.Difficulty, creatorPlan.SampleCount, creatorPlan.Epochs, selected));
        }

        var maxTrain = perCreator.Count == 0 ? 0 : perCreator.Max(x => x.Examples.Length);
        var merged = new List<GenesisExample>(perCreator.Sum(x => x.Examples.Length));
        for (var i = 0; i < maxTrain; i++)
        {
            foreach (var creator in perCreator)
            {
                if (i < creator.Examples.Length)
                    merged.Add(creator.Examples[i]);
            }
        }

        var creatorRounds = perCreator
            .Select(c => new GenesisAutonomousTrainingRound(
                Round: data.RoundIndex + 1,
                CreatorName: c.CreatorName,
                SampleCount: c.SampleCount,
                Difficulty: c.Difficulty,
                Epochs: c.Epochs,
                Report: new GenesisTrainingReport(
                    Epochs: 0,
                    ExampleCount: 0,
                    AverageLoss: new GenesisStepLoss(0, 0, 0, 0, 0, 0),
                    ContradictionRate: 0,
                    ConservationDrift: 0,
                    MemoryOverwriteRate: 0,
                    IntrospectionCycles: 0,
                    PendingQueueDepth: 0,
                    SpaceManagementCycles: 0,
                    NodesPruned: 0,
                    RelationsPruned: 0,
                    FinalNodeCount: 0,
                    FinalRelationCount: 0,
                    SpaceNoiseRatio: 0),
                CreatorProgress: null))
            .ToArray();
        data.UiLogger?.Invoke(
            $"[auto] round {plan.Round}: mixed batch ready (datasets={creatorRounds.Length}, trained={merged.Count})");

        return ValueTask.FromResult(data with
        {
            TrainingExamples = merged,
            CreatorRounds = creatorRounds
        });
    }

    private static void ShuffleInPlace<T>(T[] values)
    {
        for (var i = values.Length - 1; i > 0; i--)
        {
            var j = Random.Shared.Next(i + 1);
            (values[i], values[j]) = (values[j], values[i]);
        }
    }
}
