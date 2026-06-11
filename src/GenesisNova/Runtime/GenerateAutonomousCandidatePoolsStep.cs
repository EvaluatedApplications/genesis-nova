using EvalApp.Consumer;
using GenesisNova.Data;
using GenesisNova.Data.Creators;
using GenesisNova.Train;
using System.Collections.Concurrent;
using System.Collections.Immutable;
using System.Diagnostics;

namespace GenesisNova.Runtime;

/// <summary>
/// Carrier for a single creator's candidate-pool generation work item.
/// Used as the <c>TItem</c> for the EvalApp gated <c>ForEach</c> so that the
/// resource-gated sub-task can transform each plan into its generated pool
/// while the merge step folds the results back into the parent task data.
/// </summary>
internal sealed record GenesisCandidatePoolItem(
    GenesisAutonomousCreatorPlan Plan,
    Action<string>? UiLogger,
    ImmutableArray<GenesisExample> Examples = default);

/// <summary>
/// Generates one creator's candidate example pool. This is the per-item body of
/// the EvalApp gated <c>ForEach</c> wired in <see cref="GenesisEvalAppRuntime"/>;
/// bounded-concurrency and resource gating are owned by the pipeline, not this step.
/// </summary>
internal sealed class GenerateCandidatePoolItemStep : IStep<GenesisCandidatePoolItem>
{
    public ValueTask<GenesisCandidatePoolItem> ExecuteAsync(
        GenesisCandidatePoolItem data,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var creatorPlan = data.Plan;
        var creator = ExampleCreatorRegistry.All
            .FirstOrDefault(c => string.Equals(c.Name, creatorPlan.CreatorName, StringComparison.OrdinalIgnoreCase))
            ?? throw new InvalidOperationException($"Creator not found: {creatorPlan.CreatorName}");

        var creatorSw = Stopwatch.StartNew();
        data.UiLogger?.Invoke(
            $"[auto]   generating dataset={creatorPlan.CreatorName} pool={creatorPlan.SampleCount} difficulty={creatorPlan.Difficulty}");
        var generated = creator.Generate(
            Math.Max(1, creatorPlan.SampleCount),
            creatorPlan.Difficulty,
            forTraining: true);

        if (creator is PublicTextCorpusCreator corpusCreator)
        {
            var cachePath = corpusCreator.LocalCorpusPath;
            long cacheSize = 0;
            if (File.Exists(cachePath))
                cacheSize = new FileInfo(cachePath).Length;
            data.UiLogger?.Invoke(
                $"[auto]   cache dataset={creatorPlan.CreatorName} file={Path.GetFileName(cachePath)} size={cacheSize} bytes");
        }

        if (generated.Length == 0)
            throw new InvalidOperationException($"Creator produced no examples: {creator.Name}");

        var examples = generated
            .Select(ex => new GenesisExample(ex.Input, ex.Output, creator.TrainingKind, creator.Name))
            .ToImmutableArray();
        data.UiLogger?.Invoke(
            $"[auto]   generated dataset={creatorPlan.CreatorName} examples={examples.Length} time={creatorSw.ElapsedMilliseconds}ms");

        return ValueTask.FromResult(data with { Examples = examples });
    }
}

/// <summary>
/// Projects a plan into the per-creator work items consumed by the gated ForEach,
/// and merges the generated pools back into the task data. Pure selection / merge
/// helpers; the actual generation runs inside the resource-gated ForEach body.
/// </summary>
internal static class GenerateAutonomousCandidatePoolsStep
{
    public static IEnumerable<GenesisCandidatePoolItem> SelectWorkItems(GenesisAutonomousTrainTaskData data)
    {
        var plan = data.Plan ?? throw new InvalidOperationException("Autonomous plan is required.");
        data.CancellationToken.ThrowIfCancellationRequested();
        data.UiLogger?.Invoke(
            $"[auto] round {plan.Round}: generating pools for {plan.CreatorPlans.Count} datasets (resource-gated ForEach)...");
        return plan.CreatorPlans.Select(p => new GenesisCandidatePoolItem(p, data.UiLogger));
    }

    public static GenesisAutonomousTrainTaskData MergePools(
        GenesisAutonomousTrainTaskData data,
        IReadOnlyList<GenesisCandidatePoolItem> results)
    {
        var pools = new ConcurrentDictionary<string, ImmutableArray<GenesisExample>>(StringComparer.OrdinalIgnoreCase);
        foreach (var item in results)
        {
            if (item.Examples.IsDefaultOrEmpty)
                throw new InvalidOperationException($"Creator produced no examples: {item.Plan.CreatorName}");
            pools[item.Plan.CreatorName] = item.Examples;
        }

        data.UiLogger?.Invoke(
            $"[auto] round {data.Plan?.Round}: pool generation complete (datasets={pools.Count})");
        return data with { CandidatePools = pools };
    }
}
