using GenesisNova.Data;
using GenesisNova.Data.Creators;
using System.Collections.Concurrent;
using System.Collections.Immutable;
using System.Diagnostics;

namespace GenesisNova.Runtime;

internal sealed class GenerateAutonomousCandidatePoolsStep
{
    public GenesisAutonomousTrainTaskData Execute(GenesisAutonomousTrainTaskData data)
    {
        var plan = data.Plan ?? throw new InvalidOperationException("Autonomous plan is required.");
        var creatorLookup = ExampleCreatorRegistry.All
            .ToDictionary(c => c.Name, StringComparer.OrdinalIgnoreCase);
        var pools = new ConcurrentDictionary<string, ImmutableArray<GenesisExample>>(StringComparer.OrdinalIgnoreCase);
        data.CancellationToken.ThrowIfCancellationRequested();
        var generationWorkers = Math.Clamp(
            data.Request.MaxGenerationConcurrency,
            1,
            Math.Max(1, Environment.ProcessorCount * 2));
        data.UiLogger?.Invoke(
            $"[auto] round {plan.Round}: generating pools for {plan.CreatorPlans.Count} datasets (workers={generationWorkers})...");
        var generationSw = Stopwatch.StartNew();

        var parallelOptions = new ParallelOptions
        {
            CancellationToken = data.CancellationToken,
            MaxDegreeOfParallelism = generationWorkers
        };

        Parallel.ForEach(plan.CreatorPlans, parallelOptions, creatorPlan =>
        {
            data.CancellationToken.ThrowIfCancellationRequested();
            if (!creatorLookup.TryGetValue(creatorPlan.CreatorName, out var creator))
                throw new InvalidOperationException($"Creator not found: {creatorPlan.CreatorName}");

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
                {
                    cacheSize = new FileInfo(cachePath).Length;
                }
                data.UiLogger?.Invoke(
                    $"[auto]   cache dataset={creatorPlan.CreatorName} file={Path.GetFileName(cachePath)} size={cacheSize} bytes");
            }
            if (generated.Length == 0)
                throw new InvalidOperationException($"Creator produced no examples: {creator.Name}");

            var examples = generated
                .Select(ex => new GenesisExample(ex.Input, ex.Output))
                .ToImmutableArray();
            pools[creator.Name] = examples;
            data.UiLogger?.Invoke(
                $"[auto]   generated dataset={creatorPlan.CreatorName} examples={examples.Length} time={creatorSw.ElapsedMilliseconds}ms");
        });

        generationSw.Stop();
        data.UiLogger?.Invoke(
            $"[auto] round {plan.Round}: pool generation complete in {generationSw.ElapsedMilliseconds}ms");

        return data with { CandidatePools = pools };
    }
}
