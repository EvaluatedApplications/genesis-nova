using GenesisNova.Data;

namespace GenesisNova.Train;

public sealed record PreTokenizedExample(
    int[] InputTokens,
    int[] TargetTokens,
    GenesisExample Original);

public sealed class GenesisTrainingOrchestrator
{
    private readonly GenesisTrainer _trainer;

    public GenesisTrainingOrchestrator(GenesisTrainer trainer)
    {
        _trainer = trainer;
    }

    /// <summary>Generate training examples from registered creators (no route labels).</summary>
    public static List<GenesisExample> GenerateExamplesFromCreators(
        int count,
        int difficulty = 0,
        Random? rng = null)
    {
        rng ??= new Random();
        var examples = new List<GenesisExample>();
        var creators = ExampleCreatorRegistry.All;

        // Distribute examples evenly across creators
        int perCreator = count / creators.Count;
        int remainder = count % creators.Count;

        for (int creatorIndex = 0; creatorIndex < creators.Count; creatorIndex++)
        {
            var creator = creators[creatorIndex];
            int toGenerate = perCreator + (remainder > 0 ? 1 : 0);
            remainder--;

            var generated = creator.Generate(toGenerate, difficulty, forTraining: true);
            foreach (var (input, output) in generated)
            {
                // No route labels - implicit routing emerges from model training
                examples.Add(new GenesisExample(input, output, null));
            }
        }

        // Shuffle final list
        for (int i = examples.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (examples[i], examples[j]) = (examples[j], examples[i]);
        }

        return examples.Take(count).ToList();
    }

    public GenesisTrainingReport Train(
        IReadOnlyList<GenesisExample> examples,
        int epochs,
        int? introspectionCyclesPerEpoch = null,
        Action<string>? logger = null)
    {
        if (examples.Count == 0)
            throw new ArgumentException("No training examples provided.", nameof(examples));

        var overallStart = System.Diagnostics.Stopwatch.StartNew();
        logger?.Invoke($"[TRAIN] Starting training: {examples.Count} examples × {epochs} epochs");

        // Pre-tokenize all examples in parallel (CPU-only, no GPU contention)
        var preTokenStart = System.Diagnostics.Stopwatch.StartNew();
        var preTokenized = PreTokenizeExamples(examples);
        preTokenStart.Stop();
        logger?.Invoke($"[PRETOK] Pre-tokenized {preTokenized.Count} examples in {preTokenStart.ElapsedMilliseconds}ms");

        var totalSteps = 0;
        var sumToken = 0.0;
        var introspected = 0;
        var batchSize = Math.Max(1, Math.Min(16, examples.Count / 2));
        logger?.Invoke($"[CONFIG] batch_size={batchSize} examples={examples.Count}");

        for (var e = 1; e <= Math.Max(1, epochs); e++)
        {
            var epochStart = System.Diagnostics.Stopwatch.StartNew();
            var epochSum = 0.0;
            var batchCount = 0;

            // Process examples in batches
            for (var batchStart = 0; batchStart < preTokenized.Count; batchStart += batchSize)
            {
                var batchEnd = Math.Min(batchStart + batchSize, preTokenized.Count);
                var batch = preTokenized
                    .Skip(batchStart)
                    .Take(batchEnd - batchStart)
                    .Select(ex => (ex.InputTokens, ex.TargetTokens))
                    .ToList();

                var batchStepStart = System.Diagnostics.Stopwatch.StartNew();
                var loss = _trainer.TrainBatchPreTokenized(batch);
                batchStepStart.Stop();

                totalSteps++;
                batchCount++;
                sumToken += loss.TokenLoss;
                epochSum += loss.TokenLoss * batch.Count;

                logger?.Invoke($"  [BATCH] {batchStart + 1:D3}-{batchEnd:D3} loss={loss.TokenLoss:F4} time={batchStepStart.ElapsedMilliseconds:D5}ms");
            }

            epochStart.Stop();

            // Clone parameters ONCE per epoch to break the computation graph chain
            _trainer.CloneParametersToBreakGraph();

            var defaultCycles = 0;
            var cyclesThisEpoch = introspectionCyclesPerEpoch ?? defaultCycles;
            cyclesThisEpoch = Math.Clamp(cyclesThisEpoch, 0, 256);
            introspected += _trainer.RunIntrospectionCycles(cyclesThisEpoch);
            var epochLoss = epochSum / Math.Max(1, examples.Count);
            logger?.Invoke($"[EPOCH] {e}/{epochs} loss={epochLoss:F4} time={epochStart.ElapsedMilliseconds:D5}ms batches={batchCount} introspected={introspected}");
        }

        overallStart.Stop();
        var avgTokenLoss = sumToken / Math.Max(1, totalSteps);
        var contradictionRate = EstimateContradictionRate(examples);

        logger?.Invoke($"[DONE] Training complete in {overallStart.ElapsedMilliseconds / 1000.0:F1}s. Avg loss={avgTokenLoss:F4}");

        return new GenesisTrainingReport(
            Epochs: Math.Max(1, epochs),
            ExampleCount: examples.Count,
            AverageLoss: new GenesisStepLoss(
                TokenLoss: avgTokenLoss,
                RouteLoss: 0.0,
                ConsistencyLoss: 0.0,
                ConservationLoss: 0.0,
                MemoryLoss: 0.0,
                TotalLoss: avgTokenLoss),
            ContradictionRate: contradictionRate,
            ConservationDrift: 0.0,
            MemoryOverwriteRate: 0.0,
            IntrospectionCycles: introspected,
            PendingQueueDepth: _trainer.QueueSize);
    }

    private List<PreTokenizedExample> PreTokenizeExamples(IReadOnlyList<GenesisExample> examples)
    {
        var result = new List<PreTokenizedExample>(examples.Count);
        var exampleDict = new Dictionary<GenesisExample, int>();
        for (var i = 0; i < examples.Count; i++)
            exampleDict[examples[i]] = i;

        var lockObj = new object();

        // Parallel tokenization (CPU-bound, no GPU interference)
        Parallel.ForEach(examples, ex =>
        {
            var input = _trainer.EncodeInput(ex.Input);
            var target = _trainer.EncodeTarget(ex.Output);
            var preToken = new PreTokenizedExample(input, target, ex);

            lock (lockObj)
                result.Add(preToken);
        });

        return result.OrderBy(x => exampleDict[x.Original]).ToList();
    }

    private static double EstimateContradictionRate(IReadOnlyList<GenesisExample> examples)
    {
        var groups = examples
            .GroupBy(x => x.Input.Trim().ToLowerInvariant())
            .Where(g => g.Select(x => x.Output.Trim().ToLowerInvariant()).Distinct().Count() > 1)
            .ToArray();

        if (examples.Count == 0)
            return 0.0;

        var contradictory = groups.Sum(g => g.Count());
        return contradictory / (double)examples.Count;
    }
}
