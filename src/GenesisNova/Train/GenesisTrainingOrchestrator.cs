using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Runtime;

namespace GenesisNova.Train;

public sealed record PreTokenizedExample(
    int[] InputTokens,
    int[] TargetTokens,
    GenesisTrainingExampleKind TrainingKind,
    GenesisExample Original);

internal sealed class ExampleProgressAccumulator
{
    public ExampleProgressAccumulator(GenesisExample example, GenesisTrainingExampleKind trainingKind)
    {
        Example = example;
        TrainingKind = trainingKind;
        BestTokenLoss = double.MaxValue;
    }

    public GenesisExample Example { get; }
    public GenesisTrainingExampleKind TrainingKind { get; }
    public int SeenCount { get; private set; }
    public int SuccessCount { get; private set; }
    public double TokenLossSum { get; private set; }
    public double LastTokenLoss { get; private set; }
    public double BestTokenLoss { get; private set; }

    public double SuccessRate => SeenCount == 0 ? 0.0 : SuccessCount / (double)SeenCount;
    public double AverageTokenLoss => SeenCount == 0 ? 0.0 : TokenLossSum / SeenCount;

    public void Observe(GenesisPerExampleLoss update)
    {
        SeenCount++;
        if (update.IsCorrect)
            SuccessCount++;

        LastTokenLoss = update.Loss.TokenLoss;
        TokenLossSum += update.Loss.TokenLoss;
        BestTokenLoss = Math.Min(BestTokenLoss, update.Loss.TokenLoss);
    }
}

internal sealed class CreatorProgressAccumulator
{
    public CreatorProgressAccumulator(string creatorName, GenesisTrainingExampleKind trainingKind)
    {
        CreatorName = creatorName;
        TrainingKind = trainingKind;
        BestTokenLoss = double.MaxValue;
    }

    public string CreatorName { get; }
    public GenesisTrainingExampleKind TrainingKind { get; }
    public int SeenCount { get; private set; }
    public int SuccessCount { get; private set; }
    public double TokenLossSum { get; private set; }
    public double LastTokenLoss { get; private set; }
    public double BestTokenLoss { get; private set; }

    public double SuccessRate => SeenCount == 0 ? 0.0 : SuccessCount / (double)SeenCount;
    public double AverageTokenLoss => SeenCount == 0 ? 0.0 : TokenLossSum / SeenCount;

    public void Observe(GenesisPerExampleLoss update)
    {
        SeenCount++;
        if (update.IsCorrect)
            SuccessCount++;

        LastTokenLoss = update.Loss.TokenLoss;
        TokenLossSum += update.Loss.TokenLoss;
        BestTokenLoss = Math.Min(BestTokenLoss, update.Loss.TokenLoss);
    }
}

public sealed class GenesisTrainingOrchestrator
{
    private readonly GenesisTrainer _trainer;
    private readonly GenesisNovaConfig _config;
    public GenesisTrainingOrchestrator(GenesisTrainer trainer, GenesisNovaConfig? config = null)
    {
        _trainer = trainer;
        _config = config ?? new GenesisNovaConfig();
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
                examples.Add(new GenesisExample(input, output, creator.TrainingKind, creator.Name));
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

    // Anneal step size by proximity to mastery — the curve validated in CoreBootstrapRegime. Keep
    // FULL steps until success is genuinely high (a sub-target plateau needs more step, not less);
    // only shrink near the top, where fixed-LR overshoot causes the ~90% oscillation.
    internal static double AnnealedLearningRateFactor(double successRate) => successRate switch
    {
        < 0.92 => 1.00,
        < 0.97 => 0.30,
        _ => 0.10,
    };

    public GenesisTrainingReport Train(
        IReadOnlyList<GenesisExample> examples,
        int epochs,
        Action<string>? logger = null,
        CancellationToken cancellationToken = default)
    {
        if (examples.Count == 0)
            throw new ArgumentException("No training examples provided.", nameof(examples));

        var overallStart = System.Diagnostics.Stopwatch.StartNew();
        logger?.Invoke($"[TRAIN] Starting training: {examples.Count} examples × {epochs} epochs");

        var totalSteps = 0;
        var sumToken = 0.0;
        var sumRoute = 0.0;
        var sumConsistency = 0.0;
        var sumConservation = 0.0;
        var sumMemory = 0.0;
        var sumTotal = 0.0;
        var spaceManagementCycles = 0;
        var totalNodesPruned = 0;
        var totalRelationsPruned = 0;
        var lastNoiseRatio = 0.0;
        var finalNodeCount = 0;
        var finalRelationCount = 0;
        var perExampleProgress = new Dictionary<string, ExampleProgressAccumulator>(StringComparer.Ordinal);
        var perCreatorProgress = new Dictionary<string, CreatorProgressAccumulator>(StringComparer.OrdinalIgnoreCase);
        var skippedCorrectExampleCount = 0;
        var promptAnswerExampleCount = 0;
        var windowedTextExampleCount = 0;
        var hasCuda = GpuCapacityPlanner.TryGetNvidiaVramMb(out var totalVramMb, out var freeVramMb);
        var usableVramMb = hasCuda ? (freeVramMb > 0 ? freeVramMb : totalVramMb) : 0;
        var maxTargetTokens = ResolveMaxTargetTokens(hasCuda, usableVramMb);

        // Pre-tokenize examples and trim overlong targets for GPU safety on smaller VRAM devices.
        var preTokenStart = System.Diagnostics.Stopwatch.StartNew();
        var preTokenized = PreTokenizeExamples(examples, maxTargetTokens, logger);
        preTokenStart.Stop();
        logger?.Invoke(
            $"[PRETOK] Pre-tokenized {preTokenized.Count} examples in {preTokenStart.ElapsedMilliseconds}ms (target_cap={maxTargetTokens})");

        var avgTargetTokens = Math.Max(1, (int)Math.Round(preTokenized.Average(x => x.TargetTokens.Length)));
        var targetVramUtilization = Math.Clamp(_config.TargetVramUtilization, 0.5, 0.95);
        var reserveVramMb = Math.Max(256, _config.ReserveVramMb);
        var estimatedBatchSize = GpuCapacityPlanner.EstimateTrainingBatchSize(
            exampleCount: examples.Count,
            hiddenSize: _trainer.HiddenSize,
            averageTargetTokens: avgTargetTokens,
            gpuAvailable: hasCuda,
            vramMb: usableVramMb,
            targetUtilization: targetVramUtilization,
            reserveVramMb: reserveVramMb,
            cpuThreads: Environment.ProcessorCount,
            debugOutput: logger);
        var baseBatchSize = Math.Max(1, estimatedBatchSize);
        logger?.Invoke(
            $"[CONFIG] batch_size={baseBatchSize} examples={examples.Count} avgTokens={avgTargetTokens} hidden={_trainer.HiddenSize}");
        logger?.Invoke(
            $"[CONFIG] vram_target={targetVramUtilization:F2} reserve_mb={reserveVramMb}");

        // LR ANNEALING — our bootstrap-regime learning applied to the REAL autonomous loop. A fixed
        // SGD step overshoots once most examples are right, so accuracy plateaus around ~90% and
        // OSCILLATES (the reported failure). Shrink the step as the round's success rate climbs, but
        // only NEAR the top — a still-climbing round keeps full steps (annealing a sub-target plateau
        // starves it and freezes it). Captured here, restored in finally so an annealed rate never
        // leaks into the next round (the planner re-plans each round at the base rate).
        var baseLearningRate = _trainer.LearningRate;
        // CONVERGENCE EARLY-STOP — the regime's StabilityWindow idea applied to the real loop. Once a
        // round's prompt-answer success holds at mastery, MORE epochs on a mostly-mastered pool just
        // drift the shared weights (interference) and re-introduce the oscillation; stop instead. The
        // planner's epoch budget is the give-up bound, not a quota to burn.
        const double RoundMasterySuccess = 0.97;
        const int RoundStabilityWindow = 2;
        var convergedStreak = 0;
        var epochsRun = 0;
        try
        {
        for (var e = 1; e <= Math.Max(1, epochs); e++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            epochsRun = e; // actual epochs (≤ requested when early-stopped) — drives the report + planner history
            var epochStart = System.Diagnostics.Stopwatch.StartNew();
            var epochSum = 0.0;
            var epochProcessedExampleCount = 0;
            var batchCount = 0;
            var epochPool = BuildEpochPool(preTokenized, perExampleProgress);

            var batchSize = baseBatchSize;
            var gcInterval = Math.Max(8, batchSize / 2);
            logger?.Invoke(
                $"[EPOCH-CONFIG] {e}/{epochs} pool={epochPool.Count} batch={batchSize} gcInterval={gcInterval}");

            // Process examples in adaptive batches.
            for (var batchStart = 0; batchStart < epochPool.Count;)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var batchEnd = Math.Min(batchStart + batchSize, epochPool.Count);
                var batch = epochPool.GetRange(batchStart, batchEnd - batchStart);

                var batchStepStart = System.Diagnostics.Stopwatch.StartNew();
                
                try
                {
                    var threadId = System.Threading.Thread.CurrentThread.ManagedThreadId;
                    logger?.Invoke($"  [BATCH-START] batch={batchStart + 1:D3}-{batchEnd:D3} size={batch.Count} thread={threadId}");
                    
                    // Validate batch state before passing to trainer
                    if (batch.Count == 0)
                        throw new InvalidOperationException("Batch is empty");
                    
                    foreach (var (idx, item) in batch.Select((x, i) => (i, x)))
                    {
                        if (item.InputTokens == null || item.InputTokens.Length == 0)
                            throw new InvalidOperationException($"Batch[{idx}] InputTokens is null or empty");
                        if (item.TargetTokens == null || item.TargetTokens.Length == 0)
                            throw new InvalidOperationException($"Batch[{idx}] TargetTokens is null or empty");
                    }
                    
                    logger?.Invoke($"  [BATCH-GPU-OP] batch={batchStart + 1:D3}-{batchEnd:D3} GPU_TRAIN_START thread={threadId}");
                    var batchResult = _trainer.TrainBatchPreTokenizedDetailed(batch, cancellationToken);
                    var loss = batchResult.AverageLoss;
                    logger?.Invoke($"  [BATCH-GPU-OP] batch={batchStart + 1:D3}-{batchEnd:D3} GPU_TRAIN_COMPLETE thread={threadId}");
                    batchStepStart.Stop();

                    totalSteps++;
                    batchCount++;
                    sumToken += loss.TokenLoss;
                    sumRoute += loss.RouteLoss;
                    sumConsistency += loss.ConsistencyLoss;
                    sumConservation += loss.ConservationLoss;
                    sumMemory += loss.MemoryLoss;
                    sumTotal += loss.TotalLoss;
                    epochSum += loss.TotalLoss * batch.Count;
                    epochProcessedExampleCount += batch.Count;
                    foreach (var update in batchResult.ExampleLosses)
                    {
                        var key = ComposeExampleKey(update.Example);
                        var trainingKind = update.Example.TrainingKind;
                        var normalizedUpdate = trainingKind == GenesisTrainingExampleKind.PromptAnswer
                            ? update
                            : update with { IsCorrect = false };
                        if (!perExampleProgress.TryGetValue(key, out var entry))
                        {
                            entry = new ExampleProgressAccumulator(update.Example, trainingKind);
                            perExampleProgress[key] = entry;
                        }

                        entry.Observe(normalizedUpdate);
                        if (update.WasSkipped && update.IsCorrect)
                            skippedCorrectExampleCount++;

                        if (trainingKind == GenesisTrainingExampleKind.PromptAnswer)
                            promptAnswerExampleCount += 1;
                        else if (trainingKind == GenesisTrainingExampleKind.WindowedText)
                            windowedTextExampleCount += 1;

                        if (!string.IsNullOrWhiteSpace(update.Example.SourceCreatorName))
                        {
                            var creatorName = update.Example.SourceCreatorName;
                            if (!perCreatorProgress.TryGetValue(creatorName, out var creatorEntry))
                            {
                                creatorEntry = new CreatorProgressAccumulator(creatorName, trainingKind);
                                perCreatorProgress[creatorName] = creatorEntry;
                            }

                            creatorEntry.Observe(normalizedUpdate);
                        }
                    }

                    logger?.Invoke(
                        $"  [BATCH] {batchStart + 1:D3}-{batchEnd:D3} " +
                        $"token={loss.TokenLoss:F4} total={loss.TotalLoss:F4} " +
                        $"cons={loss.ConsistencyLoss:F4} mem={loss.MemoryLoss:F4} " +
                        $"time={batchStepStart.ElapsedMilliseconds:D5}ms");
                    batchStart = batchEnd;
                }
                catch (System.AccessViolationException ex)
                {
                    batchStepStart.Stop();
                    var threadId = System.Threading.Thread.CurrentThread.ManagedThreadId;
                    logger?.Invoke($"  [BATCH-GPU-CRASH] batch={batchStart + 1:D3}-{batchEnd:D3} AccessViolation: {ex.Message} thread={threadId}");
                    logger?.Invoke($"  [BATCH-GPU-CRASH-DETAIL] Exception Type={ex.GetType().FullName} StackTrace={ex.StackTrace}");
                    throw new InvalidOperationException($"Native memory error during batch training at batch {batchStart + 1:D3}-{batchEnd:D3}. This may indicate a GPU memory issue or thread safety violation. Thread={threadId}", ex);
                }
                catch (Exception ex)
                {
                    batchStepStart.Stop();
                    if (IsGpuOutOfMemory(ex) && batchSize > 1)
                    {
                        var newBatchSize = Math.Max(1, batchSize / 2);
                        logger?.Invoke(
                            $"  [BATCH-OOM] {batchStart + 1:D3}-{batchEnd:D3} reducing batch {batchSize} -> {newBatchSize} and retrying.");
                        RunEmergencyMemoryCleanup(logger);
                        batchSize = newBatchSize;
                        gcInterval = Math.Max(4, batchSize / 2);
                        continue;
                    }
                    if (IsGpuOutOfMemory(ex) && batchSize == 1)
                    {
                        logger?.Invoke(
                            $"  [BATCH-OOM] {batchStart + 1:D3}-{batchEnd:D3} persisted at batch=1, skipping example and continuing.");
                        RunEmergencyMemoryCleanup(logger);
                        batchStart = batchEnd;
                        continue;
                    }
                    if (IsAutogradGraphError(ex))
                    {
                        // RESILIENCE for long unattended autonomous runs: a transient TorchSharp autograd
                        // error ("backward through the graph a second time" / freed saved tensors — seen
                        // when the edit-head's separate REINFORCE backward coincides with arithmetic-phase
                        // training) must NOT kill the session. Reset the autograd + optimizer state by
                        // breaking the graph, skip this batch, and carry on — same policy as the OOM path.
                        logger?.Invoke(
                            $"  [BATCH-AUTOGRAD-RECOVER] {batchStart + 1:D3}-{batchEnd:D3} {ex.Message} — resetting graph, skipping batch.");
                        try { _trainer.CloneParametersToBreakGraph(); } catch { }
                        RunEmergencyMemoryCleanup(logger);
                        batchStart = batchEnd;
                        continue;
                    }
                    logger?.Invoke($"  [BATCH-ERROR] {batchStart + 1:D3}-{batchEnd:D3} {ex.GetType().Name}: {ex.Message}");
                    throw;
                }
            }

            epochStart.Stop();

            // Clone parameters ONCE per epoch to break the computation graph chain
            _trainer.CloneParametersToBreakGraph();

            var spaceResult = _trainer.ManagePlatonicSpace();
            spaceManagementCycles++;
            totalNodesPruned += spaceResult.NodesPruned;
            totalRelationsPruned += spaceResult.RelationsPruned;
            lastNoiseRatio = spaceResult.NoiseRatio;
            finalNodeCount = spaceResult.NodesAfter;
            finalRelationCount = spaceResult.RelationsAfter;
            if (spaceResult.Compacted)
            {
                logger?.Invoke(
                    $"[SPACE] observed nodes={spaceResult.NodesAfter} relations={spaceResult.RelationsAfter} " +
                    $"noise={spaceResult.NoiseRatio:F3}");
            }

            var epochLoss = epochSum / Math.Max(1, epochProcessedExampleCount);
            var epochPromptAnswer = perExampleProgress.Values
                .Where(x => x.TrainingKind == GenesisTrainingExampleKind.PromptAnswer)
                .ToArray();
            var epochWindowed = perExampleProgress.Values
                .Where(x => x.TrainingKind == GenesisTrainingExampleKind.WindowedText)
                .ToArray();
            var epochSuccessRate = epochPromptAnswer.Length == 0
                ? 0.0
                : epochPromptAnswer.Average(x => x.SuccessRate);

            // Anneal for the NEXT epoch by how close this round's prompt-answer success is to mastery.
            // Pure windowed-text rounds (no prompt-answer examples) report success 0 → full LR, which
            // is correct: continuous LM loss doesn't have the discrete-accuracy oscillation.
            if (epochPromptAnswer.Length > 0)
            {
                _trainer.LearningRate = baseLearningRate * AnnealedLearningRateFactor(epochSuccessRate);
                logger?.Invoke($"[ANNEAL] {e}/{epochs} success={epochSuccessRate:P0} lr={_trainer.LearningRate:F4}");

                // Stop once mastery holds — don't overtrain a converged pool into interference drift.
                if (epochSuccessRate >= RoundMasterySuccess)
                {
                    if (++convergedStreak >= RoundStabilityWindow)
                    {
                        logger?.Invoke($"[CONVERGED] {e}/{epochs} success={epochSuccessRate:P0} — round mastered, stopping early");
                        break;
                    }
                }
                else
                {
                    convergedStreak = 0;
                }
            }

            var epochWindowedTokenLoss = epochWindowed.Length == 0
                ? 0.0
                : epochWindowed.Average(x => x.AverageTokenLoss);
            logger?.Invoke($"[EPOCH] {e}/{epochs} loss={epochLoss:F4} time={epochStart.ElapsedMilliseconds:D5}ms batches={batchCount}");
            logger?.Invoke(
                $"[EPOCH] {e}/{epochs} prompt_answer_success={epochSuccessRate:P1} " +
                $"windowed_text_token_loss={epochWindowedTokenLoss:F4}");
        }
        }
        finally
        {
            _trainer.LearningRate = baseLearningRate; // never leak an annealed rate to the next round
        }

        overallStart.Stop();
        var avgTokenLoss = sumToken / Math.Max(1, totalSteps);
        var avgRouteLoss = sumRoute / Math.Max(1, totalSteps);
        var avgConsistencyLoss = sumConsistency / Math.Max(1, totalSteps);
        var avgConservationLoss = sumConservation / Math.Max(1, totalSteps);
        var avgMemoryLoss = sumMemory / Math.Max(1, totalSteps);
        var avgTotalLoss = sumTotal / Math.Max(1, totalSteps);
        var contradictionRate = EstimateContradictionRate(examples);
        var trackedExamples = perExampleProgress.Values.ToArray();
        var promptAnswerExamples = trackedExamples
            .Where(x => x.TrainingKind == GenesisTrainingExampleKind.PromptAnswer)
            .ToArray();
        var windowedTextExamples = trackedExamples
            .Where(x => x.TrainingKind == GenesisTrainingExampleKind.WindowedText)
            .ToArray();
        var correctCount = promptAnswerExamples.Count(x => x.SuccessRate >= 0.5);
        var incorrectCount = Math.Max(0, promptAnswerExamples.Length - correctCount);
        var successRate = promptAnswerExamples.Length == 0
            ? 0.0
            : promptAnswerExamples.Average(x => x.SuccessRate);
        var promptAnswerAvgTokenLoss = promptAnswerExamples.Length == 0
            ? 0.0
            : promptAnswerExamples.Average(x => x.AverageTokenLoss);
        var windowedTextAvgTokenLoss = windowedTextExamples.Length == 0
            ? 0.0
            : windowedTextExamples.Average(x => x.AverageTokenLoss);
        var weakExamples = promptAnswerExamples
            .OrderBy(x => x.SuccessRate)
            .ThenByDescending(x => x.AverageTokenLoss)
            .ThenByDescending(x => x.SeenCount)
            .Take(32)
            .Select(x => new GenesisExampleProgress(
                ExampleKey: ComposeExampleKey(x.Example),
                InputPreview: Abbreviate(x.Example.Input, 96),
                OutputPreview: Abbreviate(x.Example.Output, 64),
                SeenCount: x.SeenCount,
                SuccessCount: x.SuccessCount,
                SuccessRate: x.SuccessRate,
                LastTokenLoss: x.LastTokenLoss,
                AverageTokenLoss: x.AverageTokenLoss,
                BestTokenLoss: x.BestTokenLoss))
            .ToArray();
        var creatorProgress = ExampleCreatorRegistry.All
            .Select(creator =>
            {
                if (perCreatorProgress.TryGetValue(creator.Name, out var progress))
                {
                    return new GenesisCreatorProgress(
                        CreatorName: progress.CreatorName,
                        TrainingKind: progress.TrainingKind,
                        SeenCount: progress.SeenCount,
                        SuccessCount: progress.SuccessCount,
                        SuccessRate: progress.SuccessRate,
                        LastTokenLoss: progress.LastTokenLoss,
                        AverageTokenLoss: progress.AverageTokenLoss,
                        BestTokenLoss: progress.BestTokenLoss);
                }

                return new GenesisCreatorProgress(
                    CreatorName: creator.Name,
                    TrainingKind: creator.TrainingKind,
                    SeenCount: 0,
                    SuccessCount: 0,
                    SuccessRate: 0.0,
                    LastTokenLoss: 0.0,
                    AverageTokenLoss: 0.0,
                    BestTokenLoss: 0.0);
            })
            .OrderByDescending(x => x.AverageTokenLoss)
            .ThenBy(x => x.CreatorName, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        logger?.Invoke($"[DONE] Training complete in {overallStart.ElapsedMilliseconds / 1000.0:F1}s. Avg loss={avgTokenLoss:F4}");
        logger?.Invoke(
            $"[DONE] prompt-answer success={successRate:P1} tracked={promptAnswerExamples.Length} weak={weakExamples.Length}");
        logger?.Invoke(
            $"[DONE] token-loss prompt-answer={promptAnswerAvgTokenLoss:F4} windowed-text={windowedTextAvgTokenLoss:F4}");
        if (creatorProgress.Length > 0)
        {
            foreach (var creator in creatorProgress)
            {
                logger?.Invoke(
                    $"[DONE] creator={creator.CreatorName} loss={creator.AverageTokenLoss:F4} success={creator.SuccessRate:P1} seen={creator.SeenCount}");
            }
        }

        return new GenesisTrainingReport(
            Epochs: Math.Max(1, epochsRun),
            ExampleCount: examples.Count,
            AverageLoss: new GenesisStepLoss(
                TokenLoss: avgTokenLoss,
                RouteLoss: avgRouteLoss,
                ConsistencyLoss: avgConsistencyLoss,
                ConservationLoss: avgConservationLoss,
                MemoryLoss: avgMemoryLoss,
                TotalLoss: avgTotalLoss),
            ContradictionRate: contradictionRate,
            ConservationDrift: 0.0,
            MemoryOverwriteRate: totalRelationsPruned / (double)Math.Max(1, examples.Count),
            IntrospectionCycles: 0,
            PendingQueueDepth: 0,
            SpaceManagementCycles: spaceManagementCycles,
            NodesPruned: totalNodesPruned,
            RelationsPruned: totalRelationsPruned,
            FinalNodeCount: finalNodeCount,
            FinalRelationCount: finalRelationCount,
            SpaceNoiseRatio: lastNoiseRatio,
            CorrectExampleCount: correctCount,
            IncorrectExampleCount: incorrectCount,
            ExampleSuccessRate: successRate,
            SkippedCorrectExampleCount: skippedCorrectExampleCount,
            PromptAnswerExampleCount: promptAnswerExampleCount,
            WindowedTextExampleCount: windowedTextExampleCount,
            PromptAnswerAverageTokenLoss: promptAnswerAvgTokenLoss,
            WindowedTextAverageTokenLoss: windowedTextAvgTokenLoss,
            WeakExamples: weakExamples,
            CreatorProgress: creatorProgress);
    }

    private static List<PreTokenizedExample> BuildEpochPool(
        IReadOnlyList<PreTokenizedExample> preTokenized,
        IReadOnlyDictionary<string, ExampleProgressAccumulator> progress)
    {
        if (progress.Count == 0)
            return preTokenized.OrderBy(_ => Random.Shared.Next()).ToList();

        return preTokenized
            .Select(ex =>
            {
                var key = ComposeExampleKey(ex.Original);
                if (!progress.TryGetValue(key, out var entry))
                    return (example: ex, priority: 0.0, jitter: Random.Shared.NextDouble());

                var weakness = (1.0 - entry.SuccessRate) + Math.Min(2.0, entry.AverageTokenLoss);
                return (example: ex, priority: weakness, jitter: Random.Shared.NextDouble());
            })
            .OrderByDescending(x => x.priority)
            .ThenBy(x => x.jitter)
            .Select(x => x.example)
            .ToList();
    }

    private static string ComposeExampleKey(GenesisExample example)
        => $"{example.Input.Trim()} => {example.Output.Trim()}";

    private static string Abbreviate(string value, int maxLength)
    {
        if (string.IsNullOrWhiteSpace(value))
            return string.Empty;

        var compact = value.Replace('\r', ' ').Replace('\n', ' ').Trim();
        return compact.Length <= maxLength ? compact : compact[..maxLength];
    }

    private List<PreTokenizedExample> PreTokenizeExamples(
        IReadOnlyList<GenesisExample> examples,
        int maxTargetTokens,
        Action<string>? logger)
    {
        var result = new List<PreTokenizedExample>(examples.Count);
        var trimmed = 0;
        // Tokenizer mutates vocabulary during Encode, so this must be exclusive.
        // Parallel tokenization corrupts tokenizer dictionaries under concurrent writes.
        foreach (var ex in examples)
        {
            var input = _trainer.EncodeInput(ex.Input);
            var target = _trainer.EncodeTarget(ex.Output);
            if (target.Length > maxTargetTokens)
            {
                target = target.Take(maxTargetTokens).ToArray();
                if (target.Length > 0)
                    target[^1] = _trainer.EosTokenId;
                trimmed++;
            }
            result.Add(new PreTokenizedExample(input, target, ex.TrainingKind, ex));
        }

        if (trimmed > 0)
            logger?.Invoke($"[PRETOK] trimmed {trimmed} targets to max_tokens={maxTargetTokens}");

        return result;
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

    private static bool IsGpuOutOfMemory(Exception ex)
    {
        Exception? current = ex;
        while (current is not null)
        {
            var message = current.Message;
            if (!string.IsNullOrWhiteSpace(message))
            {
                if (message.Contains("out of memory", StringComparison.OrdinalIgnoreCase) ||
                    message.Contains("cuda", StringComparison.OrdinalIgnoreCase) && message.Contains("memory", StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            current = current.InnerException;
        }

        return false;
    }

    // A transient TorchSharp autograd error: "Trying to backward through the graph a second time" or
    // accessing saved tensors after they were freed. Matched by message so the loop can recover (reset
    // the graph/optimizer) and continue instead of dying mid-run.
    private static bool IsAutogradGraphError(Exception ex)
    {
        for (Exception? e = ex; e is not null; e = e.InnerException)
        {
            var m = e.Message;
            if (string.IsNullOrEmpty(m))
                continue;
            if (m.Contains("backward through the graph", StringComparison.OrdinalIgnoreCase) ||
                m.Contains("second time", StringComparison.OrdinalIgnoreCase) ||
                m.Contains("have already been freed", StringComparison.OrdinalIgnoreCase) ||
                (m.Contains("saved", StringComparison.OrdinalIgnoreCase) && m.Contains("freed", StringComparison.OrdinalIgnoreCase)))
                return true;
        }
        return false;
    }

    private void RunEmergencyMemoryCleanup(Action<string>? logger)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }

    private static int ResolveMaxTargetTokens(bool gpuAvailable, int usableVramMb)
    {
        if (!gpuAvailable)
            return 512;

        if (usableVramMb <= 4096)
            return 96;
        if (usableVramMb <= 6144)
            return 128;
        if (usableVramMb <= 8192)
            return 192;

        return 256;
    }
}
