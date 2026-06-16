using GenesisNova.Axioms;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Infer;
using GenesisNova.Model;
using GenesisNova.Tokenization;
using TorchSharp;
using static TorchSharp.torch;

namespace GenesisNova.Train;

public sealed record GenesisPerExampleLoss(
    GenesisExample Example,
    GenesisStepLoss Loss,
    bool IsCorrect,
    bool WasSkipped = false);

public sealed record GenesisBatchTrainResult(
    GenesisStepLoss AverageLoss,
    IReadOnlyList<GenesisPerExampleLoss> ExampleLosses);

public sealed class GenesisTrainer
{
    private readonly IGenesisTokenizer _tokenizer;
    private readonly GenesisNeuralModel _model;
    private readonly PlatonicSpaceMemory _platonicSpace;
    private readonly SpaceManager _spaceManager;
    private GenesisInferenceEngine _inferencePolicy;
    private readonly GenesisCompositeObjective _objective;
    private readonly FoldPathDiscovery _foldPathDiscovery;
    private readonly TransformAccumulator _transformAccumulator;
    private readonly GenesisLabelResolver _labelResolver;
    private readonly Queue<SpacePolicyTransition> _spacePolicyTrajectory = new();
    private int _spacePolicyStepCounter;
    private int _priorSpaceActionId;
    private int _trainStepCount;
    // Speed throttles: the per-example concept-planner GENERATION and the edit-head REINFORCE are the dominant
    // per-example costs. The planner's output is mostly discarded by the coupling (the deterministic mirror
    // dominates — see ObservePlatonicSpace), so run it only every Nth example; reinforce the edit head every
    // Nth example. Both subsystems still contribute, at a fraction of the cost.
    private int _conceptPlanTick;
    private const int ConceptPlanStride = 8;       // run the NN concept-planner 1-in-N examples (else use mirror)
    private const int EditHeadReinforceStride = 4; // reinforce the edit head 1-in-N examples
    // Model-driven edit-head wiring: ObservePlatonicSpace stashes the (explored) magnitude it
    // requested from _model.PredictEditMagnitude (plus the tokens it conditioned on); the surrounding
    // train step then rewards the head via _model.ReinforceEditHead with a CAUSAL, space-state-
    // dependent outcome (does the platonic space now retrieve the correct output for this input?).
    private IReadOnlyList<int>? _pendingEditTokens;
    private double _pendingEditMagnitude;
    private double[]? _pendingEditPerception; // the space-perception vector the edit head conditioned on this step
    private double _cachedConservationLoss;
    // Retrievability of the current example's target BEFORE this step's space writes, snapshotted by
    // ObservePlatonicSpace and consumed by RewardEditHead so the edit head is rewarded by the CAUSAL DELTA
    // its write produced (post − pre) in the same step — dense, immediate, and order-invariant. (Replaces a
    // per-(input+output)-string baseline that the answer-order shuffling defeated → reward was ~always 0.)
    private double _pendingPreEditOutcome;
    private readonly Dictionary<string, int> _conceptCoverageCounts = new(StringComparer.OrdinalIgnoreCase);
    private readonly SpaceDecisionJournal _spaceDecisionJournal = new(MaxSpaceDecisionJournalEntries);
    private readonly Queue<MasteredRehearsalExample> _masteredRehearsalRing = new();
    private readonly Dictionary<int, int> _spacePolicyActionCounters = new();
    private readonly Queue<ConceptPlanDecisionJournalEntry> _conceptPlanDecisionJournal = new();
    private static readonly SpaceToolKind[] SpaceToolActions =
    [
        SpaceToolKind.DefaultAlgorithm,
        SpaceToolKind.Observe,
        SpaceToolKind.Stabilize,
        SpaceToolKind.Expand,
        SpaceToolKind.Rebalance,
        SpaceToolKind.Reinforce,
        SpaceToolKind.CreateConcept,
        SpaceToolKind.EditConceptFace,
        SpaceToolKind.EditRelationContradiction,
        SpaceToolKind.CreateOrStrengthenRelation,
        SpaceToolKind.WeakenOrDecayRelation,
        SpaceToolKind.TriadConsistencyEdit,
        SpaceToolKind.NeighborhoodRetype,
        SpaceToolKind.CentroidPullPush,
        SpaceToolKind.MergeConceptHint,
        SpaceToolKind.PruneHint,
        SpaceToolKind.AnchorBindingEdit,
        SpaceToolKind.AttentionScopeSelect,
        SpaceToolKind.CommitLevelSet,
        SpaceToolKind.RewardTagEmit,
        SpaceToolKind.DiscoverAbstractions
    ];
    private const int MaxSpacePolicyTrajectory = 16;
    private const int MaxConceptCount = 24;
    private const double MinConceptMirrorCoverage = 0.35;
    private const int MaxSpaceDecisionJournalEntries = 256;
    private const int MaxMasteredRehearsalEntries = 64;
    private const int MasteredRehearsalInterval = 25;
    private const double MasteredRehearsalProbability = 0.15;
    private const double MasteredRehearsalLossScale = 0.25;
    private const int MaxConceptPlanDecisionJournalEntries = 256;

    private int _biasAppliedCount;
    private int _biasAppliedCorrectCount;
    private int _biasNotAppliedCount;
    private int _biasNotAppliedCorrectCount;
    private int _platonicConfidenceCorrectCount;
    private int _platonicConfidenceIncorrectCount;
    private double _platonicConfidenceCorrectSum;
    private double _platonicConfidenceIncorrectSum;
    private int _fallbackCount;
    private int _fallbackCorrectCount;
    private int _telemetryObservationCount;
    private int _spaceToolParseFailureCount;
    private readonly Queue<GenesisExample> _pendingInferenceIntrospection = new();
    private const int MaxPendingInferenceIntrospection = 128;
    private InferenceTelemetryHint _inferenceTelemetryHint = InferenceTelemetryHint.Default;
    private int _conceptPlanCalls;
    private int _conceptPlanDirectCount;
    private int _conceptPlanMergedCount;
    private int _conceptPlanFallbackCount;
    private double _conceptPlanCoverageSum;
    private double _conceptPlanNoveltySum;
    private int _masteredSkippedCount;
    private int _masteredRehearsalCount;
    private int _masteredInterleavedRehearsalCount;
    private int _spacePolicyRetrospectiveCreditCount;
    private int _conceptPlanRetrospectiveCreditCount;
    
    // Quality metrics caching (Performance optimization)
    private const int QualityLossRefreshInterval = 24;
    private const int MaxQualityOperationsPerCompute = 12;
    private double? _cachedQualityLoss;
    private int _lastRelationCountForQuality;
    private int _lastQualityComputationStep = int.MinValue;

    public GenesisTrainer(
        IGenesisTokenizer tokenizer,
        GenesisNeuralModel model,
        PlatonicSpaceMemory platonicSpace,
        GenesisNovaConfig? config = null,
        GenesisCompositeObjective? objective = null)
    {
        _tokenizer = tokenizer;
        _labelResolver = new GenesisLabelResolver(tokenizer);
        _model = model;
        _platonicSpace = platonicSpace;
        var runtimeConfig = config ?? new GenesisNovaConfig();
        _foldPathDiscovery = new FoldPathDiscovery();
        _transformAccumulator = new TransformAccumulator(Math.Max(4, _platonicSpace.FaceDimension));
        _spaceManager = new SpaceManager(_platonicSpace, new SpaceManagerSettings(
            Enabled: runtimeConfig.AutoManagePlatonicSpace,
            MaxNodes: Math.Max(256, runtimeConfig.MaxPlatonicNodes),
            MaxRelations: Math.Max(1_024, runtimeConfig.MaxPlatonicRelations)));
        _objective = objective ?? new GenesisCompositeObjective(
            TokenWeight: 1.0,
            RouteWeight: 0.3,
            ConsistencyWeight: 0.1,
            ConservationWeight: 0.1,
            MemoryWeight: 0.05);
        
        _inferencePolicy = new GenesisInferenceEngine(
            tokenizer,
            model,
            platonicSpace,
            transformAccumulator: _transformAccumulator,
            foldPathDiscovery: _foldPathDiscovery);
    }

    public int[] EncodeInput(string input)
        => _tokenizer.Encode(input);

    public int[] EncodeTarget(string output)
        => _tokenizer.Encode(output, addEos: true);

    public int EosTokenId => _tokenizer.EosTokenId;

    // Exposed so the runtime can hand the trained learned-op stores to inference (the learned-op route).
    public TransformAccumulator TransformAccumulator => _transformAccumulator;
    public FoldPathDiscovery FoldPathDiscovery => _foldPathDiscovery;
    public int HiddenSize => _model.HiddenSize;

    /// <summary>Current SGD step size — settable so a regime can anneal it (see CoreBootstrapRegime).</summary>
    public double LearningRate
    {
        get => _model.LearningRate;
        set => _model.LearningRate = value;
    }

    /// <summary>
    /// PUNISH NEURAL: when true (default), a prompt-answer example that is output-correct ONLY via the
    /// NEURAL path (not the platonic tools) does NOT count as correct — so it is not skipped (it keeps
    /// getting trained, weakness-prioritised) and its creator's success stays low (so the autonomous
    /// curriculum won't advance it). The mission is to USE the platonic space, not memorise answers
    /// neurally. Set false to credit any correct output regardless of path.
    /// </summary>
    public bool RequirePlatonicForCorrect { get; set; } = true;

    /// <summary>
    /// SPACE-AWARE EDITING (SPACE_AWARE_GRU.md §B/§F): before writing the cue→answer edge, READ the anchor's
    /// current nearest neighbour; if a NON-target concept is winning (a distractor / a prior bad write), REPEL
    /// it so the answer can become nearest. This is the read-before-write policy — it lets the controller UNDO a
    /// bad write, which the blind (attract-only) edit cannot (measured: blind poison-recovery stalls ~63%).
    /// On by default (all runtimes, incl. the ClaudeMemory daemon): the read-repel is part of every training run;
    /// the learned <see cref="PerceptionEdit"/> magnitude scales it. Set false for legacy attract-only behaviour.
    /// </summary>
    public bool SpaceAwareEdit { get; set; } = true;

    /// <summary>
    /// LEARNED space-awareness (SPACE_AWARE_GRU.md §A/§E): the edit head conditions its magnitude on a perceived
    /// space-state vector (rank-of-target / distractor-winning / nearest-is-target) and that magnitude scales the
    /// distractor REPULSION — so the GRU can LEARN (via the within-step reward) when/how hard to undo a bad write,
    /// rather than the magnitude being hand-coded. On by default (all runtimes): the perception-conditioned edit
    /// head is trained on every run (the within-step reward updates _editPerceptionW). Set false for token-only.
    /// </summary>
    public bool PerceptionEdit { get; set; } = true;

    /// <summary>
    /// POINTER-over-neighbourhood (SPACE_AWARE_GRU.md §D): how many of the anchor's nearest NON-target
    /// neighbours to repel when space-aware editing, instead of only the single nearest. >1 lets the controller
    /// clear several competing distractors at once. Default 3: with <see cref="PerceptionEdit"/> on, the learned
    /// magnitude m chooses HOW MANY of the top-k (1..RepelNeighbors) to clear, so the pointer head can clear
    /// several competitors when its read says to (and stays at 1 when it doesn't). Default 3 (all runtimes).
    /// </summary>
    public int RepelNeighbors { get; set; } = 3;

    public InferenceTelemetryHint InferenceTelemetryHint => _inferenceTelemetryHint;
    public int SpaceToolParseFailureCount => _spaceToolParseFailureCount;
    public int PendingInferenceIntrospectionCount => _pendingInferenceIntrospection.Count;
    public int MasteredRehearsalCount => _masteredRehearsalCount;
    public int MasteredInterleavedRehearsalCount => _masteredInterleavedRehearsalCount;
    
    public GenesisTrainerLearningState ExportLearningState()
        => new(
            SpaceDecisionJournalEntries: _spaceDecisionJournal.Entries,
            MasteredRehearsalRing: _masteredRehearsalRing.ToArray(),
            SpacePolicyActionCounters: new Dictionary<int, int>(_spacePolicyActionCounters),
            SpacePolicyTrajectory: _spacePolicyTrajectory
                .Select(t => new SpacePolicyTransitionSnapshot(
                    t.State.ToDeterministicEncoding(),
                    t.ActionId,
                    t.NoiseRatio,
                    t.AverageBridgeConfidence,
                    t.RelationPressure,
                    t.StepIndex))
                .ToArray(),
            ConceptCoverageCounts: new Dictionary<string, int>(_conceptCoverageCounts, StringComparer.OrdinalIgnoreCase),
            Telemetry: new GenesisTrainerTelemetryState(
                _biasAppliedCount,
                _biasAppliedCorrectCount,
                _biasNotAppliedCount,
                _biasNotAppliedCorrectCount,
                _platonicConfidenceCorrectCount,
                _platonicConfidenceIncorrectCount,
                _platonicConfidenceCorrectSum,
                _platonicConfidenceIncorrectSum,
                _fallbackCount,
                _fallbackCorrectCount,
                _telemetryObservationCount,
                _spaceToolParseFailureCount,
                _conceptPlanCalls,
                _conceptPlanDirectCount,
                _conceptPlanMergedCount,
                _conceptPlanFallbackCount,
                _conceptPlanCoverageSum,
                _conceptPlanNoveltySum,
                _masteredSkippedCount,
                _masteredRehearsalCount,
                _masteredInterleavedRehearsalCount,
                _spacePolicyRetrospectiveCreditCount,
                _conceptPlanRetrospectiveCreditCount),
            SpacePolicyStepCounter: _spacePolicyStepCounter,
            PriorSpaceActionId: _priorSpaceActionId,
            ConceptPlanDecisionJournalEntries: _conceptPlanDecisionJournal.ToArray(),
            TransformAccumulator: _transformAccumulator.ExportSnapshot(),
            FoldPaths: _foldPathDiscovery.ExportSnapshot());

    public void ImportLearningState(GenesisTrainerLearningState? state)
    {
        if (state is null)
            return;

        _spaceDecisionJournal.ReplaceWith((state.SpaceDecisionJournalEntries ?? Array.Empty<SpaceDecisionJournalEntry>()).Take(MaxSpaceDecisionJournalEntries));
        _masteredRehearsalRing.Clear();
        foreach (var example in (state.MasteredRehearsalRing ?? Array.Empty<MasteredRehearsalExample>()).Where(e => !string.IsNullOrWhiteSpace(e.Input) && !string.IsNullOrWhiteSpace(e.Output)).TakeLast(MaxMasteredRehearsalEntries))
            _masteredRehearsalRing.Enqueue(example);

        _spacePolicyActionCounters.Clear();
        foreach (var pair in state.SpacePolicyActionCounters ?? new Dictionary<int, int>())
        {
            if (pair.Key >= 0 && pair.Key <= SpaceToolActions.Length)
                _spacePolicyActionCounters[pair.Key] = Math.Clamp(pair.Value, 0, int.MaxValue);
        }

        _spacePolicyTrajectory.Clear();
        foreach (var item in (state.SpacePolicyTrajectory ?? Array.Empty<SpacePolicyTransitionSnapshot>()).TakeLast(MaxSpacePolicyTrajectory))
        {
            if (!SpacePolicyState.TryFromDeterministicEncoding(item.StateEncoding, out var transitionState))
                continue;
            _spacePolicyTrajectory.Enqueue(new SpacePolicyTransition(
                transitionState,
                Math.Clamp(item.ActionId, 0, SpaceToolActions.Length),
                Math.Clamp(item.NoiseRatio, 0.0, 2.0),
                Math.Clamp(item.AverageBridgeConfidence, 0.0, 1.0),
                Math.Clamp(item.RelationPressure, 0.0, 2.0),
                Math.Max(0, item.StepIndex)));
        }

        _conceptCoverageCounts.Clear();
        foreach (var pair in state.ConceptCoverageCounts ?? new Dictionary<string, int>())
        {
            var concept = NormalizeConceptToken(pair.Key);
            if (concept.Length == 0)
                continue;
            _conceptCoverageCounts[concept] = Math.Clamp(pair.Value, 0, int.MaxValue);
        }

        var telemetry = state.Telemetry;
        if (telemetry is not null)
        {
            _biasAppliedCount = Math.Max(0, telemetry.BiasAppliedCount);
            _biasAppliedCorrectCount = Math.Max(0, telemetry.BiasAppliedCorrectCount);
            _biasNotAppliedCount = Math.Max(0, telemetry.BiasNotAppliedCount);
            _biasNotAppliedCorrectCount = Math.Max(0, telemetry.BiasNotAppliedCorrectCount);
            _platonicConfidenceCorrectCount = Math.Max(0, telemetry.PlatonicConfidenceCorrectCount);
            _platonicConfidenceIncorrectCount = Math.Max(0, telemetry.PlatonicConfidenceIncorrectCount);
            _platonicConfidenceCorrectSum = Math.Max(0.0, telemetry.PlatonicConfidenceCorrectSum);
            _platonicConfidenceIncorrectSum = Math.Max(0.0, telemetry.PlatonicConfidenceIncorrectSum);
            _fallbackCount = Math.Max(0, telemetry.FallbackCount);
            _fallbackCorrectCount = Math.Max(0, telemetry.FallbackCorrectCount);
            _telemetryObservationCount = Math.Max(0, telemetry.TelemetryObservationCount);
            _spaceToolParseFailureCount = Math.Max(0, telemetry.SpaceToolParseFailureCount);
            _conceptPlanCalls = Math.Max(0, telemetry.ConceptPlanCalls);
            _conceptPlanDirectCount = Math.Max(0, telemetry.ConceptPlanDirectCount);
            _conceptPlanMergedCount = Math.Max(0, telemetry.ConceptPlanMergedCount);
            _conceptPlanFallbackCount = Math.Max(0, telemetry.ConceptPlanFallbackCount);
            _conceptPlanCoverageSum = Math.Max(0.0, telemetry.ConceptPlanCoverageSum);
            _conceptPlanNoveltySum = Math.Max(0.0, telemetry.ConceptPlanNoveltySum);
            _masteredSkippedCount = Math.Max(0, telemetry.MasteredSkippedCount);
            _masteredRehearsalCount = Math.Max(0, telemetry.MasteredRehearsalCount);
            _masteredInterleavedRehearsalCount = Math.Max(0, telemetry.MasteredInterleavedRehearsalCount);
            _spacePolicyRetrospectiveCreditCount = Math.Max(0, telemetry.SpacePolicyRetrospectiveCreditCount);
            _conceptPlanRetrospectiveCreditCount = Math.Max(0, telemetry.ConceptPlanRetrospectiveCreditCount);
        }

        _spacePolicyStepCounter = Math.Max(0, state.SpacePolicyStepCounter);
        _priorSpaceActionId = Math.Clamp(state.PriorSpaceActionId, 0, SpaceToolActions.Length);
        _conceptPlanDecisionJournal.Clear();
        foreach (var entry in (state.ConceptPlanDecisionJournalEntries ?? Array.Empty<ConceptPlanDecisionJournalEntry>()).TakeLast(MaxConceptPlanDecisionJournalEntries))
        {
            if (!string.IsNullOrWhiteSpace(entry.Prompt) && entry.SelectedLatentConcepts.Count > 0)
                _conceptPlanDecisionJournal.Enqueue(entry);
        }

        // Restore the learned discovered-transform inference path (lost on reload before this).
        // The runtime constructs the Trainer (fresh empty accumulator/foldpath) THEN calls
        // ImportLearningState THEN wires the inference engine to these same instances, so restoring
        // into the existing instances is correct. Null-guarded so old checkpoints (no snapshot) no-op.
        if (state.TransformAccumulator is not null)
            _transformAccumulator.ImportSnapshot(state.TransformAccumulator);
        if (state.FoldPaths is not null)
            _foldPathDiscovery.ImportSnapshot(state.FoldPaths);
    }

    public void SetInferencePolicy(GenesisInferenceEngine inferencePolicy)
        => _inferencePolicy = inferencePolicy ?? throw new ArgumentNullException(nameof(inferencePolicy));

    public GenesisStepLoss TrainStepPreTokenized(int[] inputTokens, int[] targetTokens)
    {
        _model.EnsureVocabularySize(_tokenizer.VocabularySize);
        _trainStepCount++;

        var inputText = _tokenizer.Decode(inputTokens);
        var outputText = _tokenizer.Decode(targetTokens);
        var example = new GenesisExample(inputText, outputText);
        var weight = ComputeTrainingWeight(example);
        var baseLoss = _model.TrainExample(
            inputTokens,
            targetTokens,
            _tokenizer.BosTokenId,
            lossScale: weight,
            routeLabel: ResolveRouteLabel(example),
            queryLabel: ResolveQueryLabel(inputTokens, example.Output),
            planLabel: ResolvePlanLabel(example.Input, example.Output));
        
        // Break computation graph after training
        _model.CloneParametersToBreakGraph();

        MineComposerShapes(example.Input, example.Output);
        var concepts = ObserveLearningSignals(example);
        RewardEditHead(example, baseLoss.TokenLoss);
        MaybeReinforceRoute(example, inputTokens);

        var consistencyLoss = EstimatePlatonicConsistency(concepts, IsNegativeText(outputText));
        var conservationLoss = EstimateConservationDrift(inputTokens);
        var memoryLoss = EstimateMemoryLoad();
        var routeLoss = EstimateUnifiedPathMismatch(example, targetTokens);

        var total = _objective.ComputeTotal(
            baseLoss.TokenLoss,
            routeLoss,
            consistencyLoss,
            conservationLoss,
            memoryLoss);

        return new GenesisStepLoss(
            baseLoss.TokenLoss,
            routeLoss,
            consistencyLoss,
            conservationLoss,
            memoryLoss,
            total);
    }

    public GenesisStepLoss TrainStep(GenesisExample example)
    {
        var inputTokens = _tokenizer.Encode(example.Input);
        var targetTokens = _tokenizer.Encode(example.Output, addEos: true);
        _model.EnsureVocabularySize(_tokenizer.VocabularySize);
        _trainStepCount++;

        var weight = ComputeTrainingWeight(example);

        var baseLoss = _model.TrainExample(
            inputTokens,
            targetTokens,
            _tokenizer.BosTokenId,
            lossScale: weight,
            routeLabel: ResolveRouteLabel(example),
            queryLabel: ResolveQueryLabel(inputTokens, example.Output),
            planLabel: ResolvePlanLabel(example.Input, example.Output));
         
        // Break computation graph after each example to allow sequential training
        _model.CloneParametersToBreakGraph();

        MineComposerShapes(example.Input, example.Output);
        var concepts = ObserveLearningSignals(example);
        RewardEditHead(example, baseLoss.TokenLoss);
        MaybeReinforceRoute(example, inputTokens);

        var consistencyLoss = EstimatePlatonicConsistency(concepts, IsNegativeText(example.Output));
        var conservationLoss = EstimateConservationDrift(inputTokens);
        var memoryLoss = EstimateMemoryLoad();
        var routeLoss = EstimateUnifiedPathMismatch(example, targetTokens);

        var total = _objective.ComputeTotal(
            baseLoss.TokenLoss,
            routeLoss,
            consistencyLoss,
            conservationLoss,
            memoryLoss);

        return new GenesisStepLoss(
            baseLoss.TokenLoss,
            routeLoss,
            consistencyLoss,
            conservationLoss,
            memoryLoss,
            total);
    }
 
    public void CloneParametersToBreakGraph()
        => _model.CloneParametersToBreakGraph();

    /// <summary>
    /// Single entry point for all inference calls - ensures consistent telemetry observation and adaptive hint application.
    /// ALL inference must use this method to maintain convergence between REPL and training paths.
    /// 
    /// Features:
    /// - Generates output from request
    /// - Determines correctness (explicit override > heuristic > default)
    /// - Reports telemetry to inference engine for adaptive learning
    /// - Computes and applies adaptive hints (bias scale, context gating)
    /// - Tracks observation count for convergence monitoring
    /// - Handles errors gracefully without breaking training
    /// 
    /// Call sites must NOT call ReportTelemetryOutcome or ApplyTelemetryHint separately;
    /// those are handled here to avoid duplication and inconsistency.
    /// </summary>
    private GenerationResult GenerateAndObserveInference(
        GenerationRequest request,
        string contextLabel = "inference",
        bool? expectedCorrect = null)
    {
        try
        {
            // Training-context generation: enable learned-store writes ONLY for the duration of this
            // Generate call. The engine instance is shared with the REPL, which must stay read-only.
            var previousLearning = _inferencePolicy.LearningEnabled;
            _inferencePolicy.LearningEnabled = true;
            GenerationResult result;
            try
            {
                result = _inferencePolicy.Generate(request);
            }
            finally
            {
                _inferencePolicy.LearningEnabled = previousLearning;
            }

            var outcome = expectedCorrect.HasValue
                ? (expectedCorrect.Value ? InferenceOutcome.Success : InferenceOutcome.Failure)
                : DeriveHeuristicOutcome(result);
            // Training-context preview generation: space mutation is allowed (this is the training loop).
            ObserveKnownInferenceOutcome(outcome, result, allowSpaceMutation: true);

            return result;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"[inference] GenerateAndObserveInference({contextLabel}) failed: {ex.Message}");
            // Return empty result to degrade gracefully
            return new GenerationResult(
                Output: string.Empty,
                GeneratedTokens: Array.Empty<int>(),
                UsedPlatonicQuery: false,
                UsedNeuralFallback: false,
                DecisionPath: $"error-{contextLabel}",
                PlatonicConfidence: 0.0,
                AppliedBiasCount: 0,
                AverageBiasMagnitude: 0.0,
                ChunksGenerated: 0,
                PlatonicHopCount: 0);
        }
    }

    /// <summary>
    /// Core public method for single training step inference observation.
    /// </summary>
    public GenesisStepLoss TrainBatchPreTokenized(
        IReadOnlyList<PreTokenizedExample> batch,
        CancellationToken cancellationToken = default)
       => TrainBatchPreTokenizedDetailed(batch, cancellationToken).AverageLoss;

    public GenesisBatchTrainResult TrainBatchPreTokenizedDetailed(
       IReadOnlyList<PreTokenizedExample> batch,
       CancellationToken cancellationToken = default)
    {
       cancellationToken.ThrowIfCancellationRequested();
        
       if (batch.Count == 0)
          return new GenesisBatchTrainResult(
              new GenesisStepLoss(0, 0, 0, 0, 0, 0),
              []);

       // Training can run on CUDA when available. UI responsiveness is maintained by the caller's Task.Run wrapper in MainWindow.
       cancellationToken.ThrowIfCancellationRequested();
        
       _model.EnsureVocabularySize(_tokenizer.VocabularySize);

       var totalTokenLoss = 0.0;
       var totalTokenWeight = 0.0;
       var consistencySum = 0.0;
       var memorySum = 0.0;
       var conservationSum = 0.0;
       var routeSum = 0.0;
       var qualitySum = 0.0;
       var skippedCorrectCount = 0;
       var perExample = new List<GenesisPerExampleLoss>(batch.Count);
       var gcInterval = Math.Max(8, Math.Min(32, Math.Max(1, batch.Count / 2)));

       // Train each example and break graph after each training step.
       // This is the safest boundary against autograd graph reuse during long/high-throughput runs.
       for (int i = 0; i < batch.Count; i++)
       {
          cancellationToken.ThrowIfCancellationRequested();
           
          var item = batch[i];
          var inputTokens = item.InputTokens;
          var targetTokens = item.TargetTokens;
          var example = item.Original;
          var weight = ComputeTrainingWeight(example);
          var concepts = ObserveLearningSignals(example);
          var shouldSkip = ShouldSkipTrainingExample(example, targetTokens);
          TrainingLoss loss;
          var effectiveWeight = weight;
          if (shouldSkip)
          {
             skippedCorrectCount++;
             _masteredSkippedCount++;
             AddMasteredRehearsalExample(example);
             if (Random.Shared.NextDouble() < MasteredRehearsalProbability)
             {
                effectiveWeight = weight * MasteredRehearsalLossScale;
                _masteredRehearsalCount++;
                loss = _model.TrainExample(
                   inputTokens,
                   targetTokens,
                   _tokenizer.BosTokenId,
                   lossScale: effectiveWeight,
                   routeLabel: ResolveRouteLabel(example),
            queryLabel: ResolveQueryLabel(inputTokens, example.Output),
            planLabel: ResolvePlanLabel(example.Input, example.Output));
                totalTokenLoss += loss.TokenLoss * targetTokens.Length * effectiveWeight;
                totalTokenWeight += targetTokens.Length * effectiveWeight;
             }
             else
             {
                loss = new TrainingLoss(0, 0);
             }
          }
          else
          {
             loss = _model.TrainExample(
                inputTokens,
                targetTokens,
                _tokenizer.BosTokenId,
                lossScale: effectiveWeight,
                routeLabel: ResolveRouteLabel(example),
            queryLabel: ResolveQueryLabel(inputTokens, example.Output),
            planLabel: ResolvePlanLabel(example.Input, example.Output));
             totalTokenLoss += loss.TokenLoss * targetTokens.Length * effectiveWeight;
             totalTokenWeight += targetTokens.Length * effectiveWeight;
          }
                  
          _trainStepCount++;
          MaybeInterleaveMasteredRehearsal();

          var consistencyLoss = EstimatePlatonicConsistency(concepts, IsNegativeText(example.Output));
          var conservationLoss = EstimateConservationDrift(inputTokens);
          var memoryLoss = EstimateMemoryLoad();
          var routeLoss = EstimateUnifiedPathMismatch(example, targetTokens);
          var qualityLoss = ComputeQualityLoss();
          consistencySum += consistencyLoss;
          conservationSum += conservationLoss;
          memorySum += memoryLoss;
          routeSum += routeLoss;
          qualitySum += qualityLoss;
          var stepTotal = (_objective.ComputeTotal(
              loss.TokenLoss,
              routeLoss,
              consistencyLoss,
              conservationLoss,
              memoryLoss) * 0.9) + (qualityLoss * 0.1);
          var stepLoss = new GenesisStepLoss(
              loss.TokenLoss,
              routeLoss,
              consistencyLoss,
              conservationLoss,
              memoryLoss,
              stepTotal);
          var isCorrect = shouldSkip;
          perExample.Add(new GenesisPerExampleLoss(example, stepLoss, isCorrect, shouldSkip));
           
          // Collect gen-0 only periodically to keep memory stable without full collection pauses.
          if ((i + 1) % gcInterval == 0)
          {
              GC.Collect(0, GCCollectionMode.Optimized);
          }

          _model.CloneParametersToBreakGraph();
          // Edit-head REINFORCE backward runs AFTER the main autograd graph is broken, so the two
          // backwards never share graph state — the "backward through the graph a second time" trigger
          // that surfaced at the arithmetic phase of long autonomous runs. ReinforceEditHead re-encodes a
          // detached snapshot internally, so it doesn't need the (now freed) main graph.
          // SPEED: it's a second forward/backward per example — reinforce only 1-in-N examples.
          if (_trainStepCount % EditHeadReinforceStride == 0)
          {
             RewardEditHead(example, loss.TokenLoss);
             MaybeReinforceRoute(example, inputTokens);
          }
       }

       var avgTokenLoss = totalTokenLoss / Math.Max(1.0, totalTokenWeight);
       var avgConsistency = consistencySum / Math.Max(1, batch.Count);
       var avgConservation = conservationSum / Math.Max(1, batch.Count);
       var avgMemory = memorySum / Math.Max(1, batch.Count);
       var avgRoute = routeSum / Math.Max(1, batch.Count);
       var avgQuality = qualitySum / Math.Max(1, batch.Count);
        
       // Integrate quality loss into total: 10% weight
       var total = (_objective.ComputeTotal(avgTokenLoss, avgRoute, avgConsistency, avgConservation, avgMemory) * 0.9) +
                   (avgQuality * 0.1);

       // Light gen-0 collection at batch boundary to free recent allocations without pausing.
       GC.Collect(0, GCCollectionMode.Optimized);

       return new GenesisBatchTrainResult(
           new GenesisStepLoss(
               avgTokenLoss,
               avgRoute,
               avgConsistency,
               avgConservation,
               avgMemory,
               total),
           perExample);
    }

    // Generation headroom for correctness checks. We generate a few tokens BEYOND the answer length so
    // a model that emits the right answer but then fails to STOP (no EOS) is caught here — exactly as it
    // manifests in the REPL, which generates open-ended. Capping generation at the answer length only
    // validated the prefix and let rambling models score as "trained-correct" while failing in the REPL.
    private const int CorrectnessStopProbeTokens = 4;

    private bool ShouldSkipTrainingExample(GenesisExample example, IReadOnlyList<int> targetTokens)
    {
        var expected = targetTokens
           .Where(t => t != _tokenizer.BosTokenId && t != _tokenizer.EosTokenId && t != _tokenizer.PadTokenId)
           .ToArray();
        if (expected.Length == 0)
           return false;

        // Same engine instance the REPL uses (see SetInferencePolicy); the only thing that previously
        // differed was the token budget. Give it headroom so the model must stop at the right place,
        // then require an EXACT full match — identical to how the REPL output is judged correct/incorrect.
        var preview = GenerateAndObserveInference(
            new GenerationRequest(example.Input, expected.Length + CorrectnessStopProbeTokens),
            contextLabel: "skip-decision");
        var predicted = preview.GeneratedTokens
           .Where(t => t != _tokenizer.BosTokenId && t != _tokenizer.EosTokenId && t != _tokenizer.PadTokenId)
           .ToArray();
        // FACE-AWARE return grading: compare by platonic VALUE, not raw tokens, so a digit and its
        // number-word are both correct ("2" ≡ "two") — honouring the learned equivalence at the gate.
        // Decode folds digit-run tokens ("1","8" → "18"); the oracle (AnswerEquivalence) uses ground
        // truth, so it never lets the model self-grade. Non-numeric answers still require exact match.
        var isCorrect = AnswerEquivalence.Equivalent(_tokenizer.Decode(predicted), _tokenizer.Decode(expected));

        // PUNISH NEURAL: a prompt-answer that's right only via the NEURAL path is NOT counted correct.
        // It then isn't skipped (keeps training) and doesn't credit the creator's success — pressing
        // the model to answer via the platonic tools. Windowed-text has no platonic path, so it is
        // exempt. (A platonic answer sets UsedPlatonicQuery and clears UsedNeuralFallback.)
        if (isCorrect
            && RequirePlatonicForCorrect
            && example.TrainingKind == GenesisTrainingExampleKind.PromptAnswer
            && !(preview.UsedPlatonicQuery && !preview.UsedNeuralFallback))
        {
            isCorrect = false;
        }

        return isCorrect;
    }

    private double EstimateUnifiedPathMismatch(GenesisExample example, IReadOnlyList<int> targetTokens)
    {
       var expected = targetTokens
           .Where(t => t != _tokenizer.EosTokenId && t != _tokenizer.BosTokenId && t != _tokenizer.PadTokenId)
           .ToArray();
       if (expected.Length == 0)
           return 0.0;

       // Only run inference on every 4th example to reduce memory pressure during long training runs.
       // Random sampling ensures representativity over time while reducing GC churn from tensor creation.
       if (_trainStepCount % 4 != 0)
           return 0.0;  // Skip inference, accept zero loss contribution for this example

       try
       {
           var preview = GenerateAndObserveInference(
               new GenerationRequest(example.Input, expected.Length + CorrectnessStopProbeTokens),
               contextLabel: "unified-path-mismatch",
               expectedCorrect: null);  // Let heuristic determine correctness
           var mismatch = ComputeNormalizedTokenDistance(preview.GeneratedTokens, expected);
           return mismatch;
       }
       catch (Exception ex)
       {
           System.Diagnostics.Debug.WriteLine($"[inference] EstimateUnifiedPathMismatch failed: {ex.Message}");
           return 0.0;  // Degrade gracefully on memory pressure
       }
    }

    private static double ComputeNormalizedTokenDistance(IReadOnlyList<int> predicted, IReadOnlyList<int> expected)
    {
       if (expected.Count == 0)
           return 0.0;

       var overlap = Math.Min(predicted.Count, expected.Count);
       var mismatches = 0;
       for (var i = 0; i < overlap; i++)
       {
           if (predicted[i] != expected[i])
               mismatches++;
       }

       mismatches += Math.Abs(predicted.Count - expected.Count);
       return Clamp01((double)mismatches / Math.Max(1, expected.Count));
    }

    public double ComputeSchedulingPriority(GenesisExample example)
    {
       var weight = ComputeTrainingWeight(example);
       var concepts = ResolveConceptsReadOnly(example);
       if (concepts.Count == 0)
           return weight;

       var rarityBoost = concepts
           .Select(c => _conceptCoverageCounts.TryGetValue(c, out var count) ? count : 0)
           .Select(count => 1.0 / (1.0 + count))
           .DefaultIfEmpty(1.0)
           .Average();
       return weight + (0.6 * rarityBoost);
    }

    private double EstimateConservationDrift(IReadOnlyList<int> inputTokens)
    {
        if (_trainStepCount % 128 != 0 && _cachedConservationLoss > 0)
            return _cachedConservationLoss;

        var snapshot = _model.Export();
        var emb = snapshot.Embeddings;
        if (emb.Length == 0)
            return 0.0;

        var rows = emb.GetLength(0);
        var cols = emb.GetLength(1);

        var sum = 0.0;
        var count = 0;
        foreach (var token in inputTokens.Distinct())
        {
            if (token < 0 || token >= rows)
                continue;
            for (var h = 0; h < cols; h++)
            {
                sum += Math.Abs(emb[token, h]);
                count++;
            }
        }

        if (count == 0)
        {
            var sampleRows = Math.Min(16, rows);
            for (var i = 0; i < sampleRows; i++)
            {
                var row = (i * 997 + _trainStepCount) % rows;
                for (var h = 0; h < cols; h++)
                {
                    sum += Math.Abs(emb[row, h]);
                    count++;
                }
            }
        }

        _cachedConservationLoss = (sum / Math.Max(1, count)) * 0.001;
        return _cachedConservationLoss;
    }

    private IReadOnlyList<string> ResolveConceptsReadOnly(GenesisExample example)
        => ResolveConcepts(example.Input, example.Output);

    private IReadOnlyList<string> ResolveConcepts(GenesisExample example)
        => ResolveConcepts(example.Input, example.Output);

    private IReadOnlyList<string> ResolveConcepts(string input, string output)
    {
        var mirror = ExtractMirrorConcepts(input, output);
        // SPEED: the concept-planner below is a full per-example generation whose result the coupling mostly
        // discards (the mirror dominates). Run it only 1-in-N examples; use the deterministic mirror otherwise.
        if (mirror.Count > 0 && (++_conceptPlanTick % ConceptPlanStride) != 0)
            return mirror;
        var prompt = BuildConceptPlanPrompt(input, output, mirror);
        var maxTokens = Math.Clamp(8 + (mirror.Count * 2), 8, 48);
        var generated = GenerateAndObserveInference(
            new GenerationRequest(
                Input: prompt,
                MaxNewTokens: maxTokens,
                ChunkTokenBudget: Math.Min(16, maxTokens)),
            contextLabel: "concept-plan",
            expectedCorrect: null);
         
        var planned = ParseConceptPlan(generated.Output);
        var mode = ConceptPlanSelectionMode.PlannedDirect;
        IReadOnlyList<string> selected;
        double mirrorCoverage;
        double noveltyRatio;
        if (planned.Count == 0)
        {
            mode = ConceptPlanSelectionMode.MirrorFallback;
            selected = mirror;
            mirrorCoverage = mirror.Count > 0 ? 1.0 : 0.0;
            noveltyRatio = 0.0;
        }
        else if (mirror.Count == 0)
        {
            selected = planned;
            mirrorCoverage = 1.0;
            noveltyRatio = 0.0;
        }
        else
        {
            var plannedSet = planned.ToHashSet(StringComparer.OrdinalIgnoreCase);
            var overlap = mirror.Count(concept => plannedSet.Contains(concept));
            mirrorCoverage = overlap / (double)Math.Max(1, mirror.Count);
            var extras = planned.Count(concept => !mirror.Contains(concept, StringComparer.OrdinalIgnoreCase));
            noveltyRatio = planned.Count == 0 ? 0.0 : extras / (double)planned.Count;
            if (mirrorCoverage >= MinConceptMirrorCoverage)
            {
                selected = planned;
            }
            else
            {
                mode = ConceptPlanSelectionMode.Merged;
                selected = mirror
                    .Concat(planned)
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .Take(MaxConceptCount)
                    .ToArray();
                var mergedExtras = selected.Count(concept => !mirror.Contains(concept, StringComparer.OrdinalIgnoreCase));
                noveltyRatio = selected.Count == 0 ? 0.0 : mergedExtras / (double)selected.Count;
            }
        }

        ObserveConceptPlanTelemetry(new ConceptPlanTelemetry(
            Prompt: prompt,
            ModelOutput: generated.Output ?? string.Empty,
            SelectedConcepts: selected,
            Mode: mode,
            MirrorCoverage: Math.Clamp(mirrorCoverage, 0.0, 1.0),
            NoveltyRatio: Math.Clamp(noveltyRatio, 0.0, 1.0)));

        return selected;
    }

    // Records concept-plan resolution telemetry only. The former policy-gradient training that
    // taught the answer token-LM to emit concept-plan tokens (which corrupted the answer weights)
    // has been removed; concept RESOLUTION itself (mirror/planned selection above) is unchanged.
    private void ObserveConceptPlanTelemetry(ConceptPlanTelemetry telemetry)
    {
        _conceptPlanCalls++;
        _conceptPlanCoverageSum += telemetry.MirrorCoverage;
        _conceptPlanNoveltySum += telemetry.NoveltyRatio;
        switch (telemetry.Mode)
        {
            case ConceptPlanSelectionMode.PlannedDirect:
                _conceptPlanDirectCount++;
                break;
            case ConceptPlanSelectionMode.Merged:
                _conceptPlanMergedCount++;
                break;
            default:
                _conceptPlanFallbackCount++;
                break;
        }
    }

    private static string BuildConceptPlanPrompt(
        string input,
        string output,
        IReadOnlyList<string> mirror)
    {
        var mirrorSeed = mirror.Count == 0
            ? "unknown"
            : string.Join(' ', mirror.Take(8));
        return
            "concept-plan\n" +
            $"input: {input}\n" +
            $"output: {output}\n" +
            $"mirror-seed: {mirrorSeed}\n" +
            "emit lowercase concept tokens separated by spaces; include mirrored input concepts, and optionally add useful latent concepts.";
    }

    private static IReadOnlyList<string> ParseConceptPlan(string output)
    {
        if (string.IsNullOrWhiteSpace(output))
            return Array.Empty<string>();

        var parsed = output
            .ToLowerInvariant()
            .Split(new[] { ' ', '\t', '\r', '\n', ',', ';', '|', ':', '[', ']', '(', ')' }, StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
            .Select(NormalizeConceptToken)
            .Where(static token => token.Length > 0)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(MaxConceptCount)
            .ToArray();
        return parsed;
    }

    // Instance (not static) so it can consult the space's op-token registry: an op-token (e.g. "find") is a
    // ROUTE TRIGGER, never a relation concept — dropping it here keeps it out of input→output coupling AND out
    // of the retrieval anchor (inputConcepts[0]), so it can never overload a target and collapse retrieval to
    // one key. The route head still learns it from the raw input tokens. (See PlatonicSpaceMemory op-tokens.)
    private IReadOnlyList<string> ExtractMirrorConcepts(string input, string output)
    {
        var words = $"{input} {output}"
            .ToLowerInvariant()
            .Split(' ', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
            .Select(NormalizeConceptToken)
            .Where(static w => w.Length > 0)
            .Where(w => !_platonicSpace.IsOperationToken(w))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(MaxConceptCount)
            .ToArray();

        return words.Length == 0 ? ["unknown"] : words;
    }

    /// <summary>
    /// Resolves the route-classification supervision label for an example. Honors an explicit
    /// <see cref="GenesisExample.RouteLabel"/> when present; otherwise derives one from CORRECTNESS
    /// (not surface form): the PLATONIC route is supervised only when the platonic path actually
    /// reproduces the target — generalized across ALL modalities, not just arithmetic.
    /// 0 = neural-only, 1 = platonic-direct, 2 = platonic-assisted.
    ///   - Arithmetic whose computed result matches the target, bare-number output  -> 1 (direct).
    ///   - Arithmetic whose computed result matches the target, worded output       -> 2 (assisted).
    ///   - Arithmetic present but the computed result is WRONG for this example     -> 0 (neural).
    ///   - Non-arithmetic, platonic RETRIEVAL reproduces the target verbatim/near   -> 1 (direct).
    ///   - Non-arithmetic, platonic retrieves the target concept (top-K / chain)    -> 2 (assisted).
    ///   - Non-arithmetic, platonic path does NOT reproduce the target              -> 0 (neural).
    ///   - Undeterminable                                                           -> null.
    /// Bounded/deterministic and cheap: it only queries the space (GetNearestConcepts /
    /// QueryConceptChain), it does NOT run heavy generation. Additive and backward-compatible:
    /// returns the existing label untouched when one is supplied.
    /// </summary>
    private int? ResolveRouteLabel(GenesisExample example)
    {
        if (example.RouteLabel is >= 0 and < GenesisNova.Model.GenesisNeuralModel.RouteCount)
            return example.RouteLabel;
        if (example.RouteLabel.HasValue)
            return null; // out-of-range stored label — ignore rather than mislabel.

        // Arithmetic modality: only supervise a platonic route when the computation reproduces the target.
        if (TryExtractArithmeticObservation(example, out var arithmetic))
        {
            if (arithmetic.AbsoluteError > 1e-6)
                return null; // malformed/noisy example — don't supervise the router toward anything.

            // Correct: direct when the answer is a bare number, assisted when embedded in a worded answer.
            var arithmeticOutput = example.Output ?? string.Empty;
            var bareNumberOut = System.Text.RegularExpressions.Regex.IsMatch(
                arithmeticOutput, @"^\s*-?\d+(?:\.\d+)?\s*$");
            return bareNumberOut ? 1 : 2;
        }

        // Predicate modality (comparison answers): the Compare/Branch glider computes it on the substrate,
        // exact and space-independent → platonic-direct (route 1). The plan head learns to route it there.
        var predOut = (example.Output ?? string.Empty).Trim().ToLowerInvariant();
        if (predOut is "greater" or "less" or "equal")
            return 1;

        // Arithmetic→word (numeric operands, number-word output): the glider plan computes the value and
        // Hop-formats it to its word via the learned digit↔word edge — platonic-direct (route 1).
        var rawArithToks = (example.Input ?? string.Empty).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var numOpsForWord = rawArithToks.Count(t => double.TryParse(t,
            System.Globalization.NumberStyles.AllowLeadingSign | System.Globalization.NumberStyles.AllowDecimalPoint,
            System.Globalization.CultureInfo.InvariantCulture, out _));
        if (numOpsForWord >= 2 && GenesisNova.Core.NumberWordVocabulary.WordToValue.ContainsKey(predOut))
            return 1;

        // Fold (variadic reduce ≥3 operands whose sum/product equals the output): the substrate reduces
        // them via R2 compose → platonic-direct (route 1).
        var foldNs = System.Globalization.NumberStyles.AllowLeadingSign | System.Globalization.NumberStyles.AllowDecimalPoint;
        var foldInv = System.Globalization.CultureInfo.InvariantCulture;
        if (numOpsForWord >= 3 && double.TryParse(predOut, foldNs, foldInv, out var foldRouteV))
        {
            var fv = rawArithToks.Where(t => double.TryParse(t, foldNs, foldInv, out _))
                                 .Select(t => double.Parse(t, foldNs, foldInv)).ToList();
            if (Math.Abs(fv.Sum() - foldRouteV) < 1e-6 || Math.Abs(fv.Aggregate(1.0, (s, v) => s * v) - foldRouteV) < 1e-6)
                return 1;
        }

        // SEQ (Concatenate-Composition) and REF (higher-order) shapes: the glider plan assembles + runs them
        // on the substrate (Literal∘Compute / Ref→larger ×2), exact and space-independent → platonic-direct
        // (route 1). Same structure detectors the plan-label uses, so route + plan agree.
        if (numOpsForWord >= 2 && GenesisLabelResolver.TrySeqSegments(rawArithToks, predOut, out _))
            return 1;
        if (numOpsForWord == 2 && GenesisLabelResolver.IsTwiceLarger(rawArithToks, predOut))
            return 1;

        // General (non-arithmetic) modality: the universal platonic op is RETRIEVAL. Label the
        // platonic route whenever the space reproduces the target for this input — relational,
        // functional, lookup, etc. Cheap deterministic space queries only (no generation).
        var input = example.Input ?? string.Empty;
        var output = example.Output ?? string.Empty;
        if (string.IsNullOrWhiteSpace(input) || string.IsNullOrWhiteSpace(output))
            return null; // undeterminable

        var inputConcepts = ExtractMirrorConcepts(input, string.Empty);
        var outputConcepts = ExtractMirrorConcepts(output, string.Empty);
        if (inputConcepts.Count == 0 || outputConcepts.Count == 0)
            return null; // undeterminable

        var outputTokens = TokenizeForRouteMatch(output);
        if (outputTokens.Count == 0)
            return null;

        // (a) Concept-chain retrieval: does the lattice walk from the input anchors reproduce the
        // target text? Exact/near-exact reproduction is a DIRECT platonic answer (route 1) when the
        // target is a single token, ASSISTED (route 2) when it is a worded multi-token reconstruction.
        var chain = _platonicSpace.QueryConceptChain(inputConcepts, maxHops: 2, beamWidth: 2);
        var chainTokens = TokenizeForRouteMatch(chain.Text);
        if (chainTokens.Count > 0)
        {
            var chainCoverage = TokenCoverage(outputTokens, chainTokens);
            if (chainCoverage >= 0.999)
                return outputTokens.Count <= 1 ? 1 : 2;
            if (chainCoverage >= 0.35)
                return 2; // partial relational reconstruction → assisted (warmed: supervise platonic earlier,
                          // as the relation builds, instead of waiting for near-complete reconstruction).
        }

        // (b) Nearest-concept retrieval: is the target's concept the nearest neighbour of the
        // primary input concept (graded by rank)? Nearest-as-single-hop is DIRECT (route 1); present
        // in the top-K but not the single nearest is ASSISTED (route 2).
        var primaryInput = inputConcepts[0];
        var nearest = _platonicSpace.GetNearestConcepts(
            primaryInput,
            candidates: null,
            maxNeighbors: 8);
        if (nearest.Count > 0)
        {
            var outputConceptSet = new HashSet<string>(
                outputConcepts.Select(c => c.ToLowerInvariant()),
                StringComparer.OrdinalIgnoreCase);
            for (var rank = 0; rank < nearest.Count; rank++)
            {
                if (outputConceptSet.Contains(nearest[rank].Symbol))
                    return rank == 0 ? 1 : 2;
            }
        }

        // Platonic path did not reproduce the target — most often because the space simply isn't
        // populated yet. Do NOT label this neural: that poisons the router into an always-neural
        // attractor (empty space → "neural" labels → router never tries platonic → space never gets
        // exercised → stays empty). Leave it UNSUPERVISED. Positive platonic labels accrue as the
        // space learns to reproduce these inputs, and platonic routes already fall back to neural
        // gracefully when a tool yields nothing — so under-labelling platonic costs nothing.
        return null;
    }

    /// <summary>
    /// Derives the PLAN-head label (which block-composition SHAPE) from the example's OWN structure — no
    /// surface grammar: 1=arithmetic (≥2 numeric operands → numeric output), 2=predicate (comparison output
    /// greater/less/equal), 3=retrieval (non-numeric concept input → non-numeric output), 0=none. The GRU
    /// learns to predict this so the learned composer assembles the right glider; null = don't supervise.
    /// </summary>
    /// <summary>
    /// Element-native mining: training targets are graded-correct by construction, so a Seq-structured output
    /// (scaffold words + the operands' sum) contributes its scaffold to the platonic CHUNK-ELEMENT STORE. The
    /// Seq composer later RETRIEVES the most-reinforced scaffold from the space — the cache/binding idea done
    /// element-natively — instead of a literal hardcoded in inference. Only Seq examples mine; others no-op.
    /// </summary>
    private void MineComposerShapes(string? input, string? output)
    {
        if (string.IsNullOrWhiteSpace(input) || string.IsNullOrWhiteSpace(output))
            return;
        if (_labelResolver.TryGetSeqScaffold(input, output, out var scaffold) && !string.IsNullOrWhiteSpace(scaffold))
            _platonicSpace.MineChunk(PlatonicSpaceMemory.SeqScaffoldTag, scaffold);
    }

    private int? ResolvePlanLabel(string? input, string? output)
        => _labelResolver.ResolvePlanLabel(input, output);

    /// <summary>
    /// Derives supervision for the GRU's platonic-query construction heads from the example's OWN
    /// numeric structure — no surface grammar. The tokenizer emits each digit as its own token, so
    /// operands are MAXIMAL DIGIT RUNS in the input token sequence. With exactly two runs (L, R) and
    /// a numeric output O, the operation is whichever single face op satisfies op(L, R) == O
    /// (add/sub/mul/div). Framing tokens ("what", "is", "plus", "?") are operand-head negatives —
    /// which is precisely how the GRU LEARNS to ignore irrelevant text. Ambiguous (multiple ops
    /// match) or non-conforming examples return null: unsupervised, costs nothing.
    /// </summary>
    internal GenesisQueryLabel? ResolveQueryLabel(IReadOnlyList<int> inputTokens, string? output)
        => _labelResolver.ResolveQueryLabel(inputTokens, output);

    private static IReadOnlyList<string> TokenizeForRouteMatch(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return Array.Empty<string>();
        return text
            .ToLowerInvariant()
            .Split(new[] { ' ', '\t', '\r', '\n', ',', ';', '.', ':', '|', '/', '\\', '(', ')', '[', ']' },
                StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
    }

    private static double TokenCoverage(IReadOnlyList<string> targetTokens, IReadOnlyList<string> candidateTokens)
    {
        if (targetTokens.Count == 0)
            return 0.0;
        var candidateSet = new HashSet<string>(candidateTokens, StringComparer.OrdinalIgnoreCase);
        var hits = targetTokens.Count(t => candidateSet.Contains(t));
        return hits / (double)targetTokens.Count;
    }

    private double ComputeTrainingWeight(GenesisExample example)
    {
        var weight = 1.0;
        var concepts = ResolveConceptsReadOnly(example);

        if (IsNegativeText(example.Output))
            weight += 1.0;

        if (concepts.Count >= 2)
        {
            var target = IsNegativeText(example.Output) ? 0.75 : 0.25;
            var current = EstimateAverageContradiction(concepts);
            var mismatch = Math.Abs(current - target);
            weight += 0.5 * Clamp01(mismatch);
        }

        return Math.Clamp(weight, 0.5, 4.0);
    }

    private static bool IsNegativeText(string text)
        => text.TrimStart().StartsWith("not ", StringComparison.OrdinalIgnoreCase);

    public void ObserveInferenceResult(string input, string output)
        => ObserveInferenceResult(input, output, telemetry: null);

    public void ObserveInferenceResult(string input, string output, GenerationResult? telemetry)
        => ObserveInferenceResult(input, output, telemetry, isCorrect: null);

    public void ObserveInferenceResult(string input, GenerationResult result, bool? isCorrect = null)
        => ObserveInferenceResult(input, result.Output, result, isCorrect);

    private void ObserveInferenceResult(string input, string output, GenerationResult? telemetry, bool? isCorrect)
    {
        if (string.IsNullOrWhiteSpace(input) && string.IsNullOrWhiteSpace(output))
            return;

        var example = new GenesisExample(input ?? string.Empty, output ?? string.Empty);
        var outcome = ResolveInferenceOutcome(example, telemetry, isCorrect);

        // Inference is READ-ONLY on the platonic space. The space is a SHARED, PERSISTENT store that
        // every future training session loads, so it is mutated ONLY by the training loop
        // (ObserveLearningSignals). Drifting it from REPL/inference queries — especially with the
        // edit-head's exploration noise, and from the model's own (possibly wrong) outputs — would
        // contaminate the shared store across all sessions. Here we only record telemetry (the
        // inference engine's own adaptive bias) and, for an undeterminable outcome, queue the example
        // for the TRAINING loop to learn from later.
        if (outcome == InferenceOutcome.Unknown)
            EnqueuePendingInferenceIntrospection(example);

        if (telemetry is not null)
            ObserveInferenceTelemetry(telemetry, outcome);
    }

    private void ObserveInferenceTelemetry(GenerationResult telemetry, InferenceOutcome outcome)
    {
        // allowSpaceMutation: false — REPL/inference telemetry adapts the inference bias but must NOT
        // write to the shared, persistent platonic space (that is training-only).
        ObserveKnownInferenceOutcome(outcome, telemetry, allowSpaceMutation: false);
    }

    private InferenceOutcome ResolveInferenceOutcome(GenesisExample example, GenerationResult? telemetry, bool? isCorrect)
    {
        if (isCorrect.HasValue)
            return isCorrect.Value ? InferenceOutcome.Success : InferenceOutcome.Failure;

        var arithmeticCorrectness = TryResolveCorrectness(example);
        if (arithmeticCorrectness.HasValue)
            return arithmeticCorrectness.Value ? InferenceOutcome.Success : InferenceOutcome.Failure;

        return telemetry is null
            ? InferenceOutcome.Unknown
            : DeriveHeuristicOutcome(telemetry);
    }

    // allowSpaceMutation defaults to FALSE — the safe option. Only the training loop opts in (true),
    // so the shared persistent platonic space can never be written from an inference/REPL caller.
    private void ObserveKnownInferenceOutcome(InferenceOutcome outcome, GenerationResult telemetry, bool allowSpaceMutation = false)
    {
        if (outcome == InferenceOutcome.Unknown)
            return;

        var resolvedCorrectness = outcome == InferenceOutcome.Success;
        _inferencePolicy.ReportTelemetryOutcome(resolvedCorrectness, telemetry);

        _telemetryObservationCount++;

        if (telemetry.AppliedBiasCount > 0)
        {
            _biasAppliedCount++;
            if (resolvedCorrectness)
                _biasAppliedCorrectCount++;
        }
        else
        {
            _biasNotAppliedCount++;
            if (resolvedCorrectness)
                _biasNotAppliedCorrectCount++;
        }

        if (telemetry.UsedPlatonicQuery)
        {
            if (resolvedCorrectness)
            {
                _platonicConfidenceCorrectCount++;
                _platonicConfidenceCorrectSum += Math.Clamp(telemetry.PlatonicConfidence, 0.0, 1.0);
            }
            else
            {
                _platonicConfidenceIncorrectCount++;
                _platonicConfidenceIncorrectSum += Math.Clamp(telemetry.PlatonicConfidence, 0.0, 1.0);
            }
        }

        if (telemetry.UsedNeuralFallback)
        {
            _fallbackCount++;
            if (resolvedCorrectness)
                _fallbackCorrectCount++;
        }

        // The only platonic-space WRITE in this method — gated to training. Inference must not mutate
        // the shared, persistent store.
        if (allowSpaceMutation)
            ObserveRetrospectiveEvidenceCredit(telemetry, resolvedCorrectness);

        _inferenceTelemetryHint = ComputeInferenceTelemetryHint();
        _inferencePolicy.ApplyTelemetryHint(_inferenceTelemetryHint);
    }

    private void EnqueuePendingInferenceIntrospection(GenesisExample example)
    {
        _pendingInferenceIntrospection.Enqueue(example);
        while (_pendingInferenceIntrospection.Count > MaxPendingInferenceIntrospection)
            _pendingInferenceIntrospection.Dequeue();
    }

    private bool? TryResolveCorrectness(GenesisExample example)
    {
        if (!TryExtractArithmeticObservation(example, out var arithmetic))
            return null;
        return arithmetic.AbsoluteError <= 0.05;
    }

    private static InferenceOutcome DeriveHeuristicOutcome(GenerationResult _)
        => InferenceOutcome.Unknown;

    private InferenceTelemetryHint ComputeInferenceTelemetryHint()
    {
        var biasAppliedRate = _biasAppliedCount > 0
            ? _biasAppliedCorrectCount / (double)_biasAppliedCount
            : 0.5;
        var biasNotAppliedRate = _biasNotAppliedCount > 0
            ? _biasNotAppliedCorrectCount / (double)_biasNotAppliedCount
            : 0.5;
        var fallbackRate = _telemetryObservationCount > 0
            ? _fallbackCount / (double)_telemetryObservationCount
            : 0.0;
        var fallbackSuccessRate = _fallbackCount > 0
            ? _fallbackCorrectCount / (double)_fallbackCount
            : 0.5;
        var confidenceDelta = 0.0;
        if (_platonicConfidenceCorrectCount > 0 && _platonicConfidenceIncorrectCount > 0)
        {
            var avgCorrect = _platonicConfidenceCorrectSum / _platonicConfidenceCorrectCount;
            var avgIncorrect = _platonicConfidenceIncorrectSum / _platonicConfidenceIncorrectCount;
            confidenceDelta = Math.Clamp(avgCorrect - avgIncorrect, -1.0, 1.0);
        }

        var rawBiasScale = 1.0 +
                           ((biasAppliedRate - biasNotAppliedRate) * 0.45) +
                           (confidenceDelta * 0.20) -
                           (fallbackRate * 0.20);
        var biasScale = Math.Clamp(rawBiasScale, 0.7, 1.4);
        var enableContextBias = biasAppliedRate + (confidenceDelta * 0.05) >= biasNotAppliedRate - 0.03 &&
                                fallbackSuccessRate >= 0.20;

        return new InferenceTelemetryHint(
            BiasScale: biasScale,
            EnableContextBias: enableContextBias);
    }

    private IReadOnlyList<string> ObserveLearningSignals(
        GenesisExample example,
        bool trackCoverage = true,
        bool allowTransformDiscovery = true)
    {
        var concepts = ResolveConcepts(example);
        ObservePlatonicSpace(example, concepts);
        if (allowTransformDiscovery)
            UpdateTransformDiscovery(example);

        if (trackCoverage)
        {
            foreach (var concept in concepts)
            {
                _conceptCoverageCounts.TryGetValue(concept, out var count);
                _conceptCoverageCounts[concept] = count + 1;
            }
        }

        return concepts;
    }

    // Evidence→contradiction credit: feeds the REAL inference channel (PlatonicSpaceMemory's
    // value-update rule). The journal-replay credit that rewarded the cut space/concept token-LM
    // policies has been removed; only the genuine space update remains.
    private void ObserveRetrospectiveEvidenceCredit(GenerationResult telemetry, bool success)
    {
        var evidence = telemetry.Evidence;
        if (evidence is null || evidence.Count == 0)
            return;

        _platonicSpace.ReinforceEvidence(evidence, success);
    }

    private void AddMasteredRehearsalExample(GenesisExample example)
    {
        if (string.IsNullOrWhiteSpace(example.Input) || string.IsNullOrWhiteSpace(example.Output))
            return;

        _masteredRehearsalRing.Enqueue(new MasteredRehearsalExample(example.Input, example.Output, ResolveRouteLabel(example)));
        while (_masteredRehearsalRing.Count > MaxMasteredRehearsalEntries)
            _masteredRehearsalRing.Dequeue();
    }

    private void MaybeInterleaveMasteredRehearsal()
    {
        if (_trainStepCount <= 0 || _trainStepCount % MasteredRehearsalInterval != 0 || _masteredRehearsalRing.Count == 0)
            return;

        var mastered = _masteredRehearsalRing.ElementAt(Random.Shared.Next(_masteredRehearsalRing.Count));
        var inputTokens = _tokenizer.Encode(mastered.Input);
        var targetTokens = _tokenizer.Encode(mastered.Output, addEos: true);
        if (targetTokens.Length == 0)
            return;

        _model.EnsureVocabularySize(_tokenizer.VocabularySize);
        _ = _model.TrainExample(
            inputTokens,
            targetTokens,
            _tokenizer.BosTokenId,
            lossScale: MasteredRehearsalLossScale,
            routeLabel: mastered.RouteLabel);
        _model.CloneParametersToBreakGraph();
        _masteredRehearsalCount++;
        _masteredInterleavedRehearsalCount++;
    }

    private void UpdateTransformDiscovery(GenesisExample example)
    {
        if (!TryExtractArithmeticObservation(example, out var arithmetic))
            return;

        if (!double.TryParse(arithmetic.LeftToken, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out var left) ||
            !double.TryParse(arithmetic.RightToken, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out var right) ||
            !double.TryParse(arithmetic.ResultToken, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out var output))
            return;

        _foldPathDiscovery.ObserveTrainingPair(arithmetic.OperationConcept, left, right, output);

        // After every 3 examples, attempt fold/log-linear discovery
        if (_trainStepCount % 3 == 0)
        {
            _foldPathDiscovery.TryRunDiscovery(
                arithmetic.OperationConcept,
                (op, a, b) => op switch
                {
                    "add" => (double?)(a + b),
                    "sub" => (double?)(a - b),
                    "mul" => (double?)(a * b),
                    "div" when b != 0 => (double?)(a / b),
                    _ => null
                });
        }

        var mode = _foldPathDiscovery.GetComposition(arithmetic.OperationConcept);
        var dim = _transformAccumulator.EmbeddingDimension;
        var inputEmbedding = InputEmbeddingComposer.ComposeInput(example.Input, mode, dim);
        var outputEmbedding = InputEmbeddingComposer.GetInputEmbedding(example.Output, dim);
        // EARNED reliability (bubble-up): BEFORE folding this example in, score whether the CURRENT transform
        // already predicts it better than identity — a held-out-ish predictive-success signal. Consistent
        // transforms keep improving; noisy ones don't. RecordOutcome feeds ReliabilityUcb → route perception so
        // the route head learns which transforms are actually useful. (First sighting has no transform → skip.)
        if (_transformAccumulator.TryGetTransform(arithmetic.OperationConcept, out _))
            _transformAccumulator.RecordOutcome(
                arithmetic.OperationConcept,
                _transformAccumulator.ApplyImprovesOverIdentity(arithmetic.OperationConcept, inputEmbedding, outputEmbedding));
        _transformAccumulator.Learn(arithmetic.OperationConcept, inputEmbedding, outputEmbedding);
    }

    // NOTE: the per-example tick loop (R1–R9 on a private `_tickState`) was removed 2026-06-14 — it was a
    // disconnected scratchpad: its elements/patterns were never written to the live PlatonicSpaceMemory nor
    // read by inference; its only effect was a telemetry counter. The one USEFUL tick capability (R2
    // Compose) is adapted directly by the glider blocks (PlatonicGlider via TickExecutor.ExecuteTick).

    // The space-PERCEPTION vector the edit head reads (SPACE_AWARE_GRU.md §A,§C): how the target currently sits
    // in the anchor's neighbourhood. A richer readout than rank alone — the learned perception weight (§C) reads
    // over: [rankNorm (0=nearest,1=far/absent), distractor-winning, nearest-is-target, target-closeness,
    // nearest-distractor-closeness, bias]. A poisoned anchor reads high rankNorm + distractor-winning + high
    // distractor-closeness → the head can learn to repel hard.
    private double[] ComputeEditPerception(IReadOnlyList<string> inputConcepts, IReadOnlyList<string> outputConcepts)
    {
        var anchor = inputConcepts.FirstOrDefault(c => !IsNumericConcept(c)) ?? inputConcepts[0];
        var targetSet = new HashSet<string>(outputConcepts.Select(c => c.ToLowerInvariant()), StringComparer.OrdinalIgnoreCase);
        // LIVE faces so the read-before-write perception conditioning the edit magnitude sees the space as it
        // IS this step (see GetNearestConceptsFresh); seed the targets so their fresh rank/closeness is exact.
        var near = _platonicSpace.GetNearestConceptsFresh(anchor, seeds: targetSet, maxNeighbors: 8);
        var rank = -1; var targetDist = double.NaN;
        for (var i = 0; i < near.Count; i++)
            if (targetSet.Contains(near[i].Symbol)) { rank = i; targetDist = near[i].Distance; break; }
        var nearestIsTarget = near.Count > 0 && targetSet.Contains(near[0].Symbol) ? 1.0 : 0.0;
        var distractorWinning = near.Count > 0 && !targetSet.Contains(near[0].Symbol) ? 1.0 : 0.0;
        var rankNorm = rank < 0 ? 1.0 : Math.Clamp(rank / 8.0, 0.0, 1.0);
        var targetCloseness = double.IsNaN(targetDist) ? 0.0 : 1.0 / (1.0 + Math.Max(0.0, targetDist));
        var nearestCloseness = near.Count > 0 ? 1.0 / (1.0 + Math.Max(0.0, near[0].Distance)) : 0.0;
        return new[] { rankNorm, distractorWinning, nearestIsTarget, targetCloseness, nearestCloseness, 1.0 };
    }

    private void ObservePlatonicSpace(GenesisExample example, IReadOnlyList<string> concepts)
    {
        if (concepts.Count == 0)
            return;

        // Snapshot how retrievable the target is BEFORE this step's writes, so RewardEditHead rewards the
        // DELTA those writes cause (a dense, immediately-causal signal for the space-BUILDING edit head).
        _pendingPreEditOutcome = ComputeEditOutcome(example);

        var hasArithmetic = TryExtractArithmeticObservation(example, out var arithmetic);
        if (hasArithmetic)
            ObserveArithmeticFaces(arithmetic);

        var isNegative = IsNegativeText(example.Output);

        // Learned edit strength: the NN's edit-head predicts HOW STRONGLY to edit the space for this
        // example, replacing the hand-coded 0.2/0.8. Direction comes from isNegative; the magnitude
        // m∈[0,1] scales how far the contradiction is pushed from neutral 0.5. The head is rewarded
        // after the step (RewardEditHead) by a CAUSAL, space-state-dependent outcome.
        //
        // Anti-collapse exploration: add small zero-mean noise to m BEFORE use/storage so the
        // do-nothing fixed point (m→0) is never a stable optimum. The noise is derived
        // deterministically from _trainStepCount (Random may be unavailable/seeded) and varies per
        // step; the explored magnitude m' is what we both APPLY and later REINFORCE.
        var editTokens = _tokenizer.Encode(example.Input);
        var rawM = Math.Clamp(_model.PredictEditMagnitude(editTokens), 0.0, 1.0);
        var explorationNoise = DeterministicEditExplorationNoise(_trainStepCount);
        var m = Math.Clamp(rawM + explorationNoise, 0.0, 1.0);
        _pendingEditTokens = editTokens;
        _pendingEditMagnitude = m;
        var contradiction = isNegative ? (0.5 + (0.5 * m)) : (0.5 - (0.5 * m));

        // Couple the GENUINE input→output tokens (deterministic mirror), NOT the NN concept-planner's
        // output. The planner hallucinates fringe tokens ("0", "0+") which, once coupled, contaminate the
        // relation graph so a number-word ends up spuriously related to noise. The directly-observed
        // input→output pair IS the genuine relationship and is what should dominate. (The planner still
        // runs once per example upstream for its concept set / telemetry — only the COUPLING source
        // changes here.)
        // Arithmetic is represented by the NUMERIC FACE HOMOMORPHISM (poly/log), NOT relation edges.
        // Canonical genesis creates NO operand↔result coupling for arithmetic (confirmed against the
        // source of truth: it uses the face homomorphism + function-application triplets, never direct
        // operand↔result edges). Such edges overload the operand concepts — digit "1" ends up related to
        // every result it ever appears in (1↔-1, 1↔3 …), drowning the genuine equivalence edge
        // ("1"↔"one") and ERASING prior lessons (measured: the dominant cause of number-word forgetting).
        // So arithmetic contributes ONLY its operation→face affinity (ObserveArithmeticFaces, above); it
        // does no relational coupling and no geometric edit.
        if (hasArithmetic)
            return;

        var inputConcepts = ExtractMirrorConcepts(example.Input, string.Empty);
        var outputConcepts = ExtractMirrorConcepts(example.Output, string.Empty);
        if (inputConcepts.Count == 0 || outputConcepts.Count == 0)
        {
            // Bounded fallback: couple only ADJACENT concepts (sequential structure), not the full
            // O(n²) pairwise mesh — keeps contamination linear in the concept count.
            for (var i = 0; i + 1 < concepts.Count; i++)
            {
                if (IsNumericConcept(concepts[i]) && IsNumericConcept(concepts[i + 1]))
                    continue; // number↔number lives in the faces, not the relation graph
                _platonicSpace.ObserveContradiction(concepts[i], concepts[i + 1], contradiction);
            }
            _platonicSpace.FineEditFromExample(concepts, concepts, isNegative);
            return;
        }

        // LEARNED space-awareness: recompute the edit magnitude CONDITIONED on a perceived space-state vector
        // (rank-of-target / distractor-winning / nearest-is-target), so the head learns a read-before-write
        // policy; that magnitude then scales the distractor repulsion below.
        if (PerceptionEdit)
        {
            _pendingEditPerception = ComputeEditPerception(inputConcepts, outputConcepts);
            var pm = Math.Clamp(_model.PredictEditMagnitude(editTokens, _pendingEditPerception), 0.0, 1.0);
            m = Math.Clamp(pm + DeterministicEditExplorationNoise(_trainStepCount + 1), 0.0, 1.0);
            _pendingEditMagnitude = m;
            contradiction = isNegative ? (0.5 + (0.5 * m)) : (0.5 - (0.5 * m));
        }
        else
        {
            _pendingEditPerception = null;
        }

        // Academic coupling: link each output concept to its SINGLE nearest input concept (the
        // genuinely-related pair, found via the lattice), NOT the full input×output mesh — and drop
        // output↔output coupling entirely (co-occurrence is not a relationship). This is the
        // minimum-contamination signal that still teaches the input→output association.
        // A NUMERIC output from a MULTI-concept input is a COMPUTATION (e.g. framed arithmetic
        // "what is 1 plus 1" → "2" that slipped past the compact-only arithmetic gate). Numbers relate
        // via the face homomorphism, NOT the relation graph — so such an example writes NO relation edge.
        // Otherwise its framing words ("what","the","sum") AND operands get coupled to the result number,
        // and those spurious word↔number edges make the GRU treat "what" as an operand (measured: framed
        // arithmetic collapsed to 0/5). A single-token input ("one"→"1") IS a genuine number-word
        // equivalence and is still coupled. (PLATONIC_SPACE.md §6.2: arithmetic writes no relation edges.)
        // SPACE-AWARE READ-BEFORE-WRITE: read each anchor's CURRENT nearest neighbour; if a non-target concept
        // is winning (a distractor or a prior bad write), repel it (high contradiction = push apart) so the
        // target can take the nearest slot. This is what lets the controller UNDO a bad write — the blind
        // attract-only path below cannot (it never looks at what's already there). Positive examples only.
        if ((SpaceAwareEdit || PerceptionEdit) && !isNegative)
        {
            // Repel strength: hand-coded (SpaceAwareEdit oracle) vs LEARNED from the perception-conditioned
            // magnitude m (PerceptionEdit) — the GRU learns, via reward, how hard to push the distractor away.
            var repel = PerceptionEdit ? Math.Clamp(0.5 + (0.5 * m), 0.5, 1.0) : 0.9;
            var targetSet = new HashSet<string>(outputConcepts.Select(c => c.ToLowerInvariant()), StringComparer.OrdinalIgnoreCase);
            // LEARNED POINTER COUNT (§D): when perception-driven, the GRU's magnitude m decides HOW MANY of the
            // nearest distractors to clear (1..RepelNeighbors) — not a fixed count. Hand-coded mode uses the cap.
            var k = PerceptionEdit
                ? Math.Clamp(1 + (int)Math.Round(m * (RepelNeighbors - 1)), 1, Math.Max(1, RepelNeighbors))
                : Math.Max(1, RepelNeighbors);
            foreach (var anchor in inputConcepts)
            {
                if (IsNumericConcept(anchor)) continue;
                // POINTER over the neighbourhood: repel the top-k nearest NON-target concepts (distractors), not
                // just the single nearest — clears multiple competitors so the target can take the nearest slot.
                var near = _platonicSpace.GetNearestConcepts(anchor, candidates: null, maxNeighbors: k);
                foreach (var n in near)
                {
                    if (targetSet.Contains(n.Symbol) || IsNumericConcept(n.Symbol)
                        || n.Symbol.Equals(anchor, StringComparison.OrdinalIgnoreCase)) continue;
                    _platonicSpace.ObserveContradiction(anchor, n.Symbol, repel); // repel a winning distractor
                }
            }
        }

        var outputComputed = inputConcepts.Count > 1;
        foreach (var output in outputConcepts)
        {
            var partners = inputConcepts
                .Where(inp => !inp.Equals(output, StringComparison.OrdinalIgnoreCase))
                .ToArray();
            if (partners.Length == 0)
                continue;
            var nearest = _platonicSpace.GetNearestConcepts(output, candidates: partners, maxNeighbors: 1);
            var partner = nearest.Count > 0 ? nearest[0].Symbol : partners[0];
            // Skip ANY edge to a numeric result of a computation (multi-concept input), and the
            // number↔number case generally — both are carried by the homomorphism, not the graph.
            if (IsNumericConcept(output) && (outputComputed || IsNumericConcept(partner)))
                continue;
            _platonicSpace.ObserveContradiction(partner, output, contradiction);
        }

        _platonicSpace.FineEditFromExample(inputConcepts, outputConcepts, isNegative);
    }

    // Numbers relate geometrically via the face homomorphism, never as relation-graph edges — so the
    // input→output coupling must never mint a number↔number relation (see ObservePlatonicSpace).
    private static bool IsNumericConcept(string concept)
        => double.TryParse(
            concept,
            System.Globalization.NumberStyles.Float | System.Globalization.NumberStyles.AllowLeadingSign,
            System.Globalization.CultureInfo.InvariantCulture,
            out _);

    // CAUSAL REINFORCE reward for the edit-head. The token forward pass never reads the space, so a
    // token-loss delta was causally disconnected from the edit. Instead we measure a
    // SPACE-STATE-DEPENDENT outcome in [0,1]: AFTER the edit was applied (in ObservePlatonicSpace,
    // which ran just before this), does the platonic space now RETRIEVE the correct output for this
    // input? Retrieval is the universal op, so this generalizes across modalities. The reward is
    // outcome - the last-seen outcome for this example (REINFORCE baseline); first sighting -> 0.
    // This nudges PredictEditMagnitude toward the write strength that genuinely improves retrieval,
    // and the exploration noise added to m (in ObservePlatonicSpace) prevents the m->0 fixed point.
    // tokenLoss is retained in the signature (callers pass it) but no longer drives the reward.
    // PERCEPTION ROUTING (SPACE_AWARE_GRU.md §I): teach the route head to route platonic from PERCEIVED
    // retrievability. We reinforce the route-perception weight TOWARD the resolved route label (which the space's
    // retrievability determines) given the TARGET-AGNOSTIC route perception of the anchor — so at inference, a
    // query whose anchor "looks answerable" gets routed platonic. No-op unless the model has PerceptionRouting on.
    private void MaybeReinforceRoute(GenesisExample example, IReadOnlyList<int> inputTokens)
    {
        if (!_model.PerceptionRouting)
            return;
        if (ResolveRouteLabel(example) is not int routeLabel)
            return;
        var inputConcepts = ExtractMirrorConcepts(example.Input, string.Empty);
        if (inputConcepts.Count == 0)
            return;
        var anchor = inputConcepts.FirstOrDefault(c => !IsNumericConcept(c)) ?? inputConcepts[0];
        // Bubble the EARNED transform reliability into the route perception so the route head is reinforced to
        // trust the function/platonic route in proportion to how proven the model's transforms are.
        var transformReliability = _model.TransformReliabilityRouting ? _transformAccumulator.BestReliabilityUcb() : 0.0;
        _model.ReinforceRouteHead(inputTokens, _platonicSpace.ComputeRoutePerception(anchor, transformReliability), routeLabel, 1.0);
    }

    private void RewardEditHead(GenesisExample example, double tokenLoss)
    {
        _ = tokenLoss; // no longer causally relevant to the edit-head reward
        var pending = _pendingEditTokens;
        _pendingEditTokens = null;
        if (pending is null || pending.Count == 0)
            return;

        // WITHIN-STEP CAUSAL REWARD: how much THIS step's writes improved the (contrastive, bidirectional)
        // retrievability of the target (post vs the pre snapshot). + helped, - hurt; order-invariant, so the
        // answer-order shuffling no longer zeroes it and multi-answer sets finally get a building signal.
        var post = ComputeEditOutcome(example);
        var reward = post - _pendingPreEditOutcome;
        _model.ReinforceEditHead(pending, _pendingEditPerception, _pendingEditMagnitude, reward);
        _pendingEditPerception = null;
    }

    // Space-state-dependent outcome in [0,1]: how COHESIVELY the space holds the input↔output association.
    // CONTRASTIVE — the target must out-rank ALL other concepts (candidates: null → distractors compete), not
    // merely be close in isolation, so a write that SEPARATES the pair from confusers scores higher (attacks
    // the "disjointed space"). BIDIRECTIONAL — scored both cue→answer and answer→cue and averaged, so the
    // association is a mutual neighbour (a real lattice edge), not a one-way coincidence. Arithmetic is credited
    // via the face path. Used both to reward the edit head (pre/post delta) and to warm the route label.
    private double ComputeEditOutcome(GenesisExample example)
    {
        // Arithmetic: credit numerical correctness directly (already mirrored into the space's faces
        // by ObserveArithmeticFaces during the edit).
        if (TryExtractArithmeticObservation(example, out var arithmetic))
            return arithmetic.AbsoluteError <= 1e-6 ? 1.0 : 0.0;

        var input = example.Input ?? string.Empty;
        var output = example.Output ?? string.Empty;
        if (string.IsNullOrWhiteSpace(input) || string.IsNullOrWhiteSpace(output))
            return 0.0;

        var inputConcepts = ExtractMirrorConcepts(input, string.Empty);
        var outputConcepts = ExtractMirrorConcepts(output, string.Empty);
        if (inputConcepts.Count == 0 || outputConcepts.Count == 0)
            return 0.0;

        var inputSet = new HashSet<string>(inputConcepts.Select(c => c.ToLowerInvariant()), StringComparer.OrdinalIgnoreCase);
        var outputSet = new HashSet<string>(outputConcepts.Select(c => c.ToLowerInvariant()), StringComparer.OrdinalIgnoreCase);

        var forward = DirectedRetrievalScore(inputConcepts[0], outputSet);   // cue → answer
        var backward = DirectedRetrievalScore(outputConcepts[0], inputSet);  // answer → cue
        return Clamp01((0.5 * forward) + (0.5 * backward));
    }

    // Contrastive directed retrieval: how high the target ranks among ALL of `from`'s nearest neighbours
    // (distractors included). Nearest-among-everything → ~1; buried under confusers / absent → ~0.
    private double DirectedRetrievalScore(string from, HashSet<string> targetSymbols)
    {
        const int maxNeighbors = 8;
        // LIVE-FACE read (see GetNearestConceptsFresh): the pre/post retrievability delta must reflect THIS
        // step's writes, so the edit head is rewarded by the edit's real effect, not a stale tree snapshot.
        // Seed with the targets so they are always scored fresh; relational + current neighbours supply distractors.
        var nearest = _platonicSpace.GetNearestConceptsFresh(from, seeds: targetSymbols, maxNeighbors: maxNeighbors);
        for (var rank = 0; rank < nearest.Count; rank++)
        {
            if (!targetSymbols.Contains(nearest[rank].Symbol))
                continue;
            var rankScore = 1.0 / (1.0 + rank);
            var distanceScore = 1.0 / (1.0 + Math.Max(0.0, nearest[rank].Distance));
            return Clamp01((0.5 * rankScore) + (0.5 * distanceScore));
        }
        return 0.0;
    }

    // Deterministic zero-mean exploration noise in ~+/-0.05, derived from the step index so it varies
    // per step without relying on a (possibly seeded/unavailable) Random. A simple integer hash gives
    // a well-mixed value in [0,1), recentred to [-0.05, 0.05).
    private static double DeterministicEditExplorationNoise(int step)
    {
        unchecked
        {
            var h = (uint)step * 2654435761u; // Knuth multiplicative hash
            h ^= h >> 15;
            h *= 2246822519u;
            h ^= h >> 13;
            var unit = (h & 0xFFFFFF) / (double)0x1000000; // [0,1)
            return (unit - 0.5) * 0.10;                    // [-0.05, 0.05)
        }
    }

    private void ObserveArithmeticFaces(ArithmeticObservation arithmetic)
    {
        if (arithmetic.Operation is ArithmeticOperation.Add or ArithmeticOperation.Subtract)
        {
            _platonicSpace.ObserveContradiction(arithmetic.OperationConcept, "face:poly", 0.05);
            _platonicSpace.ObserveContradiction(arithmetic.OperationConcept, "face:log", 0.85);
        }
        else if (arithmetic.Operation is ArithmeticOperation.Multiply or ArithmeticOperation.Divide)
        {
            _platonicSpace.ObserveContradiction(arithmetic.OperationConcept, "face:poly", 0.85);
            _platonicSpace.ObserveContradiction(arithmetic.OperationConcept, "face:log", 0.05);
        }

        // NOTE: arithmetic deliberately creates NO operand↔result relation edges. The face homomorphism
        // above IS the platonic representation of the computation; canonical genesis stores no such edges.
        // Coupling operands to results (1↔-1, 3↔4 …) overloaded the operand concepts and erased the
        // number-word equivalence — removed 2026-06-14. Only the operation→face affinity is observed.
    }

    private static bool TryExtractArithmeticObservation(GenesisExample example, out ArithmeticObservation observation)
    {
        observation = default;
        var input = example.Input ?? string.Empty;
        var output = example.Output ?? string.Empty;

        // Only handle compact symbol-based expressions (same pattern as inference).
        // Natural language forms are not recognized here — they route through the ML layer.
        var compact = System.Text.RegularExpressions.Regex.Match(
            input.ToLowerInvariant(),
            @"^\s*(-?\d+(?:\.\d+)?)\s*([+\-*/x])\s*(-?\d+(?:\.\d+)?)\s*$");
        if (!compact.Success)
            return false;

        if (!double.TryParse(compact.Groups[1].Value, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out var left) ||
            !double.TryParse(compact.Groups[3].Value, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out var right))
            return false;

        var outputMatch = System.Text.RegularExpressions.Regex.Match(output, @"-?\d+(\.\d+)?");
        if (!outputMatch.Success ||
            !double.TryParse(outputMatch.Value, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out var actual))
            return false;

        ArithmeticOperation operation;
        string operationConcept;
        double expected;

        switch (compact.Groups[2].Value)
        {
            case "+":
                operation = ArithmeticOperation.Add;
                operationConcept = "add";
                expected = left + right;
                break;
            case "-":
                operation = ArithmeticOperation.Subtract;
                operationConcept = "sub";
                expected = left - right;
                break;
            case "*":
            case "x":
                operation = ArithmeticOperation.Multiply;
                operationConcept = "mul";
                expected = left * right;
                break;
            case "/":
                if (Math.Abs(right) <= 1e-12)
                    return false;
                operation = ArithmeticOperation.Divide;
                operationConcept = "div";
                expected = left / right;
                break;
            default:
                return false;
        }

        observation = new ArithmeticObservation(
            Operation: operation,
            OperationConcept: operationConcept,
            LeftToken: compact.Groups[1].Value,
            RightToken: compact.Groups[3].Value,
            ResultToken: outputMatch.Value,
            AbsoluteError: Math.Abs(expected - actual));
        return true;
    }

    private double EstimatePlatonicConsistency(IReadOnlyList<string> concepts, bool negativeExample)
    {
        if (concepts.Count < 2)
            return 0.0;

        var target = negativeExample ? 0.75 : 0.25;
        var avg = EstimateAverageContradiction(concepts);
        return Clamp01(Math.Abs(avg - target));
    }

    private double EstimateAverageContradiction(IReadOnlyList<string> concepts)
    {
        if (concepts.Count < 2)
            return 0.5;

        var total = 0.0;
        var pairs = 0;
        for (var i = 0; i < concepts.Count; i++)
        {
            for (var j = i + 1; j < concepts.Count; j++)
            {
                total += _platonicSpace.GetContradiction(concepts[i], concepts[j]);
                pairs++;
            }
        }
        return pairs == 0 ? 0.5 : total / pairs;
    }

    private double EstimateMemoryLoad()
    {
        var scale = Math.Max(1, _platonicSpace.NodeCount + _platonicSpace.RelationCount);
        return Clamp01(scale / 50_000.0);
    }

    private static string NormalizeConceptToken(string raw)
    {
        var cleaned = new string(raw.Where(ch =>
                char.IsLetterOrDigit(ch) || ch is '+' or '-' or '*' or '/' or '=' or '.')
            .ToArray());
        return cleaned.Trim();
    }

     private double ComputeQualityLoss()
     {
         // Rate-limit full quality recomputation. This avoids per-example full relation scans,
         // which otherwise grow more expensive over long autonomous runs.
         if (_cachedQualityLoss.HasValue &&
             _lastQualityComputationStep != int.MinValue &&
             (_trainStepCount - _lastQualityComputationStep) < QualityLossRefreshInterval)
         {
             return _cachedQualityLoss.Value;
         }

         var allRelations = _platonicSpace.GetAllRelations();
         if (allRelations.Count == 0)
         {
             _cachedQualityLoss = 0.0;
             _lastRelationCountForQuality = 0;
             _lastQualityComputationStep = _trainStepCount;
             return 0.0;
         }

         // Compute average quality penalty across all operations in the space
         var qualitySum = 0.0;
         var sampledTargetsByOperation = new Dictionary<string, (string Target, long ObservationCount)>(StringComparer.OrdinalIgnoreCase);
         
         foreach (var (left, right, obsCount) in allRelations)
         {
             if (!sampledTargetsByOperation.ContainsKey(left))
                 sampledTargetsByOperation[left] = (right, obsCount);
         }

         var stride = Math.Max(1, sampledTargetsByOperation.Count / MaxQualityOperationsPerCompute);
         var phase = _trainStepCount % stride;
         var opIndex = 0;
         var computedCount = 0;

         foreach (var (operation, targetRel) in sampledTargetsByOperation)
         {
             if (stride > 1 && (opIndex % stride) != phase)
             {
                 opIndex++;
                 continue;
             }
             opIndex++;

             var qualityPenalty = TransformQualityMetrics.ComputeQualityLossPenalty(
                 operation: operation,
                 target: targetRel.Target,
                 relationObservationCount: targetRel.ObservationCount,
                 allRelations: allRelations,
                 utilityThreshold: 0.3);
             qualitySum += qualityPenalty;
             computedCount++;
         }

         // Cache result and return
         var result = computedCount > 0 ? qualitySum / computedCount : 0.0;
         _lastRelationCountForQuality = allRelations.Count;
         _cachedQualityLoss = result;
         _lastQualityComputationStep = _trainStepCount;
         return result;
     }

     private static double Clamp01(double value)
         => Math.Max(0.0, Math.Min(1.0, value));

     public SpaceManagementResult ManagePlatonicSpace()
         => _spaceManager.Manage();

     public (int Nodes, int Relations) GetPlatonicSpaceSize()
         => (_platonicSpace.NodeCount, _platonicSpace.RelationCount);

     private readonly record struct SpacePolicyTransition(
         SpacePolicyState State,
         int ActionId,
         double NoiseRatio,
         double AverageBridgeConfidence,
         double RelationPressure,
         int StepIndex);

     private enum InferenceOutcome
     {
         Unknown = 0,
         Success = 1,
         Failure = 2
     }

     private readonly record struct ConceptPlanTelemetry(
         string Prompt,
         string ModelOutput,
         IReadOnlyList<string> SelectedConcepts,
         ConceptPlanSelectionMode Mode,
         double MirrorCoverage,
         double NoveltyRatio);

     private enum ConceptPlanSelectionMode
     {
         MirrorFallback = 0,
         PlannedDirect = 1,
         Merged = 2
     }

     private readonly record struct SpacePolicyState(
         int Nodes,
         int Relations,
         double Noise,
         double Bridge,
         double RelationPressure,
         int PriorActionId,
         string ConceptFeatures)
     {
         public string ToDeterministicEncoding()
             => FormattableString.Invariant(
                 $"sp|n={Nodes}|r={Relations}|z={Noise:F4}|b={Bridge:F4}|p={RelationPressure:F4}|pa={PriorActionId}|cf={ConceptFeatures}");

         public static bool TryFromDeterministicEncoding(string value, out SpacePolicyState state)
         {
             state = default;
             if (string.IsNullOrWhiteSpace(value))
                 return false;

             var segments = value.Split('|', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
             if (segments.Length < 7 || !segments[0].Equals("sp", StringComparison.OrdinalIgnoreCase))
                 return false;

             var dict = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
             foreach (var segment in segments.Skip(1))
             {
                 var index = segment.IndexOf('=');
                 if (index <= 0 || index >= segment.Length - 1)
                     continue;
                 dict[segment[..index]] = segment[(index + 1)..];
             }

             if (!dict.TryGetValue("n", out var nodesRaw) ||
                 !dict.TryGetValue("r", out var relationsRaw) ||
                 !dict.TryGetValue("z", out var noiseRaw) ||
                 !dict.TryGetValue("b", out var bridgeRaw) ||
                 !dict.TryGetValue("p", out var pressureRaw) ||
                 !dict.TryGetValue("pa", out var priorActionRaw))
             {
                 return false;
             }

             if (!int.TryParse(nodesRaw, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var nodes) ||
                 !int.TryParse(relationsRaw, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var relations) ||
                 !double.TryParse(noiseRaw, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var noise) ||
                 !double.TryParse(bridgeRaw, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var bridge) ||
                 !double.TryParse(pressureRaw, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var relationPressure) ||
                 !int.TryParse(priorActionRaw, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var priorActionId))
             {
                 return false;
             }

             if (!dict.TryGetValue("cf", out var conceptFeatures))
                 dict.TryGetValue("ca", out conceptFeatures);
             state = new SpacePolicyState(
                 Nodes: Math.Max(0, nodes),
                 Relations: Math.Max(0, relations),
                 Noise: Math.Clamp(noise, 0.0, 2.0),
                 Bridge: Math.Clamp(bridge, 0.0, 1.0),
                 RelationPressure: Math.Clamp(relationPressure, 0.0, 2.0),
                 PriorActionId: Math.Max(0, priorActionId),
                 ConceptFeatures: conceptFeatures ?? string.Empty);
             return true;
         }
     }

     private readonly record struct ArithmeticObservation(
         ArithmeticOperation Operation,
         string OperationConcept,
         string LeftToken,
         string RightToken,
         string ResultToken,
         double AbsoluteError);

     private enum ArithmeticOperation
     {
         Add,
         Subtract,
         Multiply,
         Divide
     }
}

public sealed record GenesisStepLoss(
    double TokenLoss,
    double RouteLoss,
    double ConsistencyLoss,
    double ConservationLoss,
    double MemoryLoss,
    double TotalLoss);
