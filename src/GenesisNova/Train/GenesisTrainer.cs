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
    private readonly int _trainingTickMultiplier;
    private GenesisInferenceEngine _inferencePolicy;
    private readonly GenesisCompositeObjective _objective;
    private readonly FoldPathDiscovery _foldPathDiscovery;
    private readonly TransformAccumulator _transformAccumulator;
    private PlatonicState _tickState;
    private readonly Queue<SpacePolicyTransition> _spacePolicyTrajectory = new();
    private int _spacePolicyStepCounter;
    private int _priorSpaceActionId;
    private int _tickPatternPromotions;
    private int _trainStepCount;
    // Model-driven edit-head wiring: ObservePlatonicSpace stashes the (explored) magnitude it
    // requested from _model.PredictEditMagnitude (plus the tokens it conditioned on); the surrounding
    // train step then rewards the head via _model.ReinforceEditHead with a CAUSAL, space-state-
    // dependent outcome (does the platonic space now retrieve the correct output for this input?).
    private IReadOnlyList<int>? _pendingEditTokens;
    private double _pendingEditMagnitude;
    private double _cachedConservationLoss;
    // Per-example baseline of the EDIT-HEAD OUTCOME metric (space-retrieval quality, not token loss),
    // used as the REINFORCE baseline so reward = outcome − last-seen-outcome (first sighting → 0).
    private readonly Dictionary<string, double> _exampleOutcomeBaseline = new(StringComparer.Ordinal);
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
    // Per-component glider-block route labeling: lets ResolveRouteLabel recognise the compact block
    // capabilities (compare/larger/scale) and supervise them as platonic-direct. See PROJECT_GLIDER.md §6.
    private readonly PlatonicGliderInterpreter _gliderInterpreter;

    public GenesisTrainer(
        IGenesisTokenizer tokenizer,
        GenesisNeuralModel model,
        PlatonicSpaceMemory platonicSpace,
        GenesisNovaConfig? config = null,
        GenesisCompositeObjective? objective = null)
    {
        _tokenizer = tokenizer;
        _model = model;
        _platonicSpace = platonicSpace;
        _gliderInterpreter = new PlatonicGliderInterpreter(platonicSpace);
        var runtimeConfig = config ?? new GenesisNovaConfig();
        _foldPathDiscovery = new FoldPathDiscovery();
        _transformAccumulator = new TransformAccumulator(Math.Max(4, _platonicSpace.FaceDimension));
        _tickState = new PlatonicState(ImmutableArray<PlatonicElement>.Empty, _transformAccumulator.EmbeddingDimension);
        _spaceManager = new SpaceManager(_platonicSpace, new SpaceManagerSettings(
            Enabled: runtimeConfig.AutoManagePlatonicSpace,
            MaxNodes: Math.Max(256, runtimeConfig.MaxPlatonicNodes),
            MaxRelations: Math.Max(1_024, runtimeConfig.MaxPlatonicRelations)));
        _trainingTickMultiplier = Math.Clamp(runtimeConfig.TrainingTickMultiplier, 1, 32);
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
            foldPathDiscovery: _foldPathDiscovery,
            transformAccumulator: _transformAccumulator);
    }

    public int[] EncodeInput(string input)
        => _tokenizer.Encode(input);

    public int[] EncodeTarget(string output)
        => _tokenizer.Encode(output, addEos: true);

    public int EosTokenId => _tokenizer.EosTokenId;

    public FoldPathDiscovery FoldPathDiscovery => _foldPathDiscovery;
    public TransformAccumulator TransformAccumulator => _transformAccumulator;
    public int TickPatternPromotions => _tickPatternPromotions;
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
    /// PROBE/design toggle: when true, face-computable arithmetic examples do NOT create relational
    /// operand↔result edges — the face homomorphism already represents the computation. Stops an
    /// operand like "3" from accumulating dozens of arithmetic co-occurrence relations (3↔-1, 3↔4 …)
    /// that drown genuine equivalence/retrieval relations (3↔three). Default false = legacy coupling.
    /// </summary>
    public bool SuppressArithmeticOperandRelations { get; set; }
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
            queryLabel: ResolveQueryLabel(inputTokens, example.Output));
        
        // Break computation graph after training
        _model.CloneParametersToBreakGraph();

        var concepts = ObserveLearningSignals(example);
        RewardEditHead(example, baseLoss.TokenLoss);

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
            queryLabel: ResolveQueryLabel(inputTokens, example.Output));
         
        // Break computation graph after each example to allow sequential training
        _model.CloneParametersToBreakGraph();

        var concepts = ObserveLearningSignals(example);
        RewardEditHead(example, baseLoss.TokenLoss);

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
            queryLabel: ResolveQueryLabel(inputTokens, example.Output));
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
            queryLabel: ResolveQueryLabel(inputTokens, example.Output));
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
          RewardEditHead(example, loss.TokenLoss);
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

    private static IReadOnlyList<string> ExtractMirrorConcepts(string input, string output)
    {
        var words = $"{input} {output}"
            .ToLowerInvariant()
            .Split(' ', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
            .Select(NormalizeConceptToken)
            .Where(static w => w.Length > 0)
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

        // Per-component glider-block capabilities (compare/larger/scale): answered by running a hand-built
        // block composition on the substrate. Exact and space-independent (face arithmetic), so this is a
        // clean DIRECT (route 1) label from step 0 whenever it reproduces the target — which teaches the
        // router to send the capability to platonic-direct. See PROJECT_GLIDER.md §6.
        if (_gliderInterpreter.TryResolveCapability(example.Input ?? string.Empty, out var blockAnswer, out _)
            && string.Equals(blockAnswer.Trim(), (example.Output ?? string.Empty).Trim(), StringComparison.OrdinalIgnoreCase))
        {
            return 1;
        }

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
            if (chainCoverage >= 0.5)
                return 2; // partial relational reconstruction → assisted.
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
    /// Derives supervision for the GRU's platonic-query construction heads from the example's OWN
    /// numeric structure — no surface grammar. The tokenizer emits each digit as its own token, so
    /// operands are MAXIMAL DIGIT RUNS in the input token sequence. With exactly two runs (L, R) and
    /// a numeric output O, the operation is whichever single face op satisfies op(L, R) == O
    /// (add/sub/mul/div). Framing tokens ("what", "is", "plus", "?") are operand-head negatives —
    /// which is precisely how the GRU LEARNS to ignore irrelevant text. Ambiguous (multiple ops
    /// match) or non-conforming examples return null: unsupervised, costs nothing.
    /// </summary>
    internal GenesisQueryLabel? ResolveQueryLabel(IReadOnlyList<int> inputTokens, string? output)
    {
        if (inputTokens.Count == 0 || string.IsNullOrWhiteSpace(output))
            return null;
        if (!double.TryParse(output.Trim(), System.Globalization.NumberStyles.AllowLeadingSign
                | System.Globalization.NumberStyles.AllowDecimalPoint,
                System.Globalization.CultureInfo.InvariantCulture, out var target))
            return null;

        var vocab = _tokenizer.Vocabulary;
        bool IsDigitToken(int id) =>
            id >= 0 && id < vocab.Count && vocab[id].Length == 1 && vocab[id][0] is >= '0' and <= '9';
        bool IsMinusToken(int id) => id >= 0 && id < vocab.Count && vocab[id] == "-";

        // Maximal SIGNED digit runs → operand values + their token index ranges. A '-' immediately
        // before a digit run is UNARY (part of the operand) when it is not itself preceded by a digit
        // — so "5 + -3" yields operands (5, −3) and stays an ADD, keeping the surface operator and
        // the supervised face op consistent. (Without this, generator examples with negative second
        // operands get value-relabelled as the opposite op and the op head learns surface noise.)
        var runs = new List<(int Start, int End, double Value)>();
        var i = 0;
        while (i < inputTokens.Count)
        {
            var negative = false;
            var start = i;
            if (IsMinusToken(inputTokens[i])
                && i + 1 < inputTokens.Count && IsDigitToken(inputTokens[i + 1])
                && (i == 0 || !IsDigitToken(inputTokens[i - 1])))
            {
                negative = true;
                i++;
            }
            if (i >= inputTokens.Count || !IsDigitToken(inputTokens[i])) { i = start + 1; continue; }
            var value = 0.0;
            while (i < inputTokens.Count && IsDigitToken(inputTokens[i]))
            {
                value = (value * 10.0) + (vocab[inputTokens[i]][0] - '0');
                i++;
            }
            runs.Add((start, i, negative ? -value : value));
        }
        if (runs.Count != 2)
            return null;

        var (l, r) = (runs[0].Value, runs[1].Value);
        var matches = new List<int>();
        if (Math.Abs((l + r) - target) < 1e-9) matches.Add(1);
        if (Math.Abs((l - r) - target) < 1e-9) matches.Add(2);
        if (Math.Abs((l * r) - target) < 1e-9) matches.Add(3);
        if (Math.Abs(r) > 1e-12 && Math.Abs((l / r) - target) < 1e-9) matches.Add(4);
        if (matches.Count != 1)
            return null; // ambiguous (e.g. 2+2 == 2*2) or no face op fits — leave unsupervised.

        var mask = new bool[inputTokens.Count];
        foreach (var run in runs)
            for (var t = run.Start; t < run.End; t++)
                mask[t] = true;

        return new GenesisQueryLabel(matches[0], mask);
    }

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
        // Academic cadence: a single principled pass per example (conforms to the source's
        // once-per-tick frontier update), not a 1–24x storm that amplifies noise and contamination.
        RunTickPatternLoop(example, concepts);

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
        _transformAccumulator.Learn(arithmetic.OperationConcept, inputEmbedding, outputEmbedding);
    }

    private void RunTickPatternLoop(GenesisExample example, IReadOnlyList<string> concepts)
    {
        if (concepts.Count == 0)
            return;

        var dimension = _tickState.EmbeddingDimension;
        _tickState = _tickState with { CurrentTick = _tickState.CurrentTick + 1 };

        var primarySymbol = concepts[0];
        var primaryId = EnsureTickElement(primarySymbol, ElementKind.Function);
        var secondaryIds = concepts.Skip(1).Take(4)
            .Select(c => EnsureTickElement(c, ElementKind.Object))
            .ToArray();
        var actionBudget = Math.Clamp(_trainingTickMultiplier + (concepts.Count / 2), 2, 8);
        var actionsExecuted = 0;

        void ExecuteIfPossible(TickAction action, string promoteKey)
        {
            if (actionsExecuted >= actionBudget)
                return;

            var (updatedState, generated) = TickExecutor.ExecuteTick(action, _tickState);
            _tickState = updatedState;
            actionsExecuted++;

            if (generated.Length > 0)
                PromoteDetectedPatterns(promoteKey, generated);
        }

        if (TryExtractArithmeticObservation(example, out var arithmetic))
        {
            var mode = _foldPathDiscovery.GetComposition(arithmetic.OperationConcept);
            var inputEmbedding = InputEmbeddingComposer.ComposeInput(example.Input, mode, dimension);
            var outputEmbedding = InputEmbeddingComposer.GetInputEmbedding(example.Output, dimension);
            var delta = new double[dimension];
            for (var i = 0; i < dimension; i++)
                delta[i] = outputEmbedding[i] - inputEmbedding[i];

            ExecuteIfPossible(
                new TickAction(
                    Kind: TickKind.LocalLearn,
                    PrimaryElementId: primaryId,
                    Parameter: arithmetic.OperationConcept,
                    AuxiliaryEmbedding: delta),
                arithmetic.OperationConcept);

            var leftId = EnsureTickElement(arithmetic.LeftToken, ElementKind.Object);
            var rightId = EnsureTickElement(arithmetic.RightToken, ElementKind.Object);
            var resultId = EnsureTickElement(arithmetic.ResultToken, ElementKind.Object);
            ExecuteIfPossible(
                new TickAction(
                    Kind: TickKind.FoldChain,
                    PrimaryElementId: primaryId,
                    SecondaryIds: [leftId, rightId, resultId],
                    Parameter: arithmetic.OperationConcept),
                arithmetic.OperationConcept);
            ExecuteIfPossible(
                new TickAction(
                    Kind: TickKind.Gap,
                    PrimaryElementId: primaryId,
                    SecondaryIds: [leftId, resultId],
                    Parameter: arithmetic.OperationConcept),
                arithmetic.OperationConcept);
            ExecuteIfPossible(
                new TickAction(
                    Kind: TickKind.Analogy,
                    PrimaryElementId: resultId,
                    SecondaryIds: [leftId, rightId],
                    Parameter: arithmetic.OperationConcept),
                arithmetic.OperationConcept);
            ExecuteIfPossible(
                new TickAction(
                    Kind: TickKind.Relate,
                    PrimaryElementId: resultId,
                    Parameter: arithmetic.OperationConcept),
                arithmetic.OperationConcept);
            return;
        }

        if (secondaryIds.Length >= 1)
        {
            ExecuteIfPossible(
                new TickAction(
                    Kind: TickKind.Relate,
                    PrimaryElementId: secondaryIds[0]),
                primarySymbol);
            ExecuteIfPossible(
                new TickAction(
                    Kind: TickKind.FoldChain,
                    PrimaryElementId: primaryId,
                    SecondaryIds: secondaryIds.Take(3).ToArray()),
                primarySymbol);
        }

        if (secondaryIds.Length >= 2)
        {
            ExecuteIfPossible(
                new TickAction(
                    Kind: TickKind.Gap,
                    PrimaryElementId: primaryId,
                    SecondaryIds: [secondaryIds[0], secondaryIds[1]]),
                primarySymbol);
            ExecuteIfPossible(
                new TickAction(
                    Kind: TickKind.Analogy,
                    PrimaryElementId: secondaryIds[0],
                    SecondaryIds: [secondaryIds[1], primaryId]),
                primarySymbol);
        }

        var composeCandidate = secondaryIds
            .Prepend(primaryId)
            .Select(FindTickElementById)
            .Where(e => e is not null && e.RelatedTo.Length >= 2)
            .Cast<PlatonicElement>()
            .OrderByDescending(e => e.LocalTransformConfidence + e.BridgeConfidence)
            .FirstOrDefault();
        if (composeCandidate != null)
        {
            ExecuteIfPossible(
                new TickAction(
                    Kind: TickKind.Compose,
                    PrimaryElementId: composeCandidate.Id),
                primarySymbol);
            ExecuteIfPossible(
                new TickAction(
                    Kind: TickKind.Surprise,
                    PrimaryElementId: composeCandidate.Id),
                primarySymbol);
        }

        if (concepts.Count >= 4 && (_tickState.CurrentTick % 5 == 0))
        {
            ExecuteIfPossible(
                new TickAction(
                    Kind: TickKind.BranchDetect,
                    PrimaryElementId: primaryId),
                primarySymbol);
        }
    }

    private void PromoteDetectedPatterns(string operation, IReadOnlyList<PlatonicElement> candidates)
    {
        if (candidates.Count == 0)
            return;

        var state = _tickState;
        var existingPromotion = state.Elements.Any(e =>
            e.Kind == ElementKind.Function &&
            e.Symbol.Equals($"promoted:{operation}", StringComparison.OrdinalIgnoreCase));
        if (existingPromotion)
            return;

        var candidate = candidates
            .Where(e => e.Kind == ElementKind.Composition)
            .OrderByDescending(e => e.NoveltyScore)
            .FirstOrDefault();
        if (candidate is null || candidate.NoveltyScore < 0.75)
            return;

        var promoted = new PlatonicElement(
            Id: state.NextId,
            Kind: ElementKind.Function,
            Embedding: candidate.Embedding.ToArray(),
            Symbol: $"promoted:{operation}",
            GeneratedAtTick: state.CurrentTick,
            NoveltyScore: candidate.NoveltyScore,
            BridgeConfidence: Math.Max(0.5, candidate.BridgeConfidence),
            RelatedTo: ImmutableArray.Create(candidate.Id),
            GenerationPath: $"tick-promote:{operation}");

        _tickState = state with
        {
            Elements = state.Elements.Add(promoted),
            NextId = state.NextId + 1
        };
        _tickPatternPromotions++;
    }

    private int EnsureTickElement(string symbol, ElementKind kind)
    {
        var existing = _tickState.Elements.FirstOrDefault(e =>
            e.Symbol.Equals(symbol, StringComparison.OrdinalIgnoreCase) && e.Kind == kind);
        if (existing is not null)
            return existing.Id;

        var baseConfidence = 0.5;
        var kindBias = kind == ElementKind.Function ? 0.1 : 0.0;

        var element = new PlatonicElement(
            Id: _tickState.NextId,
            Kind: kind,
            Embedding: InputEmbeddingComposer.GetInputEmbedding(symbol, _tickState.EmbeddingDimension),
            Symbol: symbol,
            GeneratedAtTick: _tickState.CurrentTick,
            NoveltyScore: 0.6,
            BridgeConfidence: Clamp01(baseConfidence + kindBias),
            RelatedTo: ImmutableArray<int>.Empty,
            GenerationPath: "tick-observe");

        _tickState = _tickState with
        {
            Elements = _tickState.Elements.Add(element),
            NextId = _tickState.NextId + 1
        };

        return element.Id;
    }

    private PlatonicElement? FindTickElementById(int id)
        => _tickState.Elements.FirstOrDefault(e => e.Id == id);

    private void ObservePlatonicSpace(GenesisExample example, IReadOnlyList<string> concepts)
    {
        if (concepts.Count == 0)
            return;

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
        // Face-computable arithmetic: the homomorphism + operation→face coupling already represent it;
        // skip the relational input→output (operand↔result) coupling that would overload operands.
        if (SuppressArithmeticOperandRelations && hasArithmetic)
            return;

        var inputConcepts = ExtractMirrorConcepts(example.Input, string.Empty);
        var outputConcepts = ExtractMirrorConcepts(example.Output, string.Empty);
        if (inputConcepts.Count == 0 || outputConcepts.Count == 0)
        {
            // Bounded fallback: couple only ADJACENT concepts (sequential structure), not the full
            // O(n²) pairwise mesh — keeps contamination linear in the concept count.
            for (var i = 0; i + 1 < concepts.Count; i++)
                _platonicSpace.ObserveContradiction(concepts[i], concepts[i + 1], contradiction);
            _platonicSpace.FineEditFromExample(concepts, concepts, isNegative);
            return;
        }

        // Academic coupling: link each output concept to its SINGLE nearest input concept (the
        // genuinely-related pair, found via the lattice), NOT the full input×output mesh — and drop
        // output↔output coupling entirely (co-occurrence is not a relationship). This is the
        // minimum-contamination signal that still teaches the input→output association.
        foreach (var output in outputConcepts)
        {
            var partners = inputConcepts
                .Where(inp => !inp.Equals(output, StringComparison.OrdinalIgnoreCase))
                .ToArray();
            if (partners.Length == 0)
                continue;
            var nearest = _platonicSpace.GetNearestConcepts(output, candidates: partners, maxNeighbors: 1);
            var partner = nearest.Count > 0 ? nearest[0].Symbol : partners[0];
            _platonicSpace.ObserveContradiction(partner, output, contradiction);
        }

        _platonicSpace.FineEditFromExample(inputConcepts, outputConcepts, isNegative);
    }

    // CAUSAL REINFORCE reward for the edit-head. The token forward pass never reads the space, so a
    // token-loss delta was causally disconnected from the edit. Instead we measure a
    // SPACE-STATE-DEPENDENT outcome in [0,1]: AFTER the edit was applied (in ObservePlatonicSpace,
    // which ran just before this), does the platonic space now RETRIEVE the correct output for this
    // input? Retrieval is the universal op, so this generalizes across modalities. The reward is
    // outcome - the last-seen outcome for this example (REINFORCE baseline); first sighting -> 0.
    // This nudges PredictEditMagnitude toward the write strength that genuinely improves retrieval,
    // and the exploration noise added to m (in ObservePlatonicSpace) prevents the m->0 fixed point.
    // tokenLoss is retained in the signature (callers pass it) but no longer drives the reward.
    private void RewardEditHead(GenesisExample example, double tokenLoss)
    {
        _ = tokenLoss; // no longer causally relevant to the edit-head reward
        var pending = _pendingEditTokens;
        _pendingEditTokens = null;
        if (pending is null || pending.Count == 0)
            return;

        var outcome = ComputeEditOutcome(example);

        var key = (example.Input ?? string.Empty) + "" + (example.Output ?? string.Empty);
        var reward = _exampleOutcomeBaseline.TryGetValue(key, out var prev)
            ? outcome - prev     // positive when retrieval improved vs the last sighting of this example
            : 0.0;               // first sighting → neutral
        _exampleOutcomeBaseline[key] = outcome;

        // Bound the baseline map so it cannot grow without limit across a long run.
        if (_exampleOutcomeBaseline.Count > 50_000)
            _exampleOutcomeBaseline.Clear();

        _model.ReinforceEditHead(pending, _pendingEditMagnitude, reward);
    }

    // Space-state-dependent outcome in [0,1]: does the platonic space now retrieve/produce the correct
    // output for this input? Retrieval is the universal op (generalizes across modalities). Scored by
    // whether the target output concept is the nearest / in the top-K neighbours of the primary input
    // concept, graded by rank and distance; arithmetic correctness is credited via the face path.
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

        const int maxNeighbors = 8;
        var primaryInput = inputConcepts[0];
        var nearest = _platonicSpace.GetNearestConcepts(
            primaryInput,
            candidates: outputConcepts,
            maxNeighbors: maxNeighbors);
        if (nearest.Count == 0)
            return 0.0;

        var outputConceptSet = new HashSet<string>(
            outputConcepts.Select(c => c.ToLowerInvariant()),
            StringComparer.OrdinalIgnoreCase);

        for (var rank = 0; rank < nearest.Count; rank++)
        {
            if (!outputConceptSet.Contains(nearest[rank].Symbol))
                continue;

            // Graded by rank (nearest = best) AND by distance (closer = better); combine so the
            // single nearest exact hit scores ~1 and deeper/farther hits decay toward 0.
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

        // Tighten arithmetic links when equation is numerically correct — operand↔result relational
        // edges. These overload operands (3↔-1, 3↔4 …) and the face homomorphism already represents
        // the computation, so they are suppressible.
        if (!SuppressArithmeticOperandRelations)
        {
            var accuracyContradiction = Clamp01(arithmetic.AbsoluteError <= 1e-6 ? 0.05 : 0.6);
            _platonicSpace.ObserveContradiction(arithmetic.LeftToken, arithmetic.ResultToken, accuracyContradiction);
            _platonicSpace.ObserveContradiction(arithmetic.RightToken, arithmetic.ResultToken, accuracyContradiction);
        }
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
