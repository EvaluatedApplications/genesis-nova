using EvalApp.Consumer;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Data.Creators;
using GenesisNova.Licensing;
using GenesisNova.Model;
using GenesisNova.Persistence;
using GenesisNova.Train;

namespace GenesisNova.Runtime;

public sealed class GenesisEvalAppRuntime
{
    private readonly GenesisNovaConfig _runtimeConfig;
    private readonly GenesisRuntimeState _state;
    private readonly ICompiledPipeline<GenesisTrainTaskData> _trainPipeline;
    private readonly ICompiledPipeline<GenesisTrainOneTaskData> _trainOnePipeline;
    private readonly ICompiledPipeline<GenesisPredictTaskData> _predictPipeline;
    private readonly ICompiledPipeline<GenesisSaveTaskData> _savePipeline;
    private readonly ICompiledPipeline<GenesisLoadTaskData> _loadPipeline;
    private readonly ICompiledPipeline<GenesisConversationTaskData> _conversationPipeline;
    private readonly ICompiledPipeline<GenesisCompactConversationTaskData> _compactConversationPipeline;
    private readonly ICompiledPipeline<GenesisAutonomousTrainTaskData> _autonomousRoundPipeline;
    
    private readonly BestLossTracker _lossTracker = new();
    private readonly AutonomousHistoryStore _historyStore = new();
    private readonly GenesisCheckpointPersister _persister;
    private readonly SemaphoreSlim _modelOpsGate = new(1, 1);
    private string? _loadedCheckpointPath;
    private DateTime _loadedCheckpointWriteUtc = DateTime.MinValue;

    public GenesisEvalAppRuntime(GenesisNovaConfig? config = null)
    {
        _runtimeConfig = config ?? new GenesisNovaConfig();
        _state = new GenesisRuntimeState(_runtimeConfig);

        // Validate license for adaptive tuning
        var licenseMode = GenesisPipelineValidator.ValidateLicense();
        if (licenseMode == LicenseMode.Licensed)
        {
            Console.WriteLine("[License] ✓ Genesis Nova licensed - full adaptive tuning enabled");
        }
        else
        {
            Console.WriteLine("[License] ⓘ Genesis Nova unlicensed - running in sequential mode");
        }

        ICompiledPipeline<GenesisTrainTaskData> train = null!;
        ICompiledPipeline<GenesisTrainOneTaskData> trainOne = null!;
        ICompiledPipeline<GenesisPredictTaskData> predict = null!;
        ICompiledPipeline<GenesisSaveTaskData> save = null!;
        ICompiledPipeline<GenesisLoadTaskData> load = null!;
        ICompiledPipeline<GenesisConversationTaskData> conversation = null!;
        ICompiledPipeline<GenesisCompactConversationTaskData> compactConversation = null!;
        ICompiledPipeline<GenesisAutonomousTrainTaskData> autonomousRound = null!;

        var autonomousPlanner = new GenesisAutonomousTrainingPlanner();
        var planAutonomousRoundStep = new PlanAutonomousRoundStep(autonomousPlanner);
        var generateCandidatePoolItemStep = new GenerateCandidatePoolItemStep();
        var buildAutonomousBatchStep = new BuildAutonomousTrainingBatchStep();
        var runAutonomousRoundStep = new RunAutonomousTrainingRoundStep(_state.Orchestrator);

        // Data-driven parallelism for the candidate-pool prep ForEach: derived from
        // measured VRAM headroom + CPU width (see GpuResourceGatePlanner). The pipeline
        // (and therefore the ForEach TunableConfig) is compiled once here, before any
        // per-call request exists, so we bound the build-time ceiling by the request's
        // default MaxGenerationConcurrency. The adaptive tuner then scales within
        // [1, max] at runtime; lower per-request caps are additionally honored by the
        // planner when sizing per-creator pools.
        var poolPrepParallelism = GpuResourceGatePlanner.CandidatePoolPrepGate(
            requestedConcurrency: new GenesisAutonomousTrainingRequest().MaxGenerationConcurrency);

        _persister = new GenesisCheckpointPersister(_state, _runtimeConfig, _historyStore);

        var loadExamplesStep = new LoadExamplesStep();
        var runTrainingStep = new RunTrainingStep(_state);
        var saveCheckpointStep = new SaveCheckpointStep(_lossTracker, _persister);
        var trainOneStep = new TrainOneStep(_state);
        var predictGpuStep = new PredictGpuStep(_state);
        var saveStep = new SaveStep(_persister);
        var loadStep = new LoadStep(_state, _historyStore, _persister, _runtimeConfig);
        var observeConversationStep = new ObserveConversationStep(_state);
        var persistConversationStep = new PersistConversationStep(_persister);
        var compactConversationStep = new CompactConversationStep(_state);
        var persistCompactConversationStep = new PersistCompactConversationStep(_persister);

        Eval.App("GenesisNovaRuntime")
            .WithContext(NullGlobalContext.Instance)
            .WithResource(ResourceKind.Cpu, new TunableConfig(Min: 1, Max: Environment.ProcessorCount, Default: Math.Max(2, Environment.ProcessorCount / 2)))
            // GPU kernel concurrency stays serialized (single CUDA stream); see GpuResourceGatePlanner.
            .WithResource(ResourceKind.Of("gpu"), GpuResourceGatePlanner.GpuGate())
            .WithResource(ResourceKind.DiskIO, new TunableConfig(Min: 1, Max: 2, Default: 1))
            .WithTuning()  // Enable adaptive tuning (requires license)
            .DefineDomain("GenesisNova", _state)
                .DefineTask<GenesisTrainTaskData>("Train")
                    .Gate(ResourceKind.Cpu, null, g => g
                        .AddStep("LoadExamples", loadExamplesStep)
                        .AddStep("RunTraining", runTrainingStep))
                    .Gate(ResourceKind.DiskIO, null, g => g.AddStep("SaveCheckpoint", saveCheckpointStep))
                    .Run(out train)
                .DefineTask<GenesisTrainOneTaskData>("TrainOne")
                    .AddStep("TrainOneStep", trainOneStep)
                    .Run(out trainOne)
                .DefineTask<GenesisPredictTaskData>("Predict")
                    .Gate(ResourceKind.Of("gpu"), null, g => g.AddStep("PredictGpu", predictGpuStep))
                    .Run(out predict)
                .DefineTask<GenesisSaveTaskData>("Save")
                    .Gate(ResourceKind.DiskIO, null, g => g.AddStep("Save", saveStep))
                    .Run(out save)
                .DefineTask<GenesisLoadTaskData>("Load")
                    .Gate(ResourceKind.DiskIO, null, g => g.AddStep("Load", loadStep))
                    .Run(out load)
                .DefineTask<GenesisConversationTaskData>("Conversation")
                    .AddStep("Observe", observeConversationStep)
                    .Gate(ResourceKind.DiskIO, null, g => g.AddStep("Persist", persistConversationStep))
                    .Run(out conversation)
                .DefineTask<GenesisCompactConversationTaskData>("CompactConversation")
                    .AddStep("Compact", compactConversationStep)
                    .Gate(ResourceKind.DiskIO, null, g => g.AddStep("Persist", persistCompactConversationStep))
                    .Run(out compactConversation)
                .DefineTask<GenesisAutonomousTrainTaskData>("AutonomousTrainRound")
                    .AddStep("PlanRound", planAutonomousRoundStep)
                    .Gate(ResourceKind.Cpu, null, g => g
                        // Candidate-pool generation: resource-gated, VRAM/CPU-tunable ForEach.
                        // Replaces the former raw Parallel.ForEach so per-creator generation
                        // participates in the shared CPU gate + adaptive tuner. Parallelism
                        // bounds are derived from measured VRAM headroom (see GpuResourceGatePlanner).
                        // NOTE: the per-item body is NOT re-gated on Cpu. Nesting the SAME resource
                        // gate inside the outer Cpu gate self-deadlocks at 0% CPU — the parent holds
                        // the Cpu budget the children wait for. The outer gate plus the bounded
                        // ForEach parallelism already throttle per-creator generation.
                        .ForEach(
                            GenerateAutonomousCandidatePoolsStep.SelectWorkItems,
                            GenerateAutonomousCandidatePoolsStep.MergePools,
                            "CandidatePools",
                            poolPrepParallelism,
                            sub => sub.AddStep("GeneratePool", generateCandidatePoolItemStep))
                        .AddStep("BuildBatch", buildAutonomousBatchStep)
                        .AddStep("RunTraining", runAutonomousRoundStep))
                    .Run(out autonomousRound)
                .Build(GenesisPipelineValidator.ActiveKey); // EVALAPP license → adaptive CPU/GPU tuning (else sequential)

        TryBootstrapLatestState();
        EnsureConfiguredHiddenSize();

        _trainPipeline = train;
        _trainOnePipeline = trainOne;
        _predictPipeline = predict;
        _savePipeline = save;
        _loadPipeline = load;
        _conversationPipeline = conversation;
        _compactConversationPipeline = compactConversation;
        _autonomousRoundPipeline = autonomousRound;
    }

    public async Task<GenesisTrainingReport> TrainAsync(
        string filePath,
        int epochs,
        string? savePath = null,
        string? logPath = null,
        Action<string>? uiLogger = null)
    {
        return await WithModelGateAsync(async () =>
        {
            PerformGpuMemoryCleanup(uiLogger);
            var result = await _trainPipeline.RunAsync(new GenesisTrainTaskData(
                FilePath: filePath,
                Epochs: epochs,
                SavePath: savePath,
                LogPath: logPath,
                UiLogger: uiLogger));
            var data = ExtractData(result);
            return data.Report ?? throw new InvalidOperationException("Training report missing.");
        });
    }

    public async Task<GenesisAutonomousTrainingRun> TrainAutonomousAsync(
        GenesisAutonomousTrainingRequest request,
        CancellationToken cancellationToken = default,
        Action<string>? uiLogger = null,
        Func<GenesisAutonomousTrainingRequest, GenesisAutonomousTrainingRequest>? liveRequestUpdater = null,
        Action<object>? onRoundProgress = null)
    {
        return await WithModelGateAsync(async () =>
        {
            PerformGpuMemoryCleanup(uiLogger);
            var resetCreators = PublicTextCorpusCreator.ResetForFreshRun(ExampleCreatorRegistry.All);
            uiLogger?.Invoke(
                $"[auto] continuity: retained {_historyStore.History.Count} prior history rounds, corpus creators reset={resetCreators}, local corpus files preserved");
            var planningHistory = _historyStore.History.ToList();
            var rounds = new List<GenesisAutonomousTrainingRound>();
            GenesisTrainingReport? lastReport = null;
            var baseRequest = request;

            try
            {
                for (var round = 0; ; round++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var currentRequest = liveRequestUpdater?.Invoke(baseRequest) ?? baseRequest;
                    var unlimitedRounds = currentRequest.MaxRounds <= 0;
                    var configuredRounds = Math.Max(1, currentRequest.MaxRounds);
                    if (!unlimitedRounds && round >= configuredRounds)
                        break;

                    var roundLimitLabel = unlimitedRounds ? "∞" : configuredRounds.ToString();
                    uiLogger?.Invoke(
                        $"[auto] round {round + 1}/{roundLimitLabel}: planning... threshold={currentRequest.LossThreshold:F3}");
                    PerformGpuMemoryCleanup(uiLogger);

                    try
                    {
                        if (round > 0 && lastReport is not null)
                        {
                            onRoundProgress?.Invoke(new GenesisAutonomousTrainingEventPayload(
                                Round: round + 1,
                                StepName: "RoundStart",
                                Dataset: currentRequest.PreferredCreator ?? "mixed",
                                Loss: lastReport.AverageLoss.TokenLoss,
                                ExampleSuccessRate: lastReport.ExampleSuccessRate,
                                SamplesTrained: lastReport.ExampleCount,
                                ElapsedMs: 0,
                                SkippedCorrectExampleCount: lastReport.SkippedCorrectExampleCount,
                                PromptAnswerExampleCount: lastReport.PromptAnswerExampleCount,
                                WindowedTextExampleCount: lastReport.WindowedTextExampleCount,
                                CreatorSummary: BuildCreatorSummary(lastReport)));
                        }

                        var result = await _autonomousRoundPipeline.RunAsync(
                            new GenesisAutonomousTrainTaskData(
                               Request: currentRequest,
                               History: planningHistory,
                               RoundIndex: round,
                               CancellationToken: cancellationToken,
                               UiLogger: uiLogger),
                            cancellationToken);
                        var roundData = ExtractData(result);
                        var plan = roundData.Plan ?? throw new InvalidOperationException("Autonomous plan missing.");
                        var report = roundData.Report ?? throw new InvalidOperationException("Autonomous round report missing.");
                        var examples = roundData.TrainingExamples ?? [];
                        lastReport = report;
                        uiLogger?.Invoke(
                            $"[auto] round {round + 1}/{roundLimitLabel}: datasets={plan.CreatorPlans.Count} " +
                            $"trained={examples.Count} epochs={plan.Epochs} loss={report.AverageLoss.TokenLoss:F4} " +
                            $"example_success={report.ExampleSuccessRate:P1} " +
                            $"correct={report.CorrectExampleCount} incorrect={report.IncorrectExampleCount} " +
                            $"skipped_correct={report.SkippedCorrectExampleCount} prompt_answer={report.PromptAnswerExampleCount} windowed_text={report.WindowedTextExampleCount}");
                        if (report.CreatorProgress is { Count: > 0 })
                        {
                            foreach (var creator in report.CreatorProgress)
                            {
                                uiLogger?.Invoke(
                                    $"[auto] creator={creator.CreatorName} loss={creator.AverageTokenLoss:F4} success={creator.SuccessRate:P1} seen={creator.SeenCount}");
                            }
                        }

                        // Fire round progress event for UI update
                        onRoundProgress?.Invoke(new GenesisAutonomousTrainingEventPayload(
                            Round: round + 1,
                            StepName: "RunTraining",
                            Dataset: currentRequest.PreferredCreator ?? "mixed",
                            Loss: report.AverageLoss.TokenLoss,
                            ExampleSuccessRate: report.ExampleSuccessRate,
                            SamplesTrained: examples.Count,
                            ElapsedMs: 0,
                            SkippedCorrectExampleCount: report.SkippedCorrectExampleCount,
                            PromptAnswerExampleCount: report.PromptAnswerExampleCount,
                            WindowedTextExampleCount: report.WindowedTextExampleCount,
                            CreatorSummary: BuildCreatorSummary(report)));

                        var avgDifficulty = plan.CreatorPlans.Count == 0
                            ? 0
                            : (int)Math.Round(plan.CreatorPlans.Average(p => p.Difficulty));
                        var roundResult = new GenesisAutonomousTrainingRound(
                            Round: round + 1,
                            CreatorName: "mixed:all-datasets",
                            SampleCount: examples.Count,
                            Difficulty: avgDifficulty,
                            Epochs: plan.Epochs,
                            Report: report,
                            CreatorProgress: null);
                        rounds.Add(roundResult);
                        var creatorRounds = (roundData.CreatorRounds ?? [])
                            .Select(r =>
                            {
                                var creatorProgress = report.CreatorProgress?.FirstOrDefault(p =>
                                    p.CreatorName.Equals(r.CreatorName, StringComparison.OrdinalIgnoreCase));
                                return r with { Report = report, CreatorProgress = creatorProgress };
                            })
                            .ToArray();
                        foreach (var creatorRound in creatorRounds)
                        {
                            planningHistory.Add(creatorRound);
                            _historyStore.Append(creatorRound);
                        }

                        var currentLoss = report.AverageLoss.TokenLoss;
                        var improved = currentLoss < _lossTracker.BestLoss;
                        if (improved)
                            _lossTracker.BestLoss = currentLoss;

                        _persister.Persist(
                            reason: improved ? "auto-train-improved" : "auto-train-completed",
                            detail: $"datasets={plan.CreatorPlans.Count} trained={examples.Count} epochs={plan.Epochs} loss={currentLoss:F4} improved={improved}",
                            exampleCount: examples.Count,
                            loss: currentLoss);
                    }
                    catch (System.AccessViolationException ex)
                    {
                        uiLogger?.Invoke($"[auto] round {round + 1}/{roundLimitLabel}: AccessViolation - {ex.Message}");
                        uiLogger?.Invoke("[auto] This indicates a native memory issue, likely in GPU operations. Stopping training.");
                        throw new InvalidOperationException("Native memory corruption detected during autonomous training. Please restart and try with smaller batch sizes or fewer epochs.", ex);
                    }
                    catch (Exception ex) when (IsGpuOutOfMemory(ex))
                    {
                        uiLogger?.Invoke($"[auto] round {round + 1}/{roundLimitLabel}: GPU OOM captured - {ex.Message}");
                        uiLogger?.Invoke("[auto] OOM was captured from training logs/exceptions. This round is being skipped and training will continue.");
                        PerformGpuMemoryCleanup(uiLogger);
                        continue;
                    }
                }
            }
            finally
            {
                _persister.Persist(
                    reason: "auto-train-state",
                    detail: $"rounds={rounds.Count}",
                    exampleCount: rounds.Count,
                    loss: lastReport?.AverageLoss.TokenLoss);
            }

            return new GenesisAutonomousTrainingRun(request, rounds, lastReport);
        }, cancellationToken);
    }

    /// <summary>
    /// Register an operation token (a language verb, e.g. "find") as a ROUTE TRIGGER: it is excluded from
    /// relation coupling/anchoring (so it never overloads a target and collapses retrieval) while the GRU
    /// route head still learns it from the raw input. Re-register from the language definition at startup —
    /// it is not persisted, being a pure function of that definition. See LANGUAGE_CREATOR.md §2.
    /// </summary>
    public void RegisterOperationToken(string token) => _state.Memory.RegisterOperationToken(token);

    public async Task<GenesisStepLoss> TrainOneAsync(GenesisExample example)
    {
        return await WithModelGateAsync(async () =>
        {
            var result = await _trainOnePipeline.RunAsync(new GenesisTrainOneTaskData(example));
            var data = ExtractData(result);
            return data.Loss ?? throw new InvalidOperationException("Loss missing.");
        });
    }

    public async Task<GenesisPredictTaskData> PredictAsync(
        string input,
        int maxTokens = 48)
    {
        return await WithModelGateAsync(() => RunPredictCoreAsync(input, maxTokens));
    }

    public async Task<GenesisPredictTaskData?> TryPredictAsync(
        string input,
        int maxTokens = 48,
        int gateWaitMilliseconds = 150,
        CancellationToken cancellationToken = default)
    {
        var waitMs = Math.Max(1, gateWaitMilliseconds);
        if (!await _modelOpsGate.WaitAsync(waitMs, cancellationToken))
            return null;

        try
        {
            return await RunPredictCoreAsync(input, maxTokens);
        }
        finally
        {
            _modelOpsGate.Release();
        }
    }

    public async Task SaveAsync(string path)
        => _ = await WithModelGateAsync(async () => await _savePipeline.RunAsync(new GenesisSaveTaskData(path)));

    public async Task LoadAsync(string path)
        => _ = await WithModelGateAsync(async () =>
        {
            _ = await _loadPipeline.RunAsync(new GenesisLoadTaskData(path));
            TrackLoadedCheckpoint(path);
            return true;
        });

    public async Task<bool> RollbackToLastGoodAsync()
        => await WithModelGateAsync(() =>
        {
            var path = GenesisLocalStateStore.ResolveLastGoodCheckpointPath(_runtimeConfig);
            if (!File.Exists(path))
                return Task.FromResult(false);

            LoadStateFromCheckpoint(path);
            GenesisLocalStateStore.AppendJournalEntry(
                _runtimeConfig,
                "rollback-last-good",
                detail: path);
            return Task.FromResult(true);
        });

    public async Task<GenesisConversationTaskData> ObserveConversationAsync(
        string userInput,
        string assistantOutput,
        bool resetSignal = false,
        string? note = null)
    {
        var result = await _conversationPipeline.RunAsync(new GenesisConversationTaskData(
            UserInput: userInput,
            AssistantOutput: assistantOutput,
            ResetSignal: resetSignal,
            Note: note));
        return ExtractData(result);
    }

    public async Task<GenesisCompactConversationTaskData> CompactConversationAsync(string? note = null)
    {
            var result = await _compactConversationPipeline.RunAsync(new GenesisCompactConversationTaskData(Note: note));
            return ExtractData(result);
    }

    public async Task<GenesisEvaluationReport> EvaluateFileAsync(string filePath, int? maxSamples = null)
    {
        var examples = await GenesisTrainingDataLoader.LoadFromFileAsync(filePath);
        if (maxSamples.HasValue && maxSamples.Value > 0)
            examples = examples.Take(maxSamples.Value).ToArray();

        var exact = 0;
        foreach (var ex in examples)
        {
            var pred = await PredictAsync(ex.Input, maxTokens: 48);
            var output = pred.Result?.Output?.Trim().ToLowerInvariant() ?? string.Empty;
            var expected = ex.Output.Trim().ToLowerInvariant();
            // Face-aware: a digit and its number-word both count (see AnswerEquivalence).
            if (GenesisNova.Core.AnswerEquivalence.Equivalent(output, expected))
                exact++;
        }

        var sampleCount = examples.Count;
        var exactAcc = sampleCount == 0 ? 0.0 : exact / (double)sampleCount;
        return new GenesisEvaluationReport(
            SampleCount: sampleCount,
            ExactMatchCount: exact,
            RouteLabeledCount: 0,
            RouteCorrectCount: 0,
            ExactMatchAccuracy: exactAcc,
            RouteAccuracy: 0.0);
    }

    public int VocabularySize => WithModelGate(() => _state.Tokenizer.VocabularySize);
    public int HiddenSize => WithModelGate(() => _state.Model.HiddenSize);

    /// <summary>The SGD step size, exposed so a caller can implement a training SCHEDULE (e.g. anneal the LR as
    /// a lesson approaches mastery, the way the autonomous orchestrator does). Reads/writes the live model.</summary>
    public double LearningRate
    {
        get => WithModelGate(() => _state.Model.LearningRate);
        set => WithModelGate(() => { _state.Model.LearningRate = value; return true; });
    }
    public string ConversationBrief => WithModelGate(() => _state.Conversation.BuildContextBrief());
    public bool AutoResumeEnabled => _runtimeConfig.AutoResume;
    public string AutoCheckpointPath => GenesisLocalStateStore.ResolveCheckpointPath(_runtimeConfig);
    public GenesisNovaConfig Config => _runtimeConfig;
    
    public void UpdateConfig(Func<GenesisNovaConfig, GenesisNovaConfig> updater)
    {
        WithModelGate(() =>
        {
            var updated = updater(_runtimeConfig);
            // Update the model's internal config reference
            _state.Model.UpdateConfig(updated);
            return true;
        });
    }
    
    public void EnsureHiddenSize(int targetSize)
    {
        WithModelGate(() =>
        {
            if (targetSize > _state.Model.HiddenSize)
                _state.Model.EnsureHiddenSize(targetSize);
            return true;
        });
    }
    
    public int[] EncodeTokens(string text) => WithModelGate(() => _state.Tokenizer.Encode(text));
    public string DecodeTokens(IReadOnlyList<int> tokens) => WithModelGate(() => _state.Tokenizer.Decode(tokens));
    public string TokenText(int tokenId)
    {
        return WithModelGate(() =>
        {
            var vocab = _state.Tokenizer.Vocabulary;
            return tokenId >= 0 && tokenId < vocab.Count
                ? vocab[tokenId]
                : $"<unk:{tokenId}>";
        });
    }

    public PlatonicActivationView AnalyzePlatonicActivation(string input, int maxNodes = 24, int maxEdges = 40)
    {
        return WithModelGate(() =>
        {
            var safeInput = input ?? string.Empty;
            var tokenIds = _state.Tokenizer.Encode(safeInput);
            var tokenTexts = tokenIds.Select(id =>
            {
                var vocab = _state.Tokenizer.Vocabulary;
                return id >= 0 && id < vocab.Count ? vocab[id] : $"<unk:{id}>";
            }).ToArray();
            var lexicalParts = System.Text.RegularExpressions.Regex
                .Matches(safeInput.ToLowerInvariant(), @"-?\d+(?:\.\d+)?|[a-z]+|[+\-*/x]")
                .Select(m => m.Value)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();

            var anchorSet = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var token in tokenTexts.Concat(lexicalParts))
            {
                if (_state.Memory.ContainsConcept(token))
                    anchorSet.Add(token);
            }

            var anchors = anchorSet.OrderBy(a => a, StringComparer.OrdinalIgnoreCase).ToArray();
            var snapshot = _state.Memory.ExportSnapshot();
            var nodes = new List<PlatonicActivatedNode>(snapshot.Nodes.Length);
            var anchorHash = new HashSet<string>(anchors, StringComparer.OrdinalIgnoreCase);

            foreach (var node in snapshot.Nodes)
            {
                var isAnchor = anchorHash.Contains(node.Name);
                var baseScore = isAnchor ? 1.0 : 0.0;
                if (!isAnchor && anchors.Length > 0)
                {
                    baseScore = anchors
                        .Select(anchor => 1.0 - _state.Memory.GetContradiction(anchor, node.Name))
                        .DefaultIfEmpty(0.0)
                        .Max();
                }

                var obsBoost = Math.Min(0.20, Math.Log10(Math.Max(1, node.ObservationCount)) / 10.0);
                var score = Math.Max(0.0, Math.Min(1.0, baseScore + obsBoost));
                nodes.Add(new PlatonicActivatedNode(node.Name, score, node.ObservationCount, isAnchor));
            }

            var selectedNodes = nodes
                .Where(n => !GenesisNova.Cognition.PlatonicSpaceMemory.IsReservedConcept(n.Name)) // hide internal face: routing markers
                .OrderByDescending(n => n.IsAnchor)
                .ThenByDescending(n => n.Score)
                .ThenByDescending(n => n.ObservationCount)
                .Take(Math.Max(4, maxNodes))
                .ToArray();

            var selectedSet = new HashSet<string>(selectedNodes.Select(n => n.Name), StringComparer.OrdinalIgnoreCase);
            var edges = snapshot.Relations
                .Where(r =>
                    (selectedSet.Contains(r.Left) && selectedSet.Contains(r.Right)) ||
                    anchorHash.Contains(r.Left) || anchorHash.Contains(r.Right))
                .Select(r =>
                {
                    var confidence = 1.0 - r.SynthesisContradiction;
                    var obsBoost = Math.Min(0.15, Math.Log10(Math.Max(1, r.ObservationCount)) / 12.0);
                    var score = Math.Max(0.0, Math.Min(1.0, confidence + obsBoost));
                    return new PlatonicActivatedEdge(
                        Left: r.Left,
                        Right: r.Right,
                        Score: score,
                        Contradiction: r.SynthesisContradiction,
                        ObservationCount: r.ObservationCount);
                })
                .OrderByDescending(e => e.Score)
                .ThenByDescending(e => e.ObservationCount)
                .Take(Math.Max(8, maxEdges))
                .ToArray();

            return new PlatonicActivationView(
                Input: safeInput,
                InputTokens: tokenTexts,
                Anchors: anchors,
                Nodes: selectedNodes,
                Edges: edges);
        });
    }

    /// <summary>
    /// Structured introspection of the live model + platonic substrate (for the diagnostic CLI). Reads the
    /// SAME runtime state inference uses, under the model gate, so it never diverges from what the model does.
    /// </summary>
    public GenesisRuntimeDiagnostics Diagnose(int topRelations = 12, int topFunctions = 16, int topChunks = 12)
    {
        return WithModelGate(() =>
        {
            var model = _state.Model;
            var mem = _state.Memory;
            var trainer = _state.Trainer;

            var transforms = trainer.TransformAccumulator.ExportSnapshot().Transforms;
            var folds = trainer.FoldPathDiscovery.ExportSnapshot();
            var chunks = mem.ExportSnapshot().Chunks ?? Array.Empty<PlatonicChunkSnapshot>();

            var funcs = mem.FunctionElements;
            var funcById = funcs.ToDictionary(f => f.Id, f => f.Symbol);
            var functionSummaries = funcs
                .Take(topFunctions)
                .Select(f => new FunctionElementSummary(
                    f.Symbol,
                    f.RelatedTo.Select(id => funcById.TryGetValue(id, out var s) ? s : $"#{id}").ToArray()))
                .ToArray();

            var topRel = mem.GetAllRelations()
                .OrderByDescending(r => r.ObservationCount)
                .ThenBy(r => r.Left, StringComparer.OrdinalIgnoreCase)
                .Take(topRelations)
                .Select(r => new RelationSummary(r.Left, r.Right, r.ObservationCount))
                .ToArray();

            var path = GenesisLocalStateStore.ResolveCheckpointPath(_runtimeConfig);

            return new GenesisRuntimeDiagnostics(
                CheckpointPath: path,
                CheckpointExists: File.Exists(path),
                CheckpointWriteUtc: File.Exists(path) ? File.GetLastWriteTimeUtc(path) : null,
                Backend: _runtimeConfig.Backend.ToString(),
                HiddenSize: model.HiddenSize,
                FaceDimension: mem.FaceDimension,
                VocabularySize: _state.Tokenizer.VocabularySize,
                ParameterCount: model.ParameterCount(),
                PlanKindCount: GenesisNeuralModel.PlanKindCount,
                NodeCount: mem.NodeCount,
                RelationCount: mem.RelationCount,
                FunctionElementCount: funcs.Count,
                LearnedTransformCount: transforms.Count,
                FoldPathCount: folds.FoldPaths.Count,
                LogLinearFitCount: folds.LogLinearFits.Count,
                ChunkTagCount: chunks.Select(c => c.Tag).Distinct(StringComparer.OrdinalIgnoreCase).Count(),
                ChunkCount: chunks.Length,
                AutonomousRounds: _historyStore.History.Count,
                MaxNodes: Math.Max(256, _runtimeConfig.MaxPlatonicNodes),
                MaxRelations: Math.Max(1024, _runtimeConfig.MaxPlatonicRelations),
                SpaceManagerEnabled: _runtimeConfig.AutoManagePlatonicSpace,
                // Soft relation budget the SpaceManager prunes toward: nodes×TargetRelationsPerNode(6)+NodeBuffer(128),
                // clamped to [MinRelations(1024), MaxRelations]. Relation-pressure = RelationCount / this budget.
                RelationBudget: Math.Clamp(mem.NodeCount * 6 + 128, 1024, Math.Max(1024, _runtimeConfig.MaxPlatonicRelations)),
                TopRelations: topRel,
                FunctionElements: functionSummaries,
                LearnedTransforms: transforms
                    .Take(topFunctions)
                    .Select(t => new TransformSummary(t.FunctionName, t.ObservationCount, t.Confidence, t.State.ToString()))
                    .ToArray(),
                Chunks: chunks
                    .OrderByDescending(c => c.Count)
                    .Take(topChunks)
                    .Select(c => new ChunkSummary(c.Tag, c.Chunk, c.Count))
                    .ToArray());
        });
    }

    private void TryBootstrapLatestState()
    {
        if (!_runtimeConfig.AutoResume)
            return;

        if (!GenesisLocalStateStore.TryResolveBootstrapCheckpoint(_runtimeConfig, out var path) || !File.Exists(path))
            return;

        LoadStateFromCheckpoint(path);
        GenesisLocalStateStore.AppendJournalEntry(
            _runtimeConfig,
            "bootstrap",
            detail: path);
    }

    private void RefreshLatestStateForReplPredict()
    {
        if (!GenesisLocalStateStore.TryResolveBootstrapCheckpoint(_runtimeConfig, out var path) || !File.Exists(path))
            return;

        var lastWriteUtc = File.GetLastWriteTimeUtc(path);
        if (string.Equals(path, _loadedCheckpointPath, StringComparison.OrdinalIgnoreCase) &&
            lastWriteUtc == _loadedCheckpointWriteUtc)
            return;

        LoadStateFromCheckpoint(path);
    }

    private void LoadStateFromCheckpoint(string path)
    {
        var loaded = GenesisCheckpointStore.LoadForRuntime(path, _runtimeConfig);
        _state.Replace(loaded.Config, loaded.Tokenizer, loaded.Model, loaded.PlatonicSpace, loaded.Conversation, loaded.TrainerLearningStateJson);
        _historyStore.Restore(loaded.AutonomousTraining);
        TrackLoadedCheckpoint(path);
    }

    private void TrackLoadedCheckpoint(string path)
    {
        _loadedCheckpointPath = path;
        _loadedCheckpointWriteUtc = File.Exists(path)
            ? File.GetLastWriteTimeUtc(path)
            : DateTime.MinValue;
    }

    private void EnsureConfiguredHiddenSize()
    {
        var target = Math.Max(_runtimeConfig.HiddenSize, _state.Model.HiddenSize);
        if (target > _state.Model.HiddenSize)
            _state.Model.EnsureHiddenSize(target);
    }

    private static T ExtractData<T>(PipelineResult<T> result)
        => result switch
        {
            PipelineResult<T>.Success s => s.Data,
            PipelineResult<T>.Failure f => throw new InvalidOperationException(f.Message ?? "Pipeline failed.", f.Exception),
            _ => throw new InvalidOperationException("Unsupported pipeline result.")
        };

    private async Task<GenesisPredictTaskData> RunPredictCoreAsync(string input, int maxTokens)
    {
        RefreshLatestStateForReplPredict();
        var result = await _predictPipeline.RunAsync(new GenesisPredictTaskData(
            Input: input,
            MaxNewTokens: maxTokens));
        return ExtractData(result);
    }

    private T WithModelGate<T>(Func<T> action)
    {
        _modelOpsGate.Wait();
        try
        {
            return action();
        }
        finally
        {
            _modelOpsGate.Release();
        }
    }

    private async Task<T> WithModelGateAsync<T>(Func<Task<T>> action, CancellationToken cancellationToken = default)
    {
        await _modelOpsGate.WaitAsync(cancellationToken);
        try
        {
            return await action();
        }
        finally
        {
            _modelOpsGate.Release();
        }
    }

    /// <summary>Delete N most recent REPL conversation turns.</summary>
    public async Task DeleteConversationTurnsAsync(int count)
    {
        await _state.Conversation.DeleteRecentTurnsAsync(count);
    }

    /// <summary>Clear all REPL conversation history.</summary>
    public async Task ClearConversationAsync()
    {
        await _state.Conversation.ClearAsync();
    }

    private void PerformGpuMemoryCleanup(Action<string>? logger = null)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        logger?.Invoke("[gpu] memory cleanup complete");
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

    private static string BuildCreatorSummary(GenesisTrainingReport report)
    {
        if (report.CreatorProgress is not { Count: > 0 })
            return string.Empty;

        var header = $"mix: prompt-answer {report.PromptAnswerExampleCount}, windowed text {report.WindowedTextExampleCount}, skipped correct {report.SkippedCorrectExampleCount}";
        var body = string.Join(
            Environment.NewLine,
            report.CreatorProgress
                .OrderBy(x => x.CreatorName, StringComparer.OrdinalIgnoreCase)
                .Select(x => $"{x.CreatorName} [{x.TrainingKind}]: loss {x.AverageTokenLoss:F3}, succ {x.SuccessRate:P0}, seen {x.SeenCount}"));
        return string.Join(Environment.NewLine, header, body);
    }

}
