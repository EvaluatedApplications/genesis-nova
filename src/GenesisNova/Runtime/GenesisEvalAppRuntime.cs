using EvalApp.Consumer;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Data.Creators;
using GenesisNova.Licensing;
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
        var generateAutonomousPoolsStep = new GenerateAutonomousCandidatePoolsStep();
        var buildAutonomousBatchStep = new BuildAutonomousTrainingBatchStep();
        var runAutonomousRoundStep = new RunAutonomousTrainingRoundStep(_state.Orchestrator);

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
            .WithResource(ResourceKind.Of("gpu"), new TunableConfig(Min: 1, Max: 1, Default: 1))  // GPU serialized to 1
            .WithResource(ResourceKind.DiskIO, new TunableConfig(Min: 1, Max: 2, Default: 1))
            .WithTuning()  // Enable adaptive tuning (requires license)
            .DefineDomain("GenesisNova", _state)
                .DefineTask<GenesisTrainTaskData>("Train")
                    .Gate(ResourceKind.Cpu, null, g => g
                        .AddStep("LoadExamples", loadExamplesStep.Execute)
                        .AddStep("RunTraining", runTrainingStep.Execute))
                    .Gate(ResourceKind.DiskIO, null, g => g.AddStep("SaveCheckpoint", saveCheckpointStep.Execute))
                    .Run(out train)
                .DefineTask<GenesisTrainOneTaskData>("TrainOne")
                    .AddStep("TrainOneStep", trainOneStep.Execute)
                    .Run(out trainOne)
                .DefineTask<GenesisPredictTaskData>("Predict")
                    .Gate(ResourceKind.Of("gpu"), null, g => g.AddStep("PredictGpu", predictGpuStep.Execute))
                    .Run(out predict)
                .DefineTask<GenesisSaveTaskData>("Save")
                    .Gate(ResourceKind.DiskIO, null, g => g.AddStep("Save", saveStep.Execute))
                    .Run(out save)
                .DefineTask<GenesisLoadTaskData>("Load")
                    .Gate(ResourceKind.DiskIO, null, g => g.AddStep("Load", loadStep.Execute))
                    .Run(out load)
                .DefineTask<GenesisConversationTaskData>("Conversation")
                    .AddStep("Observe", observeConversationStep.Execute)
                    .Gate(ResourceKind.DiskIO, null, g => g.AddStep("Persist", persistConversationStep.Execute))
                    .Run(out conversation)
                .DefineTask<GenesisCompactConversationTaskData>("CompactConversation")
                    .AddStep("Compact", compactConversationStep.Execute)
                    .Gate(ResourceKind.DiskIO, null, g => g.AddStep("Persist", persistCompactConversationStep.Execute))
                    .Run(out compactConversation)
                .DefineTask<GenesisAutonomousTrainTaskData>("AutonomousTrainRound")
                    .AddStep("PlanRound", planAutonomousRoundStep.Execute)
                    .Gate(ResourceKind.Cpu, null, g => g
                        .AddStep("GeneratePools", generateAutonomousPoolsStep.Execute)
                        .AddStep("BuildBatch", buildAutonomousBatchStep.Execute)
                        .AddStep("RunTraining", runAutonomousRoundStep.Execute))
                    .Run(out autonomousRound)
                .Build();

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
            _historyStore.Clear();
            var resetCreators = PublicTextCorpusCreator.ResetForFreshRun(ExampleCreatorRegistry.All);
            uiLogger?.Invoke(
                $"[auto] fresh run reset: history cleared, corpus creators reset={resetCreators}, local corpus files preserved");
            var planningHistory = new List<GenesisAutonomousTrainingRound>();
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

                        var currentLoss = report.AverageLoss.TokenLoss;
                        if (currentLoss < _lossTracker.BestLoss)
                        {
                            _lossTracker.BestLoss = currentLoss;
                            _persister.Persist(
                                reason: "auto-train-improved",
                                detail: $"datasets={plan.CreatorPlans.Count} trained={examples.Count} epochs={plan.Epochs} loss={currentLoss:F4}",
                                exampleCount: examples.Count,
                                loss: currentLoss);
                        }

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
        return await WithModelGateAsync(async () =>
        {
            var result = await _predictPipeline.RunAsync(new GenesisPredictTaskData(
                Input: input,
                MaxNewTokens: maxTokens));
            return ExtractData(result);
        });
    }

    public async Task SaveAsync(string path)
        => _ = await WithModelGateAsync(async () => await _savePipeline.RunAsync(new GenesisSaveTaskData(path)));

    public async Task LoadAsync(string path)
        => _ = await WithModelGateAsync(async () => await _loadPipeline.RunAsync(new GenesisLoadTaskData(path)));

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
            if (output == expected)
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

    public int VocabularySize => _state.Tokenizer.VocabularySize;
    public int HiddenSize => _state.Model.HiddenSize;
    public string ConversationBrief => _state.Conversation.BuildContextBrief();
    public bool AutoResumeEnabled => _runtimeConfig.AutoResume;
    public string AutoCheckpointPath => GenesisLocalStateStore.ResolveCheckpointPath(_runtimeConfig);
    public GenesisNovaConfig Config => _runtimeConfig;
    
    public void UpdateConfig(Func<GenesisNovaConfig, GenesisNovaConfig> updater)
    {
        var updated = updater(_runtimeConfig);
        // Update the model's internal config reference
        _state.Model.UpdateConfig(updated);
    }
    
    public void EnsureHiddenSize(int targetSize)
    {
        if (targetSize > _state.Model.HiddenSize)
            _state.Model.EnsureHiddenSize(targetSize);
    }
    
    public int[] EncodeTokens(string text) => _state.Tokenizer.Encode(text);
    public string DecodeTokens(IReadOnlyList<int> tokens) => _state.Tokenizer.Decode(tokens);
    public string TokenText(int tokenId)
    {
        var vocab = _state.Tokenizer.Vocabulary;
        return tokenId >= 0 && tokenId < vocab.Count
            ? vocab[tokenId]
            : $"<unk:{tokenId}>";
    }

    public PlatonicActivationView AnalyzePlatonicActivation(string input, int maxNodes = 24, int maxEdges = 40)
    {
        var safeInput = input ?? string.Empty;
        var tokenIds = _state.Tokenizer.Encode(safeInput);
        var tokenTexts = tokenIds.Select(TokenText).ToArray();
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
    }

    private void TryBootstrapLatestState()
    {
        if (!_runtimeConfig.AutoResume)
            return;

        if (!GenesisLocalStateStore.TryResolveBootstrapCheckpoint(_runtimeConfig, out var path) || !File.Exists(path))
            return;

        var loaded = GenesisCheckpointStore.LoadForRuntime(path, _runtimeConfig);
        _state.Replace(loaded.Config, loaded.Tokenizer, loaded.Model, loaded.PlatonicSpace, loaded.Conversation);
        _historyStore.Restore(loaded.AutonomousTraining);
        GenesisLocalStateStore.AppendJournalEntry(
            _runtimeConfig,
            "bootstrap",
            detail: path);
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
