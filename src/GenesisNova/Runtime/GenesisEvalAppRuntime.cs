using EvalApp.Consumer;
using GenesisNova.Cognition;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Infer;
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
    
    private double _bestTrainingLoss = double.MaxValue;  // Track best loss for conditional saving
    private AxiomaticPermutationEngine? _permutationEngine;
    private readonly SemaphoreSlim _modelOpsGate = new(1, 1);
    private readonly List<GenesisAutonomousTrainingRound> _autonomousHistory = [];
    private const int MaxAutonomousHistory = 512;

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

        Eval.App("GenesisNovaRuntime")
            .WithContext(NullGlobalContext.Instance)
            .WithResource(ResourceKind.Cpu, new TunableConfig(Min: 1, Max: Environment.ProcessorCount, Default: Math.Max(2, Environment.ProcessorCount / 2)))
            .WithResource(ResourceKind.Of("gpu"), new TunableConfig(Min: 1, Max: 1, Default: 1))  // GPU serialized to 1
            .WithResource(ResourceKind.DiskIO, new TunableConfig(Min: 1, Max: 2, Default: 1))
            .WithTuning()  // Enable adaptive tuning (requires license)
            .DefineDomain("GenesisNova", _state)
                .DefineTask<GenesisTrainTaskData>("Train")
                    .AddStep("LoadExamples", data => data with
                    {
                        Examples = GenesisTrainingDataLoader.LoadFromFile(data.FilePath)
                    })
                    .AddStep("RunTraining", data =>
                    {
                        StreamWriter? writer = null;
                        Action<string>? logger = null;
                        if (!string.IsNullOrWhiteSpace(data.LogPath))
                        {
                            var dir = Path.GetDirectoryName(data.LogPath);
                            if (!string.IsNullOrWhiteSpace(dir))
                                Directory.CreateDirectory(dir);
                            writer = new StreamWriter(data.LogPath, append: false);
                            
                            if (data.UiLogger != null)
                            {
                                // Chain both file logging and UI logger
                                logger = line =>
                                {
                                    writer.WriteLine(line);
                                    writer.Flush();
                                    data.UiLogger?.Invoke(line);
                                };
                            }
                            else
                            {
                                // File logging only
                                logger = line =>
                                {
                                    writer.WriteLine(line);
                                    writer.Flush();
                                };
                            }
                        }
                        else if (data.UiLogger != null)
                        {
                            // UI logger only (no file)
                            logger = data.UiLogger;
                        }

                        try
                        {
                            var report = _state.Orchestrator.Train(
                                data.Examples ?? [],
                                data.Epochs,
                                logger);
                            return data with { Report = report };
                        }
                        finally
                        {
                            writer?.Dispose();
                        }
                    })
                    .Gate(ResourceKind.DiskIO, null, g => g.AddStep("SaveCheckpoint", data =>
                    {
                        // Only persist if loss improved or is first checkpoint
                        var currentLoss = data.Report?.AverageLoss.TotalLoss ?? double.MaxValue;
                        if (currentLoss < _bestTrainingLoss)
                        {
                            _bestTrainingLoss = currentLoss;
                            PersistCheckpoint(
                                reason: "train-improved",
                                explicitPath: data.SavePath,
                                detail: $"epochs={data.Epochs} loss={currentLoss:F4}",
                                exampleCount: data.Examples?.Count ?? 0,
                                loss: currentLoss);
                        }
                        return data;
                    }))
                    .Run(out train)
                .DefineTask<GenesisTrainOneTaskData>("TrainOne")
                    .AddStep("TrainOneStep", data =>
                    {
                        var loss = _state.Trainer.TrainStep(data.Example);
                        return data with { Loss = loss };
                    })
                    .Run(out trainOne)
                .DefineTask<GenesisPredictTaskData>("Predict")
                    .Gate(ResourceKind.Of("gpu"), null, g => g.AddStep("PredictGpu", data =>
                    {
                        var result = _state.Inference.Generate(new GenerationRequest(
                            Input: data.Input,
                            MaxNewTokens: data.MaxNewTokens));
                        _state.Trainer.ObserveInferenceResult(data.Input, result.Output);
                        return data with { Result = result };
                    }))
                    .Run(out predict)
                .DefineTask<GenesisSaveTaskData>("Save")
                    .Gate(ResourceKind.DiskIO, null, g => g.AddStep("Save", data =>
                    {
                        PersistCheckpoint(
                            reason: "save",
                            explicitPath: data.Path,
                            detail: "manual");
                        return data with { Saved = true };
                    }))
                    .Run(out save)
                .DefineTask<GenesisLoadTaskData>("Load")
                    .Gate(ResourceKind.DiskIO, null, g => g.AddStep("Load", data =>
                    {
                        var loaded = GenesisCheckpointStore.LoadForRuntime(data.Path, _runtimeConfig);
                        _state.Replace(loaded.Config, loaded.Tokenizer, loaded.Model, loaded.PlatonicSpace, loaded.Conversation);
                        RestoreAutonomousHistory(loaded.AutonomousTraining);
                        PersistCheckpoint(
                            reason: "load",
                            detail: data.Path);
                    return data with { Loaded = true };
                    }))
                    .Run(out load)
                .DefineTask<GenesisConversationTaskData>("Conversation")
                    .AddStep("Observe", data =>
                    {
                        _state.Conversation.ObserveTurn("user", data.UserInput, resetSignal: data.ResetSignal, note: data.Note);
                        _state.Conversation.ObserveTurn("assistant", data.AssistantOutput, note: data.Note);
                        return data with
                        {
                            ContextBrief = _state.Conversation.BuildContextBrief(),
                            RecentTurnCount = _state.Conversation.RecentTurns.Count
                        };
                    })
                    .Gate(ResourceKind.DiskIO, null, g => g.AddStep("Persist", data =>
                    {
                        PersistCheckpoint(
                            reason: data.ResetSignal ? "conversation-reset" : "conversation",
                            detail: data.Note ?? data.UserInput);
                        return data;
                    }))
                    .Run(out conversation)
                .DefineTask<GenesisCompactConversationTaskData>("CompactConversation")
                    .AddStep("Compact", data =>
                    {
                        var compacted = _state.Conversation.Compact();
                        return data with
                        {
                            Compacted = compacted,
                            ContextBrief = _state.Conversation.BuildContextBrief(),
                            RecentTurnCount = _state.Conversation.RecentTurns.Count
                        };
                    })
                    .Gate(ResourceKind.DiskIO, null, g => g.AddStep("Persist", data =>
                    {
                        PersistCheckpoint(
                            reason: "conversation-compact",
                            detail: data.Note ?? "manual-compact");
                        return data;
                    }))
                    .Run(out compactConversation)
                .Build();

        TryBootstrapLatestState();
        EnsureConfiguredHiddenSize();
        InitializePermutationEngine();

        _trainPipeline = train;
        _trainOnePipeline = trainOne;
        _predictPipeline = predict;
        _savePipeline = save;
        _loadPipeline = load;
        _conversationPipeline = conversation;
        _compactConversationPipeline = compactConversation;
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
        Action<string>? uiLogger = null)
    {
        return await WithModelGateAsync(() => Task.Run(() =>
        {
            var planner = new GenesisAutonomousTrainingPlanner();
            var planningHistory = _autonomousHistory.ToList();
            var rounds = new List<GenesisAutonomousTrainingRound>();
            GenesisTrainingReport? lastReport = null;
            var unlimitedRounds = request.MaxRounds <= 0;
            var configuredRounds = Math.Max(1, request.MaxRounds);
            var roundLimitLabel = unlimitedRounds ? "∞" : configuredRounds.ToString();

            try
            {
                for (var round = 0; unlimitedRounds || round < configuredRounds; round++)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    var plan = planner.Suggest(request, planningHistory);
                    uiLogger?.Invoke(
                        $"[auto] round {round + 1}/{roundLimitLabel}: creator={plan.CreatorName} " +
                        $"pool={plan.SampleCount} train={plan.TrainCount} difficulty={plan.Difficulty} epochs={plan.Epochs} :: {plan.Reason}");

                    var creator = ExampleCreatorRegistry.All.FirstOrDefault(c =>
                        c.Name.Equals(plan.CreatorName, StringComparison.OrdinalIgnoreCase))
                        ?? throw new InvalidOperationException($"Creator not found: {plan.CreatorName}");

                    var candidatePool = creator.Generate(Math.Max(1, plan.SampleCount), plan.Difficulty, forTraining: true);
                    if (candidatePool.Length == 0)
                        throw new InvalidOperationException($"Creator produced no examples: {creator.Name}");

                    var trainCount = Math.Clamp(plan.TrainCount, 1, candidatePool.Length);
                    var startIndex = round % candidatePool.Length;
                    var examples = new List<GenesisExample>(trainCount);
                    for (var i = 0; i < trainCount; i++)
                    {
                        var selected = candidatePool[(startIndex + i) % candidatePool.Length];
                        examples.Add(new GenesisExample(selected.Input, selected.Output));
                    }

                    // Graceful stop: complete the current round safely, then stop before the next round.
                    var report = _state.Orchestrator.Train(examples, plan.Epochs, uiLogger, CancellationToken.None);
                    lastReport = report;

                    var currentLoss = report.AverageLoss.TokenLoss;
                    if (currentLoss < _bestTrainingLoss)
                    {
                        _bestTrainingLoss = currentLoss;
                        PersistCheckpoint(
                            reason: "auto-train-improved",
                            detail: $"creator={creator.Name} difficulty={plan.Difficulty} pool={plan.SampleCount} trained={examples.Count} loss={currentLoss:F4}",
                            exampleCount: examples.Count,
                            loss: currentLoss);
                    }

                    var roundResult = new GenesisAutonomousTrainingRound(
                        Round: round + 1,
                        CreatorName: creator.Name,
                        SampleCount: plan.SampleCount,
                        Difficulty: plan.Difficulty,
                        Epochs: plan.Epochs,
                        Report: report);
                    rounds.Add(roundResult);
                    planningHistory.Add(roundResult);
                    AppendAutonomousHistory(roundResult);
                }
            }
            finally
            {
                PersistCheckpoint(
                    reason: "auto-train-state",
                    detail: $"rounds={rounds.Count}",
                    exampleCount: rounds.Count,
                    loss: lastReport?.AverageLoss.TokenLoss);
            }

            return new GenesisAutonomousTrainingRun(request, rounds, lastReport);
        }, cancellationToken), cancellationToken);
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
        var examples = GenesisTrainingDataLoader.LoadFromFile(filePath);
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
        RestoreAutonomousHistory(loaded.AutonomousTraining);
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

    private GenesisNovaConfig CreateCheckpointConfig()
        => _state.Config with
        {
            HiddenSize = _state.Model.HiddenSize,
            AutoPersist = _runtimeConfig.AutoPersist,
            AutoResume = _runtimeConfig.AutoResume,
            AutoScaleVram = _runtimeConfig.AutoScaleVram,
            TargetVramUtilization = _runtimeConfig.TargetVramUtilization,
            ReserveVramMb = _runtimeConfig.ReserveVramMb,
            LocalStateDirectory = _runtimeConfig.LocalStateDirectory
        };

    private void PersistCheckpoint(
        string reason,
        string? explicitPath = null,
        string? detail = null,
        int? exampleCount = null,
        double? loss = null)
    {
        var snapshotConfig = CreateCheckpointConfig();
        var autoPath = GenesisLocalStateStore.ResolveCheckpointPath(_runtimeConfig);
        var wrote = false;

        if (!string.IsNullOrWhiteSpace(explicitPath))
        {
            GenesisCheckpointStore.Save(
                explicitPath,
                snapshotConfig,
                _state.Tokenizer,
                _state.Model,
                platonicSpace: _state.Memory.ExportSnapshot(),
                conversation: _state.Conversation.ExportSnapshot(),
                autonomousTraining: ExportAutonomousHistory());
            wrote = true;
        }

        if (_runtimeConfig.AutoPersist)
        {
            if (string.IsNullOrWhiteSpace(explicitPath) ||
                !string.Equals(explicitPath, autoPath, StringComparison.OrdinalIgnoreCase))
            {
                GenesisCheckpointStore.Save(
                    autoPath,
                    snapshotConfig,
                    _state.Tokenizer,
                    _state.Model,
                    platonicSpace: _state.Memory.ExportSnapshot(),
                    conversation: _state.Conversation.ExportSnapshot(),
                    autonomousTraining: ExportAutonomousHistory());
                wrote = true;
            }
        }

        if (wrote)
        {
            GenesisLocalStateStore.AppendJournalEntry(
                _runtimeConfig,
                reason,
                detail,
                exampleCount,
                loss);
        }
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

    /// <summary>Initialize permutation engine after model is ready.</summary>
    private void InitializePermutationEngine()
    {
        try
        {
            _permutationEngine = new AxiomaticPermutationEngine(_state.Model, _state.Tokenizer);
        }
        catch
        {
            // Non-critical, permutations optional
            _permutationEngine = null;
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

    private GenesisAutonomousTrainingSnapshot ExportAutonomousHistory()
        => new(_autonomousHistory.TakeLast(MaxAutonomousHistory).ToArray());

    private void RestoreAutonomousHistory(GenesisAutonomousTrainingSnapshot? snapshot)
    {
        _autonomousHistory.Clear();
        if (snapshot?.History is null || snapshot.History.Length == 0)
            return;

        foreach (var round in snapshot.History.TakeLast(MaxAutonomousHistory))
            _autonomousHistory.Add(round);
    }

    private void AppendAutonomousHistory(GenesisAutonomousTrainingRound round)
    {
        _autonomousHistory.Add(round);
        if (_autonomousHistory.Count <= MaxAutonomousHistory)
            return;

        var removeCount = _autonomousHistory.Count - MaxAutonomousHistory;
        _autonomousHistory.RemoveRange(0, removeCount);
    }

}
