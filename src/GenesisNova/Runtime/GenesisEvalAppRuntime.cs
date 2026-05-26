using EvalApp.Consumer;
using GenesisNova.Core;
using GenesisNova.Data;
using GenesisNova.Infer;
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
    private readonly ICompiledPipeline<GenesisIntrospectTaskData> _introspectPipeline;
    private readonly ICompiledPipeline<GenesisRelateTaskData> _relatePipeline;
    private readonly ICompiledPipeline<GenesisConceptTaskData> _conceptPipeline;
    private readonly ICompiledPipeline<GenesisSaveTaskData> _savePipeline;
    private readonly ICompiledPipeline<GenesisLoadTaskData> _loadPipeline;
    private readonly ICompiledPipeline<GenesisConversationTaskData> _conversationPipeline;
    private readonly ICompiledPipeline<GenesisCompactConversationTaskData> _compactConversationPipeline;

    public GenesisEvalAppRuntime(GenesisNovaConfig? config = null)
    {
        _runtimeConfig = config ?? new GenesisNovaConfig();
        _state = new GenesisRuntimeState(_runtimeConfig);

        ICompiledPipeline<GenesisTrainTaskData> train = null!;
        ICompiledPipeline<GenesisTrainOneTaskData> trainOne = null!;
        ICompiledPipeline<GenesisPredictTaskData> predict = null!;
        ICompiledPipeline<GenesisIntrospectTaskData> introspect = null!;
        ICompiledPipeline<GenesisRelateTaskData> relate = null!;
        ICompiledPipeline<GenesisConceptTaskData> concept = null!;
        ICompiledPipeline<GenesisSaveTaskData> save = null!;
        ICompiledPipeline<GenesisLoadTaskData> load = null!;
        ICompiledPipeline<GenesisConversationTaskData> conversation = null!;
        ICompiledPipeline<GenesisCompactConversationTaskData> compactConversation = null!;

        Eval.App("GenesisNovaRuntime")
            .WithContext(NullGlobalContext.Instance)
            .WithResource(ResourceKind.Cpu, new TunableConfig(Min: 1, Max: 8, Default: 4))
            .WithResource(ResourceKind.DiskIO, new TunableConfig(Min: 1, Max: 2, Default: 1))
            .WithTuning()
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
                            logger = line =>
                            {
                                writer.WriteLine(line);
                                writer.Flush();
                            };
                        }

                        try
                        {
                            var report = _state.Orchestrator.Train(
                                data.Examples ?? [],
                                data.Epochs,
                                data.IntrospectionCyclesPerEpoch,
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
                        PersistCheckpoint(
                            reason: "train",
                            explicitPath: data.SavePath,
                            detail: $"epochs={data.Epochs}",
                            exampleCount: data.Examples?.Count ?? 0,
                            loss: data.Report?.AverageLoss.TotalLoss ?? 0.0,
                            queueDepth: _state.Trainer.QueueSize);
                        return data;
                    }))
                    .Run(out train)
                .DefineTask<GenesisTrainOneTaskData>("TrainOne")
                    .AddStep("TrainOneStep", data =>
                    {
                        var loss = _state.Trainer.TrainStep(data.Example);
                        return data with { Loss = loss, QueueDepth = _state.Trainer.QueueSize };
                    })
                    .Run(out trainOne)
                .DefineTask<GenesisPredictTaskData>("Predict")
                    .AddStep("Predict", data =>
                    {
                        var result = _state.Inference.Generate(new GenerationRequest(data.Input, data.MaxNewTokens));
                        return data with { Result = result };
                    })
                    .AddStep("BackgroundIntrospect", data =>
                    {
                        if (!data.EnableIntrospection)
                            return data with { IntrospectionProcessed = 0, QueueDepth = _state.Trainer.QueueSize };
                        var processed = _state.Trainer.RunIntrospectionCycles(1);
                        return data with { IntrospectionProcessed = processed, QueueDepth = _state.Trainer.QueueSize };
                    })
                    .Run(out predict)
                .DefineTask<GenesisIntrospectTaskData>("Introspect")
                    .AddStep("Introspect", data =>
                    {
                        var processed = _state.Trainer.RunIntrospectionCycles(Math.Max(1, data.Cycles));
                        return data with { Processed = processed, QueueDepth = _state.Trainer.QueueSize };
                    })
                    .Run(out introspect)
                .DefineTask<GenesisRelateTaskData>("Relate")
                    .AddStep("Relate", data =>
                    {
                        _state.Trainer.ObserveDirectContradiction(data.Left, data.Right, data.Contradiction);
                        return data with { QueueDepth = _state.Trainer.QueueSize };
                    })
                    .Run(out relate)
                .DefineTask<GenesisConceptTaskData>("Concept")
                    .AddStep("Describe", data => data with
                    {
                        Description = _state.Trainer.DescribeConcept(data.Concept)
                    })
                    .Run(out concept)
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
                        _state.Replace(loaded.Config, loaded.Tokenizer, loaded.Model, loaded.Cognition, loaded.Conversation);
                        PersistCheckpoint(
                            reason: "load",
                            detail: data.Path);
                        return data with { Loaded = true, Cognition = loaded.Cognition };
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
                            detail: data.Note ?? data.UserInput,
                            queueDepth: _state.Trainer.QueueSize);
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
                            detail: data.Note ?? "manual-compact",
                            queueDepth: _state.Trainer.QueueSize);
                        return data;
                    }))
                    .Run(out compactConversation)
                .Build();

        TryBootstrapLatestState();
        EnsureConfiguredHiddenSize();

        _trainPipeline = train;
        _trainOnePipeline = trainOne;
        _predictPipeline = predict;
        _introspectPipeline = introspect;
        _relatePipeline = relate;
        _conceptPipeline = concept;
        _savePipeline = save;
        _loadPipeline = load;
        _conversationPipeline = conversation;
        _compactConversationPipeline = compactConversation;
    }

    public async Task<GenesisTrainingReport> TrainAsync(
        string filePath,
        int epochs,
        int? introspectionCyclesPerEpoch = null,
        string? savePath = null,
        string? logPath = null)
    {
        var result = await _trainPipeline.RunAsync(new GenesisTrainTaskData(
            FilePath: filePath,
            Epochs: epochs,
            IntrospectionCyclesPerEpoch: introspectionCyclesPerEpoch,
            SavePath: savePath,
            LogPath: logPath));
        var data = ExtractData(result);
        return data.Report ?? throw new InvalidOperationException("Training report missing.");
    }

    public async Task<GenesisStepLoss> TrainOneAsync(GenesisExample example)
    {
        var result = await _trainOnePipeline.RunAsync(new GenesisTrainOneTaskData(example));
        var data = ExtractData(result);
        return data.Loss ?? throw new InvalidOperationException("Loss missing.");
    }

    public async Task<GenesisPredictTaskData> PredictAsync(string input, int maxTokens = 48, bool enableIntrospection = true)
    {
        var result = await _predictPipeline.RunAsync(new GenesisPredictTaskData(
            Input: input,
            MaxNewTokens: maxTokens,
            EnableIntrospection: enableIntrospection));
        return ExtractData(result);
    }

    public async Task<GenesisIntrospectTaskData> IntrospectAsync(int cycles)
    {
        var result = await _introspectPipeline.RunAsync(new GenesisIntrospectTaskData(cycles));
        return ExtractData(result);
    }

    public async Task RelateAsync(string left, string right, double contradiction)
    {
        await _relatePipeline.RunAsync(new GenesisRelateTaskData(left, right, contradiction));
    }

    public async Task<string> DescribeConceptAsync(string concept)
    {
        var result = await _conceptPipeline.RunAsync(new GenesisConceptTaskData(concept));
        return ExtractData(result).Description;
    }

    public async Task SaveAsync(string path)
        => _ = await _savePipeline.RunAsync(new GenesisSaveTaskData(path));

    public async Task LoadAsync(string path)
        => _ = await _loadPipeline.RunAsync(new GenesisLoadTaskData(path));

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
            var pred = await PredictAsync(ex.Input, maxTokens: 48, enableIntrospection: false);
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

    public int QueueSize => _state.Trainer.QueueSize;
    public int VocabularySize => _state.Tokenizer.VocabularySize;
    public int HiddenSize => _state.Model.HiddenSize;
    public string ConversationBrief => _state.Conversation.BuildContextBrief();

    private void TryBootstrapLatestState()
    {
        if (!_runtimeConfig.AutoResume)
            return;

        if (!GenesisLocalStateStore.TryResolveBootstrapCheckpoint(_runtimeConfig, out var path) || !File.Exists(path))
            return;

        var loaded = GenesisCheckpointStore.LoadForRuntime(path, _runtimeConfig);
        _state.Replace(loaded.Config, loaded.Tokenizer, loaded.Model, loaded.Cognition, loaded.Conversation);
        GenesisLocalStateStore.AppendJournalEntry(
            _runtimeConfig,
            "bootstrap",
            detail: path,
            queueDepth: _state.Trainer.QueueSize);
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
        double? loss = null,
        int? queueDepth = null)
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
                _state.Trainer.ExportCognitionSnapshot(),
                _state.Conversation.ExportSnapshot());
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
                    _state.Trainer.ExportCognitionSnapshot(),
                    _state.Conversation.ExportSnapshot());
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
                loss,
                queueDepth ?? _state.Trainer.QueueSize);
        }
    }

    private static T ExtractData<T>(PipelineResult<T> result)
        => result switch
        {
            PipelineResult<T>.Success s => s.Data,
            PipelineResult<T>.Failure f => throw new InvalidOperationException(f.Message ?? "Pipeline failed.", f.Exception),
            _ => throw new InvalidOperationException("Unsupported pipeline result.")
        };
}
