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

public sealed partial class GenesisEvalAppRuntime
{
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
        => _ = await WithModelGateAsync(async () =>
        {
            await _savePipeline.RunAsync(new GenesisSaveTaskData(path));
            // CRITICAL: advance the reload-on-change watermark to the file we just wrote. RefreshLatestStateForReplPredict
            // reloads when the watched autosave file's write-time differs from this watermark — and AutoPersist writes
            // that file on every save. Without this, the gym's OWN autosave bumps the timestamp and the next probe
            // predict RELOADS the checkpoint we just wrote, tearing down + rebuilding model/space/trainer mid-run
            // (a lossy self-reload that reverts in-RAM progress, wipes conversation, and desyncs levels). Our own save
            // is never an "external change", so this is always correct.
            if (GenesisLocalStateStore.TryResolveBootstrapCheckpoint(_runtimeConfig, out var watched))
                TrackLoadedCheckpoint(watched);
            return true;
        });

    public async Task LoadAsync(string path)
        => _ = await WithModelGateAsync(async () =>
        {
            _ = await _loadPipeline.RunAsync(new GenesisLoadTaskData(path));
            TrackLoadedCheckpoint(path);
            return true;
        });

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
        // OFF by default: the in-process gym is the sole writer, so a "change" is always our OWN autosave — reloading
        // it tears down + rebuilds model/space/trainer mid-run (lossy → the 0%-on-resume / "model lost" bug). Only
        // when an EXTERNAL process writes the live checkpoint is this worth doing. See [[nova-save-reload-lifecycle]].
        if (!_runtimeConfig.WatchExternalCheckpoint)
            return;
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
        _state.Replace(loaded.Config, loaded.Tokenizer, loaded.Model, loaded.PlatonicSpace, loaded.Conversation, loaded.TrainerLearningStateJson, loaded.GrammarRoles, loaded.Navigator, loaded.NavigatorSelfField);
        _reloadCount++;
        _state.Inference.TalkEnabled = _conversationalMode; // Replace built a fresh engine — re-apply the session talk route
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
