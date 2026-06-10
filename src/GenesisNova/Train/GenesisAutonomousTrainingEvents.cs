using GenesisNova.Runtime;
using System.Diagnostics;

namespace GenesisNova.Train;

/// <summary>
/// Implements event-based progress reporting for autonomous training rounds.
/// Provides real-time UI updates without blocking the pipeline.
/// </summary>
public class GenesisAutonomousTrainingEvents : IPipelineEvents<GenesisAutonomousTrainTaskData>
{
    private readonly Action<string>? _uiLogger;
    private readonly Action<GenesisAutonomousTrainingEventPayload>? _onRoundProgress;
    private readonly Stopwatch _roundTimer = Stopwatch.StartNew();

    public GenesisAutonomousTrainingEvents(
        Action<string>? uiLogger = null,
        Action<GenesisAutonomousTrainingEventPayload>? onRoundProgress = null)
    {
        _uiLogger = uiLogger;
        _onRoundProgress = onRoundProgress;
    }

    /// <summary>
    /// Called when a pipeline step starts execution.
    /// Fire-and-forget: doesn't block the pipeline.
    /// </summary>
    public ValueTask OnStepStartedAsync(string stepName, GenesisAutonomousTrainTaskData data)
    {
        _ = Task.Run(() =>
        {
            try
            {
                _roundTimer.Restart();
                var roundNum = data.RoundIndex + 1;
                var logMsg = $"[step-start] Round {roundNum}: {stepName}";
                _uiLogger?.Invoke(logMsg);
            }
            catch (Exception ex)
            {
                _uiLogger?.Invoke($"[events] error in OnStepStarted: {ex.Message}");
            }
        });

        return default;
    }

    /// <summary>
    /// Called when a pipeline step completes successfully.
    /// Extracts and fires structured round data for UI update.
    /// </summary>
    public ValueTask OnStepCompletedAsync(string stepName, GenesisAutonomousTrainTaskData data, long elapsedMs)
    {
        _ = Task.Run(() =>
        {
            try
            {
                var roundNum = data.RoundIndex + 1;
                var elapsedSec = elapsedMs / 1000.0;

                // Extract training metrics if available
                var loss = data.Report?.AverageLoss.TokenLoss ?? double.NaN;
                var success = data.Report?.ExampleSuccessRate ?? double.NaN;
                var samplesCount = data.TrainingExamples?.Count ?? 0;
                var datasetCount = data.Plan?.CreatorPlans.Count ?? 0;
                var promptAnswerCount = data.Report?.PromptAnswerExampleCount ?? 0;
                var windowedTextCount = data.Report?.WindowedTextExampleCount ?? 0;
                var skippedCorrectCount = data.Report?.SkippedCorrectExampleCount ?? 0;

                var msg = $"[step-complete] Round {roundNum}/{GetRoundLimit(data)}: {stepName} " +
                    $"datasets={datasetCount} samples={samplesCount} loss={loss:F4} success={success:P1} " +
                    $"prompt_answer={promptAnswerCount} windowed_text={windowedTextCount} skipped_correct={skippedCorrectCount} elapsed={elapsedSec:F1}s";
                _uiLogger?.Invoke(msg);

                // Fire event for UI dispatcher to handle
                var dataset = data.Request.PreferredCreator ?? "mixed";
                var payload = new GenesisAutonomousTrainingEventPayload(
                    Round: roundNum,
                    StepName: stepName,
                    Dataset: dataset,
                    Loss: loss,
                    ExampleSuccessRate: success,
                    SamplesTrained: samplesCount,
                    ElapsedMs: elapsedMs,
                    SkippedCorrectExampleCount: skippedCorrectCount,
                    PromptAnswerExampleCount: promptAnswerCount,
                    WindowedTextExampleCount: windowedTextCount,
                    CreatorSummary: string.Empty);

                _onRoundProgress?.Invoke(payload);
            }
            catch (Exception ex)
            {
                _uiLogger?.Invoke($"[events] error in OnStepCompleted: {ex.Message}");
            }
        });

        return default;
    }

    /// <summary>
    /// Called when a pipeline step encounters an error.
    /// Logs error details and notifies UI for graceful degradation.
    /// </summary>
    public ValueTask OnErrorAsync(string stepName, Exception exception, GenesisAutonomousTrainTaskData partialState)
    {
        _ = Task.Run(() =>
        {
            try
            {
                var roundNum = partialState.RoundIndex + 1;
                var errorMsg = $"[step-error] Round {roundNum}: {stepName} - {exception.GetType().Name}: {exception.Message}";
                _uiLogger?.Invoke(errorMsg);

                // Partial state preserved for potential recovery
                if (partialState.Report != null)
                {
                    _uiLogger?.Invoke($"[step-error] Partial report available: loss={partialState.Report.AverageLoss.TokenLoss:F4}");
                }
            }
            catch (Exception ex)
            {
                _uiLogger?.Invoke($"[events] error in OnError: {ex.Message}");
            }
        });

        return default;
    }

    /// <summary>
    /// Called to report overall progress percentage.
    /// Used for progress bar updates in the UI.
    /// </summary>
    public ValueTask OnProgressAsync(double percentage, string message)
    {
        _ = Task.Run(() =>
        {
            try
            {
                var clamped = Math.Clamp(percentage, 0, 100);
                var progressMsg = $"[progress] {clamped:F1}% - {message}";
                _uiLogger?.Invoke(progressMsg);
            }
            catch (Exception ex)
            {
                _uiLogger?.Invoke($"[events] error in OnProgress: {ex.Message}");
            }
        });

        return default;
    }

    private static string GetRoundLimit(GenesisAutonomousTrainTaskData data)
    {
        var unlimitedRounds = data.Request.MaxRounds <= 0;
        return unlimitedRounds ? "∞" : data.Request.MaxRounds.ToString();
    }
}

/// <summary>
/// Payload for round progress events fired to the UI.
/// </summary>
public sealed record GenesisAutonomousTrainingEventPayload(
    int Round,
    string StepName,
    string Dataset,
    double Loss,
    double ExampleSuccessRate,
    int SamplesTrained,
    long ElapsedMs,
    int SkippedCorrectExampleCount = 0,
    int PromptAnswerExampleCount = 0,
    int WindowedTextExampleCount = 0,
    string CreatorSummary = "");
