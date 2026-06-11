using EvalApp.Consumer;
using GenesisNova.Train;

namespace GenesisNova.Runtime;

internal sealed class RunAutonomousTrainingRoundStep : IStep<GenesisAutonomousTrainTaskData>
{
    private readonly GenesisTrainingOrchestrator _orchestrator;

    public RunAutonomousTrainingRoundStep(GenesisTrainingOrchestrator orchestrator)
    {
        _orchestrator = orchestrator;
    }

    public ValueTask<GenesisAutonomousTrainTaskData> ExecuteAsync(GenesisAutonomousTrainTaskData data, CancellationToken ct)
    {
        var plan = data.Plan ?? throw new InvalidOperationException("Autonomous plan is required.");
        var examples = data.TrainingExamples ?? throw new InvalidOperationException("Training examples are required.");
        ct.ThrowIfCancellationRequested();
        data.UiLogger?.Invoke($"[auto] round {plan.Round}: starting train step (examples={examples.Count}, epochs={plan.Epochs})...");
        var report = _orchestrator.Train(examples, plan.Epochs, data.UiLogger, ct);
        var creatorRounds = (data.CreatorRounds ?? [])
            .Select(r => r with { Report = report })
            .ToArray();

        return ValueTask.FromResult(data with
        {
            CreatorRounds = creatorRounds,
            Report = report
        });
    }
}
